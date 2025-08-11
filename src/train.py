# SPDX-License-Identifier: MIT
# --- allocator tuning BEFORE torch import (avoid allocator bug) -------------
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
os.environ.setdefault("TORCHINDUCTOR_CUDAGRAPHS", "0")
os.environ.setdefault("TORCHINDUCTOR_CUDAGRAPH_TREES", "0")

import yaml
import time
import multiprocessing
import logging
import json
import csv
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# quiet noisy but harmless warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io.image")
warnings.filterwarnings("ignore", message=".*'has_cuda' is deprecated.*")
warnings.filterwarnings("ignore", message=".*'has_cudnn' is deprecated.*")
warnings.filterwarnings("ignore", message=".*'has_mps' is deprecated.*")
warnings.filterwarnings("ignore", message=".*'has_mkldnn' is deprecated.*")

import torch._dynamo
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors  = True
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)

# extra belt-and-suspenders: make Inductor avoid cudagraphs where possible
try:
    import torch._inductor.config as _ind_cfg
    if hasattr(_ind_cfg, "triton") and hasattr(_ind_cfg.triton, "cudagraphs"):
        _ind_cfg.triton.cudagraphs = False
    if hasattr(_ind_cfg, "cuda") and hasattr(_ind_cfg.cuda, "cudagraphs"):
        _ind_cfg.cuda.cudagraphs = False
    if hasattr(_ind_cfg, "cudagraph_trees"):
        _ind_cfg.cudagraph_trees = False
except Exception:
    pass

# Prefer Flash/Mem-efficient SDPA for MultiheadAttention when possible
try:
    from torch.backends.cuda import sdp_kernel
    sdp_kernel.enable_flash(True)
    sdp_kernel.enable_mem_efficient(True)
    sdp_kernel.enable_math(False)
except Exception:
    pass

from dataset import RadioMLDataset
from model import DenoiserAutoEnc, MultiBranchGNNClassifier as GNNModClassifier

# ─────────── speed flags ─────────────────────────────────────────────────────
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


# ─────────── CUDA prefetcher ─────────────────────────────────────────────────
class CUDAPrefetcher:
    def __init__(self, loader, device, keys=("tm","spec","cwt","snr","y"), use_channels_last=True):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream() if device.type == "cuda" else None
        self.keys   = keys
        self.use_channels_last = use_channels_last
        self._preload()

    def _move_to(self, t: torch.Tensor):
        t = t.to(self.device, non_blocking=True)
        if self.use_channels_last and t.ndim == 4:
            t = t.contiguous(memory_format=torch.channels_last)
        return t

    def _preload(self):
        try:
            self._next = next(self.loader)
        except StopIteration:
            self._next = None
            return
        if self.stream is None:
            return
        with torch.cuda.stream(self.stream):
            for k in self.keys:
                v = self._next.get(k, None)
                if isinstance(v, torch.Tensor):
                    self._next[k] = self._move_to(v)

    def __iter__(self): return self
    def __next__(self):
        if self._next is None:
            raise StopIteration
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        batch = self._next
        self._preload()
        return batch


def _tqdm(iterable, total, desc, leave=False, position=0):
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=leave, position=position)


# ───────── config loader ─────────────────────────────────────────────────────
def _f(x, d=0.0): return d if x is None else float(x)

def load_cfg():
    with open("configs/config.yaml","r") as f:
        cfg = yaml.safe_load(f)
    tr = cfg["training"]
    # ints
    for k in ("batch_size","test_batch_size","epochs"):
        tr[k] = int(tr[k])
    # floats
    for k in ("lr","weight_decay","focal_gamma","label_smoothing",
              "grad_clip_norm","snr_loss_weight_low","snr_loss_weight_high"):
        tr[k] = _f(tr.get(k),1.0)
    # toggles / knobs
    tr.setdefault("use_autoenc",  True)
    tr.setdefault("use_compile",  False)
    tr.setdefault("compile_mode", "reduce-overhead")
    tr.setdefault("compile_pyg",  False)
    tr.setdefault("channels_last", True)
    tr.setdefault("adamw_fused", False)

    # capping + micro-batch
    tr["max_samples_per_epoch"] = int(tr.get("max_samples_per_epoch", 0))
    tr["val_max_batches"]       = int(tr.get("val_max_batches", 0))
    tr["micro_batch_size"]      = int(tr.get("micro_batch_size", 0))

    # workers
    max_w = min(8, max(1, multiprocessing.cpu_count()-1))
    tr["num_workers"]        = int(tr.get("num_workers", max_w))
    tr["persistent_workers"] = tr["num_workers"] > 0
    tr["prefetch_factor"]    = max(2, int(tr.get("prefetch_factor", 2)))

    # scheduler
    sc = tr.setdefault("scheduler",{"type":"cosine"})
    sc["type"] = sc["type"].lower()
    if sc["type"]=="onecycle":
        for k in ("max_lr","pct_start","div_factor","final_div_factor"):
            sc[k] = _f(sc.get(k),0.0)
        sc["anneal_strategy"] = sc.get("anneal_strategy","cosine").lower()
    else:
        sc["T_max"]   = int(sc.get("T_max",tr["epochs"]))
        sc["eta_min"] = _f(sc.get("eta_min",0.0))

    # AE settings
    ae = cfg.setdefault("autoenc", {})
    ae.setdefault("noise_std",          0.01)
    ae.setdefault("use_amp",            True)
    ae.setdefault("snr_low",           -20.0)
    ae.setdefault("snr_high",           10.0)
    ae.setdefault("epochs",             2)
    ae.setdefault("batch_size",         min(2048, max(512, tr["batch_size"])))
    ae.setdefault("num_workers",        min(4, tr["num_workers"]))
    ae.setdefault("pin_memory",         True)
    ae.setdefault("prefetch_factor",    2)
    ae.setdefault("persistent_workers", False)
    ae.setdefault("max_steps",          600)
    ae.setdefault("max_minutes",        10)
    ae.setdefault("use_compile",        tr.get("use_compile", False))
    ae.setdefault("compile_mode",       tr.get("compile_mode", "reduce-overhead"))
    return cfg


# ─── optimizer + scheduler ───────────────────────────────────────────────────
def build_optim_and_sched(net, tr, steps_per_epoch, device):
    fused_flag = bool(tr.get("adamw_fused", False))
    if fused_flag and tr.get("channels_last", True):
        print("⚠️  Disabling fused AdamW because channels_last is enabled (mixed layouts).")
        fused_flag = False

    opt_kwargs = dict(lr=tr["lr"], weight_decay=tr["weight_decay"])
    try:
        if fused_flag:
            opt = optim.AdamW(net.parameters(), **opt_kwargs, fused=True)
        else:
            opt = optim.AdamW(net.parameters(), **opt_kwargs)
    except TypeError:
        opt = optim.AdamW(net.parameters(), **opt_kwargs)

    sc  = tr["scheduler"]
    if sc["type"]=="onecycle":
        total = steps_per_epoch*tr["epochs"]
        sched = optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=sc["max_lr"],
            total_steps=total,
            pct_start=sc["pct_start"],
            div_factor=sc["div_factor"],
            final_div_factor=sc["final_div_factor"],
            anneal_strategy="cos" if sc["anneal_strategy"].startswith("cos") else "linear"
        )
        per_batch=True
    else:
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=sc["T_max"], eta_min=sc["eta_min"])
        per_batch=False
    return opt, sched, per_batch


# ─── metrics helpers ─────────────────────────────────────────────────────────
def update_topk_counters(logits, labels, topk_counts, total):
    with torch.no_grad():
        float_logits = logits.float()
        B = labels.size(0)
        _, pred = float_logits.topk(5,dim=1)
        for k in (1,3,5):
            correct_k = (pred[:,:k]==labels.unsqueeze(1)).any(dim=1).sum().item()
            topk_counts[f"top{k}"] += correct_k
        total[0] += B

def update_confusion(conf_all, conf_bins, bins, edges, preds, trues, snrs):
    for p,t in zip(preds, trues):
        conf_all[t,p] += 1
    for p,t,s in zip(preds, trues, snrs):
        idx = np.searchsorted(edges, s, side="right")-1
        idx = max(0,min(idx,len(bins)-1))
        conf_bins[bins[idx]][t,p] += 1

def compute_f1_from_conf(conf):
    C = conf.shape[0]
    f1s=[]
    for c in range(C):
        tp = conf[c,c]
        fp = conf[:,c].sum() - tp
        fn = conf[c,:].sum() - tp
        if tp+fp>0 and tp+fn>0:
            precision=tp/(tp+fp)
            recall   =tp/(tp+fn)
            den = (precision + recall)
            f1 = 0.0 if den == 0 else (2*precision*recall/den)
        else:
            f1=0.0
        f1s.append(float(f1))
    return f1s


# ─── label/shape sanity checks ───────────────────────────────────────────────
def _ensure_targets_ok(y: torch.Tensor, num_classes: int, where: str):
    y_min = int(y.min().item())
    y_max = int(y.max().item())
    if y_min < 0 or y_max >= num_classes:
        bad = y[(y < 0) | (y >= num_classes)]
        sample = bad[:10].tolist()
        raise RuntimeError(
            f"[{where}] Target index out of range: min={y_min}, max={y_max}, "
            f"num_classes={num_classes}. Offending examples (up to 10): {sample}"
        )

def _ensure_logits_ok(logits: torch.Tensor, num_classes: int, where: str):
    C = logits.size(1)
    if C != num_classes:
        raise RuntimeError(
            f"[{where}] Logits shape mismatch: logits.size(1)={C} vs num_classes={num_classes}. "
            "Check model head and config."
        )


# ─── micro-batch loops ───────────────────────────────────────────────────────
def _chunk_slices(B, micro_bs):
    if micro_bs <= 0 or micro_bs >= B:
        yield slice(0, B); return
    for start in range(0, B, micro_bs):
        yield slice(start, min(B, start+micro_bs))

class CompilePoolError(RuntimeError):
    pass

def _is_compile_capture_err(msg_lower: str) -> bool:
    return (
        ("could not find 0x" in msg_lower) or
        ("cudagraph" in msg_lower) or
        ("stream is capturing" in msg_lower) or
        ("during capture" in msg_lower) or
        ("capture_end" in msg_lower)
    )

def train_step_with_microbatch(net, batch, tr, device, opt, scaler, num_classes: int, sched=None, per_batch: bool=False):
    tm = batch["tm"]; bsp = batch["spec"]; cwt = batch["cwt"]; snr = batch["snr"]; y = batch["y"]
    B  = y.size(0)
    micro_bs = int(tr.get("micro_batch_size", 0))
    while True:
        try:
            opt.zero_grad(set_to_none=True)
            total_loss = total_correct = total_seen = 0
            for sl in _chunk_slices(B, micro_bs):
                tm_s, bsp_s, cwt_s, snr_s, y_s = tm[sl], bsp[sl], cwt[sl], snr[sl], y[sl]
                _ensure_targets_ok(y_s, num_classes, "train")
                with autocast():
                    logits = net(tm_s, bsp_s, cwt_s, snr_s)
                    _ensure_logits_ok(logits, num_classes, "train")
                    ce = F.cross_entropy(logits, y_s, reduction="none",
                                         label_smoothing=tr["label_smoothing"])
                    mask = ((snr_s<0).float()*tr["snr_loss_weight_low"]
                            + (snr_s>=0).float()*tr["snr_loss_weight_high"])
                    if tr["focal_gamma"]>0:
                        pt   = torch.exp(-ce)
                        loss = ((1-pt)**tr["focal_gamma"]*ce)*mask
                    else:
                        loss = ce*mask
                    loss = loss.mean()
                scaler.scale(loss).backward()
                total_loss   += loss.detach().item() * y_s.size(0)
                total_seen   += y_s.size(0)
                total_correct+= (logits.argmax(1)==y_s).sum().item()
            if tr["grad_clip_norm"]>0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), tr["grad_clip_norm"])
            scaler.step(opt); scaler.update()
            if per_batch and (sched is not None):
                # step AFTER optimizer.step() to avoid the LR warning
                sched.step()
            return total_loss/max(1,total_seen), total_correct, total_seen, micro_bs
        except RuntimeError as e:
            msg = str(e).lower()
            if _is_compile_capture_err(msg):
                raise CompilePoolError(e)
            if "out of memory" not in msg:
                raise
            torch.cuda.empty_cache()
            micro_bs = max(1, (micro_bs or (B//2)) // 2)
            if micro_bs == 1: raise

def val_step_with_microbatch(net, batch, tr, num_classes: int):
    tm = batch["tm"]; bsp = batch["spec"]; cwt = batch["cwt"]; snr = batch["snr"]; y = batch["y"]
    B  = y.size(0)
    micro_bs = int(tr.get("micro_batch_size", 0))
    preds_all = []
    with torch.inference_mode():
        with autocast():
            for sl in _chunk_slices(B, micro_bs):
                y_s = y[sl]
                _ensure_targets_ok(y_s, num_classes, "val")
                logits = net(tm[sl], bsp[sl], cwt[sl], snr[sl]).detach().clone()
                _ensure_logits_ok(logits, num_classes, "val")
                preds_all.append(logits)
    return torch.cat(preds_all, dim=0)


# ─── tm-only AE dataset (fast pretrain) ──────────────────────────────────────
def _snr_batch_tm(x4: np.ndarray) -> np.ndarray:
    I, Q = x4[..., 0], x4[..., 1]
    power = (I*I + Q*Q).mean(axis=1)
    Imean = I.mean(axis=1, keepdims=True)
    Qmean = Q.mean(axis=1, keepdims=True)
    noise = ((I - Imean)**2).mean(axis=1) + ((Q - Qmean)**2).mean(axis=1) + 1e-12
    return 10.0 * np.log10(power / noise)

class AEDenoiseDataset(Dataset):
    def __init__(self, processed_dir, split, snr_min, snr_max, max_samples=1_000_000, scan_chunk=8192):
        base = os.path.join(processed_dir, split)
        shards = sorted(int(f.split("_")[-1].split(".")[0]) for f in os.listdir(base) if f.startswith("y_"))
        self.mems = [np.load(f"{base}/X_tm_{s}.npy", mmap_mode="r") for s in shards]
        rng = np.random.default_rng(1234)
        order = list(range(len(self.mems))); rng.shuffle(order)
        self.idxs = []
        need = max_samples if max_samples and max_samples>0 else float("inf")
        for si in order:
            tm = self.mems[si]; N = len(tm)
            for st in range(0, N, scan_chunk):
                ed = min(st+scan_chunk, N)
                x  = np.asarray(tm[st:ed], dtype=np.float32)
                snr = _snr_batch_tm(x)
                m = (snr >= snr_min) & (snr <= snr_max)
                js = np.nonzero(m)[0]
                for j in js:
                    self.idxs.append((si, st+j))
                    if len(self.idxs) >= need: break
                if len(self.idxs) >= need: break
            if len(self.idxs) >= need: break
        if not self.idxs:
            raise RuntimeError("No AE samples found in the requested SNR range.")
    def __len__(self): return len(self.idxs)
    def __getitem__(self, i):
        si, j = self.idxs[i]
        tm = np.array(self.mems[si][j], dtype=np.float32, copy=True)
        return {"tm": torch.from_numpy(tm)}


# ─── helpers to robustly fall back to eager ──────────────────────────────────
def _extract_state_dict_from_any(model):
    """Works with nn.Module, DataParallel(nn.Module), and compiled wrappers."""
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    # compiled wrappers still expose state_dict; fall back to _orig_mod if present
    try:
        return model.state_dict()
    except Exception:
        base = getattr(model, "_orig_mod", model)
        return base.state_dict()

def _rebuild_eager_model_from(net_or_dp, cfg, device, channels_last=True):
    """Create a fresh eager model and load weights from an arbitrary (possibly compiled/DP) model."""
    sd = _extract_state_dict_from_any(net_or_dp)
    fresh = GNNModClassifier(cfg).to(device)
    if channels_last:
        fresh = fresh.to(memory_format=torch.channels_last)
    missing, unexpected = fresh.load_state_dict(sd, strict=False)
    if missing or unexpected:
        # Non-fatal; print once to help debugging
        print(f"ℹ️ eager reload: missing={len(missing)}, unexpected={len(unexpected)}")
    if torch.cuda.device_count()>1:
        fresh = torch.nn.DataParallel(fresh)
    return fresh


# ─── main ────────────────────────────────────────────────────────────────────
def main():
    print("🚀  starting training …")
    C      = load_cfg()
    tr     = C["training"]
    ae_cfg = C["autoenc"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(tr["save_dir"], exist_ok=True)

    # ── optional fast AE pretrain ────────────────────────────────────────────
    autoenc=None
    if tr["use_autoenc"]:
        print(f"🔧  pretraining AE ({int(ae_cfg['snr_low'])}–{int(ae_cfg['snr_high'])} dB)…")
        max_samples = int(ae_cfg.get("max_steps", 0)) * int(ae_cfg["batch_size"]) if int(ae_cfg.get("max_steps", 0))>0 else 1_000_000
        ae_ds = AEDenoiseDataset(
            C["data"]["processed_dir"], "train",
            snr_min=float(ae_cfg["snr_low"]), snr_max=float(ae_cfg["snr_high"]),
            max_samples=max_samples
        )
        ae_loader = DataLoader(
            ae_ds,
            batch_size=int(ae_cfg["batch_size"]),
            shuffle=True, drop_last=True, pin_memory=True,
            num_workers=int(ae_cfg["num_workers"]),
            persistent_workers=bool(ae_cfg["persistent_workers"]),
            prefetch_factor=int(ae_cfg["prefetch_factor"]) if int(ae_cfg["num_workers"])>0 else None
        )
        autoenc=DenoiserAutoEnc(in_ch=4,hidden=32).to(device)
        if ae_cfg.get("use_compile", False) and hasattr(torch, "compile"):
            try:
                autoenc = torch.compile(autoenc, mode=ae_cfg.get("compile_mode","reduce-overhead"),
                                        fullgraph=False, dynamic=True)
            except Exception as e:
                print(f"⚠️ torch.compile (AE) skipped: {e}")
        ae_opt=optim.AdamW(autoenc.parameters(),lr=1e-3,weight_decay=1e-5)
        scaler=GradScaler(enabled=bool(ae_cfg["use_amp"]))
        autoenc.train()
        start_wall = time.perf_counter()
        max_minutes = float(ae_cfg.get("max_minutes", 0) or 0)
        for ep in range(1, int(ae_cfg["epochs"]) + 1):
            epoch_loss = 0.0; seen = 0
            pbar = _tqdm(ae_loader, total=len(ae_loader), desc=f"AE ep{ep:02d}", leave=False)
            for step, batch in enumerate(pbar):
                if max_minutes > 0:
                    elapsed = (time.perf_counter() - start_wall) / 60.0
                    if elapsed >= max_minutes:
                        print(f"⏱️  AE time cap reached ({elapsed:.1f} min ≥ {max_minutes} min). Stopping AE pretrain early.")
                        break
                tm = batch["tm"].to(device, non_blocking=True)
                noise   = torch.randn_like(tm) * float(ae_cfg["noise_std"])
                corrupt = tm + noise
                with autocast(enabled=bool(ae_cfg["use_amp"])):
                    recon = autoenc(corrupt)
                    loss  = F.mse_loss(recon, tm)
                ae_opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(ae_opt); scaler.update()
                bs = tm.size(0); epoch_loss += loss.item() * bs; seen += bs
                if int(ae_cfg.get("max_steps", 0))>0 and (step+1) >= int(ae_cfg["max_steps"]):
                    break
            print(f" AE ep{ep:02d} ➜ mse {epoch_loss/max(1,seen):.8f}")
            if max_minutes > 0 and (time.perf_counter() - start_wall) / 60.0 >= max_minutes:
                break
        for p in autoenc.parameters(): p.requires_grad=False
        autoenc.eval()

    # ── model: COMPILE BEFORE DataParallel ───────────────────────────────────
    base_raw = GNNModClassifier(C).to(device)
    if tr.get("channels_last", True):
        base_raw = base_raw.to(memory_format=torch.channels_last)

    base = base_raw
    if tr.get("compile_pyg", False):
        try:
            from torch_geometric.compile import compile as pyg_compile  # PyG >= 2.5
            base = pyg_compile(base)
            print("✅ torch_geometric.compile enabled.")
        except Exception as e:
            print(f"⚠️ torch_geometric.compile skipped: {e}")

    if autoenc:
        ae_src = getattr(autoenc, "_orig_mod", autoenc)
        base.autoenc.load_state_dict(ae_src.state_dict(), strict=False)
        for p in base.autoenc.parameters():
            p.requires_grad = False
        base.autoenc.eval()

    compiled_used = False
    if tr.get("use_compile", False) and hasattr(torch,"compile"):
        try:
            base = torch.compile(
                base,
                mode=tr.get("compile_mode","reduce-overhead"),
                fullgraph=False,
                dynamic=True
            )
            compiled_used = True
            print("✅ torch.compile enabled.")
        except Exception as e:
            print(f"⚠️ torch.compile skipped: {e}")
            compiled_used = False

    if torch.cuda.device_count()>1:
        print("ℹ️ Using DataParallel. (No change to training math.)")
        net = torch.nn.DataParallel(base)
    else:
        net = base

    # ── static validation loader ─────────────────────────────────────────────
    test_ds  = RadioMLDataset(C["data"]["processed_dir"],"test",
        augment=False, seg_cfg=C["segmentation"], aug_cfg=C["augmentation"],
        min_snr_db=None, max_snr_db=None)
    tr = C["training"]  # refresh ref
    dl_va=DataLoader(test_ds,
        batch_size=tr["test_batch_size"],shuffle=False,
        pin_memory=True,num_workers=max(0, tr["num_workers"]//2),
        persistent_workers=tr["persistent_workers"],
        prefetch_factor=max(2, tr["prefetch_factor"]//2))

    # writers
    csv_path=Path(tr["save_dir"])/"epoch_metrics.csv"
    csv_f=open(csv_path,"w",newline="")
    csv_w=None
    jsonl_path=Path(tr["save_dir"])/"epoch_metrics.jsonl"

    # bins
    edges=[-20,-10,0,10,20,30]
    bins=["[-20,-10)","[-10,0)","[0,10)","[10,20)","[20,30]"]

    def make_prefetch(loader):
        return CUDAPrefetcher(loader, device, use_channels_last=tr.get("channels_last", True)) \
               if device.type=="cuda" else iter(loader)

    num_classes = int(C["model"]["num_classes"])

    # preflight labels early
    try:
        tmp_loader = DataLoader(
            RadioMLDataset(C["data"]["processed_dir"],"train",
                           augment=False, seg_cfg=C["segmentation"], aug_cfg=C["augmentation"],
                           min_snr_db=None, max_snr_db=None),
            batch_size=8192, shuffle=False
        )
        tmp_batch = next(iter(tmp_loader))
        _ensure_targets_ok(tmp_batch["y"], num_classes, "preflight(train_ds)")
        del tmp_loader, tmp_batch
    except Exception as e:
        raise RuntimeError(f"Dataset label preflight failed: {e}")

    # curriculum-aware train loader builder
    cur_cfg = C.get("curriculum", {"enabled": False})

    def _clamp01(x: float) -> float:
        return max(0.0, min(1.0, x))

    def build_train_loader_for_epoch(ep: int):
        if cur_cfg.get("enabled", False):
            start = float(cur_cfg.get("start_snr_db", -20.0))
            end   = float(cur_cfg.get("end_snr_db",   10.0))
            pace  = max(1, int(cur_cfg.get("pace_epochs", 5)))
            prog  = _clamp01((ep-1)/(pace-1)) if pace > 1 else 1.0
            curr_low = start + prog * (end - start)
            ds = RadioMLDataset(C["data"]["processed_dir"],"train",
                augment=True, seg_cfg=C["segmentation"], aug_cfg=C["augmentation"],
                min_snr_db=curr_low, max_snr_db=None)
            note = f"SNR≥{curr_low:.1f}dB"
        else:
            ds = RadioMLDataset(C["data"]["processed_dir"],"train",
                augment=True, seg_cfg=C["segmentation"], aug_cfg=C["augmentation"],
                min_snr_db=None, max_snr_db=None)
            note = "no-curriculum"

        if len(ds) == 0:
            note += " → fallback(all SNRs)"
            ds = RadioMLDataset(C["data"]["processed_dir"],"train",
                augment=True, seg_cfg=C["segmentation"], aug_cfg=C["augmentation"],
                min_snr_db=None, max_snr_db=None)

        if len(ds) == 0:
            print("⚠️  Train dataset is empty even after fallback. Skipping this epoch.")
            dl = DataLoader(
                ds,
                batch_size=tr["batch_size"],
                shuffle=False,
                sampler=SequentialSampler(ds),
                drop_last=False,
                pin_memory=True, num_workers=0
            )
            return ds, dl, note + " (empty)"

        sampler = None
        if tr["max_samples_per_epoch"] > 0:
            sampler = RandomSampler(ds, replacement=True, num_samples=tr["max_samples_per_epoch"])
        dl = DataLoader(
            ds,
            batch_size=tr["batch_size"],
            shuffle=(sampler is None), sampler=sampler, drop_last=True,
            pin_memory=True, num_workers=tr["num_workers"],
            persistent_workers=tr["persistent_workers"],
            prefetch_factor=tr["prefetch_factor"],
        )
        return ds, dl, note

    opt=sched=per_batch=None
    scaler=GradScaler()
    best_full=0.0

    for ep in range(1,tr["epochs"]+1):
        train_ds, dl_tr, cur_note = build_train_loader_for_epoch(ep)
        print(f"📚 epoch {ep:02d} curriculum: {cur_note} · train_samples={len(train_ds)}")

        if opt is None or sched is None:
            opt, sched, per_batch = build_optim_and_sched(net, tr, len(dl_tr), device)

        # ── train ─────────────────────────────────────────────────────────────
        net.train()
        tot_loss=tot_samples=correct=0
        prefetch = make_prefetch(dl_tr)
        pbar = _tqdm(prefetch, total=len(dl_tr), desc=f"train {ep:02d}", leave=False, position=0)
        for batch in pbar:
            if device.type != "cuda":
                for k in ("tm","spec","cwt","snr","y"):
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(device, non_blocking=True)
            try:
                loss_b, corr_b, seen_b, used_micro = train_step_with_microbatch(
                    net, batch, tr, device, opt, scaler, num_classes, sched=sched, per_batch=per_batch
                )
            except CompilePoolError:
                # robust fallback: rebuild a fresh eager model from state_dict
                print("⚠️  TorchInductor cudagraph/capture error detected. Falling back to eager model…")
                net = _rebuild_eager_model_from(net, C, device, channels_last=tr.get("channels_last", True))
                # disable compile for the rest of the run
                tr["use_compile"] = False
                opt, sched, per_batch = build_optim_and_sched(net, tr, len(dl_tr), device)
                scaler = GradScaler()
                # retry once on eager
                loss_b, corr_b, seen_b, used_micro = train_step_with_microbatch(
                    net, batch, tr, device, opt, scaler, num_classes, sched=sched, per_batch=per_batch
                )

            tot_loss+=loss_b*seen_b; tot_samples+=seen_b; correct += corr_b
            if used_micro and used_micro > 0:
                pbar.set_postfix(micro_bs=used_micro)

        if not per_batch and hasattr(sched, "step"):
            sched.step()

        train_loss=tot_loss/max(1,tot_samples)
        train_acc =100*correct/max(1,tot_samples)
        print(f"epoch {ep:02d} ➜ loss {train_loss:.4f} · train acc {train_acc:.2f}%")

        # ── validate ─────────────────────────────────────────────────────────
        net.eval()
        masked_corr=masked_tot=0
        hi_corr=hi_tot=0
        topk_counts={"top1":0,"top3":0,"top5":0}
        total_count=[0]
        Cn=num_classes
        conf_all=np.zeros((Cn,Cn),dtype=int)
        conf_bins={b:np.zeros((Cn,Cn),dtype=int) for b in bins}

        prefetch_val = make_prefetch(dl_va)
        max_val_batches = tr["val_max_batches"] if tr["val_max_batches"]>0 else len(dl_va)
        bcount = 0
        for batch in _tqdm(prefetch_val, total=max_val_batches, desc=f"val   {ep:02d}", leave=False, position=1):
            if device.type != "cuda":
                for k in ("tm","spec","cwt","snr","y"):
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(device, non_blocking=True)

            logits = val_step_with_microbatch(net, batch, tr, num_classes)
            preds  = logits.argmax(1)
            y      = batch["y"]; snr = batch["snr"]

            m=(snr>=0)
            if m.any():
                masked_corr+=(preds[m]==y[m]).sum().item()
                masked_tot += m.sum().item()

            mh=(snr>=10) & (snr<=30)
            if mh.any():
                hi_corr+=(preds[mh]==y[mh]).sum().item()
                hi_tot += mh.sum().item()

            update_topk_counters(logits, y, topk_counts, total_count)
            p_np=preds.cpu().numpy(); t_np=y.cpu().numpy(); s_np=snr.cpu().numpy()
            update_confusion(conf_all,conf_bins,bins,edges,p_np,t_np,s_np)

            bcount += 1
            if bcount >= max_val_batches: break

        masked_val_acc = 100.0 * masked_corr / max(1, masked_tot)
        val_acc_10_30  = 100.0 * hi_corr     / max(1, hi_tot)

        topk = {k: 100.*v/total_count[0] for k,v in topk_counts.items()}
        class_f1_all = compute_f1_from_conf(conf_all)
        macro_f1_all = float(np.mean(class_f1_all))
        bin_f1 = {b: float(np.mean(compute_f1_from_conf(conf_bins[b]))) for b in bins}
        bin_acc= {b: 100.*conf_bins[b].trace()/conf_bins[b].sum() if conf_bins[b].sum()>0 else 0.0 for b in bins}

        if csv_w is None:
            cols = [
                "epoch","train_loss","train_acc","masked_val_acc","val_acc_10_30",
                "top1","top3","top5","macro_f1"
            ]
            for i in range(Cn):
                cols.append(f"class_{i}_f1")
            for b in bins:
                cols.append(f"snr_{b}_acc")
                cols.append(f"snr_{b}_macro_f1")
            csv_w=csv.DictWriter(csv_f,fieldnames=cols)
            csv_w.writeheader()

        print(f"val {ep:02d} ➜ acc[10–30 dB] {val_acc_10_30:.2f}%")

        flat={
            "epoch":ep,
            "train_loss":train_loss,
            "train_acc":train_acc,
            "masked_val_acc":masked_val_acc,
            "val_acc_10_30":val_acc_10_30,
            **topk,
            "macro_f1":macro_f1_all
        }
        for i,f1 in enumerate(class_f1_all):
            flat[f"class_{i}_f1"]=f1
        for b in bins:
            flat[f"snr_{b}_acc"]=bin_acc[b]
            flat[f"snr_{b}_macro_f1"]=bin_f1[b]

        csv_w.writerow(flat); csv_f.flush()
        with open(jsonl_path,"a") as jf: jf.write(json.dumps({
            **flat, "class_f1": class_f1_all, "snr_bin_acc": bin_acc, "snr_bin_macro_f1": bin_f1
        })+"\n")

        if topk["top1"]>best_full:
            best_full=topk["top1"]
            torch.save(net.state_dict(),os.path.join(tr["save_dir"],"best.pth"))
            print(f"🏅 new best model (top1={best_full:.2f}%)")

    csv_f.close()
    print(f"\n✅ done — best top1 = {best_full:.2f}%")

if __name__=="__main__":
    main()
