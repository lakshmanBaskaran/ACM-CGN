import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
os.environ.setdefault("TORCHINDUCTOR_CUDAGRAPHS", "0")
os.environ.setdefault("TORCHINDUCTOR_CUDAGRAPH_TREES", "0")

import yaml, time, multiprocessing, logging, json, csv, warnings
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# quiet, harmless warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io.image")
for msg in ("'has_cuda' is deprecated", "'has_cudnn' is deprecated",
            "'has_mps' is deprecated", "'has_mkldnn' is deprecated",
            "Detected call of `lr_scheduler.step"):
    warnings.filterwarnings("ignore", message=f".*{msg}.*")

import torch._dynamo
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors  = True
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)

# Also hard-disable Inductor cudagraphs via config
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

# Prefer Flash/Mem-efficient SDPA
try:
    from torch.backends.cuda import sdp_kernel
    sdp_kernel.enable_flash(True)
    sdp_kernel.enable_mem_efficient(True)
    sdp_kernel.enable_math(False)
except Exception:
    pass

from dataset import RadioMLDataset
from model import DenoiserAutoEnc, MultiBranchGNNClassifier as GNNModClassifier

# ─────────── speed flags (unchanged numerics) ────────────────────────────────
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ─────────── helpers ─────────────────────────────────────────────────────────
def _tqdm(iterable, total, desc, leave=False, position=0):
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=leave, position=position)

def _f(x, d=0.0): return d if x is None else float(x)

# Simple EMA for finetune
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        self.enabled = self.decay > 0
        self.shadow = {}
        if not self.enabled: return
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().clone()

    @torch.no_grad()
    def update(self, model):
        if not self.enabled: return
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def apply_shadow(self, model):
        if not self.enabled: return None
        backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n].data)
        return backup

    def restore(self, model, backup):
        if not self.enabled or backup is None: return
        for n, p in model.named_parameters():
            if n in backup:
                p.data.copy_(backup[n].data)

# CUDA prefetcher: overlaps H2D
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

# ───────── config loader ─────────────────────────────────────────────────────
def load_cfg():
    with open("configs/config.yaml","r") as f:
        cfg = yaml.safe_load(f)

    tr = cfg["training"]

    # ints/floats
    for k in ("batch_size","test_batch_size","epochs"):
        tr[k] = int(tr[k])
    for k in ("lr","weight_decay","focal_gamma","label_smoothing",
              "grad_clip_norm","snr_loss_weight_low","snr_loss_weight_high"):
        tr[k] = _f(tr.get(k),1.0)

    # toggles + new eval controls
    tr.setdefault("use_autoenc",  True)
    tr["use_compile"] = False
    tr["compile_pyg"] = False
    tr.setdefault("compile_mode", "reduce-overhead")
    tr.setdefault("channels_last", True)
    tr.setdefault("adamw_fused", False)
    tr.setdefault("shuffle_val", False)           # NEW: allow shuffled val sampling
    tr.setdefault("f1_skip_zero_support", True)   # NEW: macro-F1 ignores zero-support classes
    tr.setdefault("f1_report_weighted", True)     # NEW: also report weighted F1

    # capping + micro-batch
    tr["max_samples_per_epoch"] = int(tr.get("max_samples_per_epoch", 0))
    tr["val_max_batches"]       = int(tr.get("val_max_batches", 0))
    tr["micro_batch_size"]      = int(tr.get("micro_batch_size", 0))

    # workers
    max_w = min(12, max(1, multiprocessing.cpu_count()-1))
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
    ae["use_compile"]  = False
    ae["compile_mode"] = tr.get("compile_mode", "reduce-overhead")

    # ----- Optional finetune overrides -----
    ft = cfg.get("finetune", {}) or {}
    if bool(ft.get("enabled", False)):
        # Hyperparams
        tr["epochs"] = int(ft.get("epochs", 5))
        tr["lr"]     = _f(ft.get("lr", 1e-4), 1e-4)
        tr["focal_gamma"] = _f(ft.get("focal_gamma", 0.0), 0.0)
        tr["label_smoothing"] = _f(ft.get("label_smoothing", 0.01), 0.01)
        tr["snr_loss_weight_high"] = _f(ft.get("snr_loss_weight_high", 4.2), 4.2)
        # Val faster during finetune (you can still force full eval with val_max_batches: 0)
        tr["val_max_batches"] = max(10, int(tr.get("val_max_batches", 25)))
        # Scheduler to cosine short run
        tr["scheduler"] = {"type":"cosine","T_max":tr["epochs"],"eta_min":0.0}
        # Model graph override(s)
        mdl = cfg["model"]
        mdl["graph_connectivity"] = ft.get("graph_connectivity", mdl.get("graph_connectivity","chain"))
        if ft.get("override_gnn_heads", None) is not None:
            mdl["gnn_heads"] = int(ft["override_gnn_heads"])
        # Curriculum disabled for finetune unless explicitly asked
        cfg["curriculum"]["enabled"] = bool(ft.get("curriculum_enabled", False))

    return cfg

# ─── optimizer + scheduler ───────────────────────────────────────────────────
def build_optim_and_sched(net, tr, steps_per_epoch):
    fused_flag = bool(tr.get("adamw_fused", False))
    if fused_flag and tr.get("channels_last", True):
        print("⚠️  Disabling fused AdamW because channels_last is enabled (mixed layouts).")
        fused_flag = False

    opt_kwargs = dict(lr=tr["lr"], weight_decay=tr["weight_decay"])
    try:
        opt = optim.AdamW(net.parameters(), **opt_kwargs, fused=fused_flag)  # fused may not exist → fallback below
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
            anneal_strategy="cos" if sc.get("anneal_strategy","cosine").startswith("cos") else "linear"
        )
        per_batch=True
    else:
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(sc.get("T_max", tr["epochs"])), eta_min=_f(sc.get("eta_min",0.0),0.0))
        per_batch=False
    return opt, sched, per_batch

# ─── metrics helpers ─────────────────────────────────────────────────────────
def update_topk_counters(logits, labels, topk_counts, total):
    with torch.no_grad():
        float_logits = logits.float()
        B = labels.size(0)
        _, pred = float_logits.topk(5,dim=1)
        topk_counts["top1"] += (pred[:,:1]==labels.unsqueeze(1)).any(dim=1).sum().item()
        topk_counts["top3"] += (pred[:,:3]==labels.unsqueeze(1)).any(dim=1).sum().item()
        topk_counts["top5"] += (pred[:,:5]==labels.unsqueeze(1)).any(dim=1).sum().item()
        total[0] += B

def update_confusion(conf_all, conf_bins, bins, edges, preds, trues, snrs):
    for p,t in zip(preds, trues):
        conf_all[t,p] += 1
    for p,t,s in zip(preds, trues, snrs):
        idx = np.searchsorted(edges, s, side="right")-1
        idx = max(0,min(idx,len(bins)-1))
        conf_bins[bins[idx]][t,p] += 1

# (kept for backward compatibility if you need per-class F1s only)
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

# NEW: clean macro-F1 (skip zero-support) + weighted-F1
def compute_f1_per_class(conf):
    C = conf.shape[0]
    f1s, supports = [], []
    for c in range(C):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        support = conf[c, :].sum()
        supports.append(int(support))
        if tp + fp > 0 and tp + fn > 0:
            precision = tp / (tp + fp)
            recall    = tp / (tp + fn)
            den = (precision + recall)
            f1 = 0.0 if den == 0 else (2 * precision * recall / den)
        else:
            f1 = 0.0
        f1s.append(float(f1))
    return f1s, supports

def macro_and_weighted_f1(conf, skip_zero_support=True):
    f1s, supports = compute_f1_per_class(conf)
    supp = np.array(supports, dtype=np.float64)
    f1a  = np.array(f1s, dtype=np.float64)

    if skip_zero_support:
        mask = supp > 0
        macro = float(f1a[mask].mean()) if mask.any() else 0.0
    else:
        macro = float(f1a.mean())

    total = supp.sum()
    weighted = float((f1a * supp).sum() / total) if total > 0 else 0.0
    return f1s, supports, macro, weighted

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

def train_step_with_microbatch(net, batch, tr, device, opt, scaler, num_classes: int, ema=None, sched=None, per_batch: bool=False):
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
            if ema is not None: ema.update(getattr(net, "module", net))
            if per_batch and (sched is not None):
                sched.step()  # after opt.step()
            return total_loss/max(1,total_seen), total_correct, total_seen, micro_bs
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" not in msg:
                raise
            torch.cuda.empty_cache()
            micro_bs = max(1, (micro_bs or (B//2)) // 2)
            if micro_bs == 1: raise

def _roll_for_tta(batch, seg_cfg, seg_shift_units):
    """
    seg_shift_units: integer shift in SEGMENT units (…,-1,0,+1,…)
    Rolls:
      - tm by samples = step * seg_shift_units
      - spec by seg units along time axis
      - cwt by samples along last axis
    """
    if seg_shift_units == 0:
        return batch

    step = int(seg_cfg["segment_len"] * (1 - seg_cfg["overlap"]))
    samp_shift = int(seg_shift_units * step)

    tm  = batch["tm"].roll(shifts=samp_shift, dims=1)
    sp  = batch["spec"].roll(shifts=seg_shift_units, dims=2)  # (B,F,Tseg,2)
    cwt = batch["cwt"]
    if isinstance(cwt, torch.Tensor) and cwt.numel() > 0:
        cwt = cwt.roll(shifts=samp_shift, dims=2)              # (B,S,T)

    return {"tm": tm, "spec": sp, "cwt": cwt, "snr": batch["snr"], "y": batch["y"]}

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

def val_step_with_tta(net, batch, tr, num_classes: int, seg_cfg, tta_shifts: int):
    # Build symmetric small shift set (in segment units)
    if tta_shifts <= 1:
        return val_step_with_microbatch(net, batch, tr, num_classes)

    half = tta_shifts // 2
    if tta_shifts % 2 == 1:
        shifts = list(range(-half, half + 1))  # e.g., 3 -> [-1,0,1]
    else:
        shifts = [s for s in range(-half, 0)] + [s for s in range(1, half + 1)]  # e.g., 2 -> [-1,1]

    logits_accum = None
    with torch.inference_mode():
        for s in shifts:
            b = _roll_for_tta(batch, seg_cfg, s)
            out = val_step_with_microbatch(net, b, tr, num_classes)
            logits_accum = out if logits_accum is None else (logits_accum + out)
    return logits_accum / float(len(shifts))

# ─── tm-only AE dataset (fast pretrain) ──────────────────────────────────────
def _snr_batch_tm(x4: np.ndarray) -> np.ndarray:
    I, Q = x4[..., 0], x4[..., 1]
    power = (I*I + Q*Q).mean(axis=1)
    Imean = I.mean(axis=1, keepdims=True); Qmean = Q.mean(axis=1, keepdims=True)
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

# ─── main ────────────────────────────────────────────────────────────────────
def main():
    print("🚀  starting training …")
    C      = load_cfg()
    tr     = C["training"]
    ae_cfg = C["autoenc"]
    ft_cfg = C.get("finetune", {}) or {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(tr["save_dir"], exist_ok=True)

    # datasets (val is static; train uses curriculum or finetune filtering)
    # validation/test split is always the original "test"
    test_ds  = RadioMLDataset(C["data"]["processed_dir"],"test",
                              augment=False, seg_cfg=C["segmentation"], aug_cfg=C["augmentation"],
                              min_snr_db=None, max_snr_db=None)

    # ── optional fast AE pretrain ────────────────────────────────────────────
    autoenc=None
    if tr["use_autoenc"] and not bool(ft_cfg.get("enabled", False)):
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
        ae_opt=optim.AdamW(autoenc.parameters(),lr=1e-3,weight_decay=1e-5)
        scaler=GradScaler(enabled=bool(ae_cfg["use_amp"]))
        autoenc.train()
        start_wall = time.perf_counter()
        max_minutes = float(ae_cfg.get("max_minutes", 0) or 0)
        for ep in range(1, int(ae_cfg["epochs"]) + 1):
            epoch_loss = 0.0; seen = 0
            for step, batch in enumerate(_tqdm(ae_loader, total=len(ae_loader), desc=f"AE ep{ep:02d}", leave=False)):
                if max_minutes > 0 and (time.perf_counter() - start_wall) / 60.0 >= max_minutes:
                    print(f"⏱️  AE time cap reached. Stopping AE pretrain early.")
                    break
                tm = batch["tm"].to(device, non_blocking=True)
                noise   = torch.randn_like(tm) * float(ae_cfg["noise_std"])
                corrupt = tm + noise
                with autocast(enabled=bool(ae_cfg["use_amp"])): recon = autoenc(corrupt); loss = F.mse_loss(recon, tm)
                ae_opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(ae_opt); scaler.update()
                bs = tm.size(0); epoch_loss += loss.item() * bs; seen += bs
                if int(ae_cfg.get("max_steps", 0))>0 and (step+1) >= int(ae_cfg["max_steps"]):
                    break
            print(f" AE ep{ep:02d} ➜ mse {epoch_loss/max(1,seen):.8f}")
        for p in autoenc.parameters(): p.requires_grad=False
        autoenc.eval()

    # ── model ────────────────────────────────────────────────────────────────
    base = GNNModClassifier(C).to(device)
    if tr.get("channels_last", True):
        base = base.to(memory_format=torch.channels_last)

    # Load AE weights if we did pretrain
    if autoenc:
        base.autoenc.load_state_dict(autoenc.state_dict(), strict=False)
        for p in base.autoenc.parameters(): p.requires_grad = False
        base.autoenc.eval()

    # Finetune: load checkpoint (partial OK if graph/head changed)
    if bool(ft_cfg.get("enabled", False)):
        ckpt_path = ft_cfg.get("load_from", "")
        if ckpt_path and os.path.exists(ckpt_path):
            sd = torch.load(ckpt_path, map_location="cpu")
            missing, unexpected = base.load_state_dict(sd, strict=False)
            print(f"ℹ️ Loaded '{ckpt_path}' (strict=False). Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        else:
            print("⚠️  finetune.enabled is True but load_from not found. Proceeding without preload.")

    if torch.cuda.device_count()>1:
        print("ℹ️ Using DataParallel. (No change to training math.)")
        net = torch.nn.DataParallel(base)
        net_for_ema = net.module
    else:
        net = base
        net_for_ema = net

    # EMA for finetune (optional)
    ema = None
    if bool(ft_cfg.get("enabled", False)) and float(ft_cfg.get("ema_decay", 0)) > 0:
        ema = EMA(net_for_ema, decay=float(ft_cfg["ema_decay"]))

    # ── validation loader (shuffle optional to reduce partial-eval bias) ─────
    val_shuffle = bool(tr.get("shuffle_val", False))
    if val_shuffle:
        val_sampler = RandomSampler(test_ds)
        val_shuffle_flag = False
    else:
        val_sampler = SequentialSampler(test_ds)
        val_shuffle_flag = False

    dl_va = DataLoader(
        test_ds,
        batch_size=tr["test_batch_size"],
        shuffle=val_shuffle_flag,
        sampler=val_sampler,
        pin_memory=True,
        num_workers=max(0, tr["num_workers"] // 2),
        persistent_workers=tr["persistent_workers"],
        prefetch_factor=max(2, tr["prefetch_factor"] // 2),
    )

    # writers
    csv_name = "epoch_metrics_finetune.csv" if bool(ft_cfg.get("enabled", False)) else "epoch_metrics.csv"
    csv_path=Path(tr["save_dir"])/csv_name
    csv_f=open(csv_path,"w",newline="")
    csv_w=None
    jsonl_path=Path(tr["save_dir"])/csv_name.replace(".csv",".jsonl")

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

    # curriculum-aware / finetune-aware train loader builder
    cur_cfg = C.get("curriculum", {"enabled": False})
    seg_cfg = C["segmentation"]

    def _clamp01(x: float) -> float:
        return max(0.0, min(1.0, x))

    def build_train_loader_for_epoch(ep: int):
        # ── Finetune path: now supports min+max SNR and optional class balance ──
        if bool(ft_cfg.get("enabled", False)):
            min_snr = float(ft_cfg.get("min_snr_db", 0.0))
            max_snr = ft_cfg.get("max_snr_db", None)
            max_snr = float(max_snr) if max_snr is not None else None

            ds = RadioMLDataset(
                C["data"]["processed_dir"], "train",
                augment=True, seg_cfg=seg_cfg, aug_cfg=C["augmentation"],
                min_snr_db=min_snr, max_snr_db=max_snr
            )
            note = f"FT SNR≥{min_snr:.1f}dB" + (f" & ≤{max_snr:.1f}dB" if max_snr is not None else "")

            if len(ds) == 0:
                print("⚠️ Finetune filter produced empty dataset; falling back to all SNRs.")
                ds = RadioMLDataset(
                    C["data"]["processed_dir"], "train",
                    augment=True, seg_cfg=seg_cfg, aug_cfg=C["augmentation"],
                    min_snr_db=None, max_snr_db=None
                )
                note += " → fallback(all)"

            # Optional: class-balanced sampling within the filtered SNR slice
            if bool(ft_cfg.get("class_balance", False)) and len(ds) > 0:
                # label histogram
                tmp_loader = DataLoader(ds, batch_size=8192, shuffle=False, num_workers=0)
                counts = torch.zeros(num_classes, dtype=torch.long)
                for b in tmp_loader:
                    yb = b["y"]
                    counts.index_add_(0, yb, torch.ones_like(yb, dtype=torch.long))

                # inverse-frequency weights
                inv = 1.0 / counts.clamp_min(1).float()
                sample_weights_parts = []
                tmp_loader = DataLoader(ds, batch_size=8192, shuffle=False, num_workers=0)
                for b in tmp_loader:
                    sample_weights_parts.append(inv[b["y"]].clone())
                sample_weights = torch.cat(sample_weights_parts)

                # choose epoch sample count
                num_samples = int(tr["max_samples_per_epoch"]) if int(tr["max_samples_per_epoch"]) > 0 else int(sample_weights.numel())
                sampler = WeightedRandomSampler(sample_weights, num_samples=num_samples, replacement=True)

                dl = DataLoader(
                    ds, batch_size=tr["batch_size"], sampler=sampler, drop_last=True,
                    pin_memory=True, num_workers=tr["num_workers"],
                    persistent_workers=tr["persistent_workers"],
                    prefetch_factor=tr["prefetch_factor"],
                )
                note += " · class-balance"
                return ds, dl, note

            # default (no class-balance)
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

        # ── base training (unchanged) ────────────────────────────────────────
        if cur_cfg.get("enabled", False):
            start = float(cur_cfg.get("start_snr_db", -20.0))
            end   = float(cur_cfg.get("end_snr_db",   10.0))
            pace  = max(1, int(cur_cfg.get("pace_epochs", 5)))
            prog  = _clamp01((ep-1)/(pace-1)) if pace > 1 else 1.0
            curr_low = start + prog * (end - start)
            ds = RadioMLDataset(C["data"]["processed_dir"],"train",
                                augment=True, seg_cfg=seg_cfg, aug_cfg=C["augmentation"],
                                min_snr_db=curr_low, max_snr_db=None)
            note = f"SNR≥{curr_low:.1f}dB"
        else:
            ds = RadioMLDataset(C["data"]["processed_dir"],"train",
                                augment=True, seg_cfg=seg_cfg, aug_cfg=C["augmentation"],
                                min_snr_db=None, max_snr_db=None)
            note = "no-curriculum"

        if len(ds) == 0:
            print("⚠️  Train dataset is empty even after fallback. Skipping this epoch.")
            dl = DataLoader(ds, batch_size=tr["batch_size"], shuffle=False,
                            sampler=SequentialSampler(ds), drop_last=False,
                            pin_memory=True, num_workers=0)
            return ds, dl, note + " (empty)"

        sampler = None
        if tr["max_samples_per_epoch"] > 0:
            sampler = RandomSampler(ds, replacement=True, num_samples=tr["max_samples_per_epoch"])
        dl = DataLoader(
            ds, batch_size=tr["batch_size"],
            shuffle=(sampler is None), sampler=sampler, drop_last=True,
            pin_memory=True, num_workers=tr["num_workers"],
            persistent_workers=tr["persistent_workers"],
            prefetch_factor=tr["prefetch_factor"],
        )
        return ds, dl, note

    opt=sched=per_batch=None
    scaler=GradScaler()
    best_full = 0.0
    best_10_30 = 0.0  # track best high-SNR slice
    ckpt_name = "best_finetune.pth" if bool(ft_cfg.get("enabled", False)) else "best.pth"

    # F1 control toggles
    skip_zero = bool(tr.get("f1_skip_zero_support", True))
    report_w  = bool(tr.get("f1_report_weighted", True))

    for ep in range(1,tr["epochs"]+1):
        train_ds, dl_tr, cur_note = build_train_loader_for_epoch(ep)
        print(f"📚 epoch {ep:02d} curriculum: {cur_note} · train_samples={len(train_ds)}")

        if opt is None or sched is None:
            opt, sched, per_batch = build_optim_and_sched(net, tr, len(dl_tr))

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
            loss_b, corr_b, seen_b, used_micro = train_step_with_microbatch(
                net, batch, tr, device, opt, scaler, num_classes, ema=ema, sched=sched, per_batch=per_batch
            )
            tot_loss+=loss_b*seen_b; tot_samples+=seen_b; correct += corr_b
            if used_micro and used_micro > 0:
                pbar.set_postfix(micro_bs=used_micro)

        if not per_batch and hasattr(sched, "step"):
            sched.step()

        train_loss=tot_loss/max(1,tot_samples)
        train_acc =100*correct/max(1,tot_samples)
        print(f"epoch {ep:02d} ➜ loss {train_loss:.4f} · train acc {train_acc:.2f}%")

        # ── validate (supports TTA + EMA shadow) ─────────────────────────────
        net.eval()
        masked_corr=masked_tot=0
        hi_corr=hi_tot=0
        topk_counts={"top1":0,"top3":0,"top5":0}
        total_count=[0]
        Cn=num_classes
        conf_all=np.zeros((Cn,Cn),dtype=int)
        conf_bins={b:np.zeros((Cn,Cn),dtype=int) for b in bins}

        # swap in EMA weights for eval if present
        backup=None
        if ema is not None:
            backup = ema.apply_shadow(net_for_ema)

        prefetch_val = make_prefetch(dl_va)
        max_val_batches = tr["val_max_batches"] if tr["val_max_batches"]>0 else len(dl_va)
        bcount = 0
        tta_n = int(ft_cfg.get("tta_shifts", 1)) if bool(ft_cfg.get("enabled", False)) else 1

        for batch in _tqdm(prefetch_val, total=max_val_batches, desc=f"val   {ep:02d}", leave=False, position=1):
            if device.type != "cuda":
                for k in ("tm","spec","cwt","snr","y"):
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(device, non_blocking=True)

            logits = val_step_with_tta(net, batch, tr, num_classes, seg_cfg, tta_n)
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

        # restore non-EMA weights
        if ema is not None:
            ema.restore(net_for_ema, backup)

        masked_val_acc = 100.0 * masked_corr / max(1, masked_tot)
        val_acc_10_30  = 100.0 * hi_corr     / max(1, hi_tot)

        topk = {k: 100.*v/total_count[0] for k,v in topk_counts.items()}

        # NEW: clean macro/weighted F1 overall
        class_f1_all, supports_all, macro_f1_clean, weighted_f1_all = macro_and_weighted_f1(
            conf_all, skip_zero_support=skip_zero
        )
        macro_f1_strict = float(np.mean(class_f1_all))  # strict (zeros included)

        # per-bin F1 stats
        bin_macro_f1_clean = {}
        bin_macro_f1_strict = {}
        bin_weighted_f1 = {}
        bin_acc= {}
        for b in bins:
            f1_b, supp_b, macro_clean_b, weighted_b = macro_and_weighted_f1(conf_bins[b], skip_zero_support=skip_zero)
            bin_macro_f1_clean[b]  = macro_clean_b
            bin_macro_f1_strict[b] = float(np.mean(f1_b))
            bin_weighted_f1[b]     = weighted_b
            bin_acc[b] = 100.*conf_bins[b].trace()/conf_bins[b].sum() if conf_bins[b].sum()>0 else 0.0

        # CSV header on first epoch
        if csv_w is None:
            cols = [
                "epoch","train_loss","train_acc","masked_val_acc","val_acc_10_30",
                "top1","top3","top5",
                "macro_f1",            # clean (skip-zero if enabled)
                "macro_f1_strict",     # strict (zeros included)
            ]
            if report_w:
                cols += ["weighted_f1"]

            for i in range(Cn):
                cols.append(f"class_{i}_f1")

            for b in bins:
                cols.append(f"snr_{b}_acc")
                cols.append(f"snr_{b}_macro_f1")         # strict (back-compat)
                cols.append(f"snr_{b}_macro_f1_clean")   # clean (new)
                if report_w:
                    cols.append(f"snr_{b}_weighted_f1")

            csv_w=csv.DictWriter(csv_f,fieldnames=cols); csv_w.writeheader()

        print(f"val {ep:02d} ➜ acc[10–30 dB] {val_acc_10_30:.2f}%")

        # row
        flat={
            "epoch":ep,
            "train_loss":train_loss,
            "train_acc":train_acc,
            "masked_val_acc":masked_val_acc,
            "val_acc_10_30":val_acc_10_30,
            **topk,
            "macro_f1":macro_f1_clean,
            "macro_f1_strict":macro_f1_strict
        }
        if report_w:
            flat["weighted_f1"] = weighted_f1_all

        for i,f1 in enumerate(class_f1_all):
            flat[f"class_{i}_f1"]=f1
        for b in bins:
            flat[f"snr_{b}_acc"]=bin_acc[b]
            flat[f"snr_{b}_macro_f1"]=bin_macro_f1_strict[b]
            flat[f"snr_{b}_macro_f1_clean"]=bin_macro_f1_clean[b]
            if report_w:
                flat[f"snr_{b}_weighted_f1"]=bin_weighted_f1[b]

        csv_w.writerow(flat); csv_f.flush()
        with open(jsonl_path,"a") as jf: jf.write(json.dumps({
            **flat,
            "class_f1": class_f1_all,
            "class_supports": supports_all
        })+"\n")

        # ── checkpointing ────────────────────────────────────────────────────
        # save LAST every epoch (unwrapped module to avoid "module." keys)
        torch.save(net_for_ema.state_dict(), os.path.join(tr["save_dir"], "last.pth"))

        # best on overall top-1 (as before)
        if topk["top1"] > best_full:
            best_full = topk["top1"]
            torch.save(net_for_ema.state_dict(), os.path.join(tr["save_dir"], ckpt_name))
            print(f"🏅 new best model (top1={best_full:.2f}%) → saved to {ckpt_name}")

        # best on the high-SNR slice 10–30 dB
        if val_acc_10_30 > best_10_30:
            best_10_30 = val_acc_10_30
            torch.save(net_for_ema.state_dict(), os.path.join(tr["save_dir"], "best_10_30.pth"))
            print(f"🏆 new best high-SNR model (10–30 dB={best_10_30:.2f}%) → saved to best_10_30.pth")

    csv_f.close()
    print(f"\n✅ done — best top1 = {best_full:.2f}% · best 10–30 dB = {best_10_30:.2f}%")

if __name__=="__main__":
    main()
