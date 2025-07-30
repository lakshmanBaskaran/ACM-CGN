# SPDX-License-Identifier: MIT
import os
import yaml
import math
import multiprocessing
import logging
import json
from pathlib import Path

# ────────── suppress torch._dynamo spam ──────────
import torch._dynamo

torch._dynamo.config.cache_size_limit = 64
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score

from dataset import RadioMLDataset
from model import (
    DenoiserAutoEnc,
    MultiBranchGNNClassifier as GNNModClassifier,
)

# ───────────── speed flags ─────────────
torch.backends.cuda.matmul.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = True


# ───────────── helpers ─────────────
def _f(x, d=0.0):
    return d if x is None else float(x)


def load_cfg() -> dict:
    with open("configs/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tr = cfg["training"]
    # ints
    for k in ("batch_size", "test_batch_size", "epochs", "num_workers", "prefetch_factor"):
        tr[k] = int(tr.get(k, 0))
    # floats
    for k in ("lr", "weight_decay", "focal_gamma", "label_smoothing",
              "grad_clip_norm", "snr_loss_weight_low", "snr_loss_weight_high",
              "ema_decay", "unfreeze_lr", "recon_loss_weight"):
        if k in tr:
            tr[k] = _f(tr.get(k), 0.0)
    tr.setdefault("recon_loss_weight", 0.0)
    tr["freeze_after_epoch"] = int(tr.get("freeze_after_epoch", -1))
    tr["unfreeze_epoch"] = int(tr.get("unfreeze_epoch", tr.get("freeze_after_epoch", -1) + 2))

    # workers
    max_w = max(1, multiprocessing.cpu_count() - 1)
    tr["num_workers"] = min(tr.get("num_workers", 4), max_w)
    tr["persistent_workers"] = tr["num_workers"] > 0
    tr["prefetch_factor"] = max(2, tr.get("prefetch_factor", 2))

    # scheduler
    sc = tr.setdefault("scheduler", {"type": "cosine"})
    sc["type"] = sc.get("type", "cosine").lower()
    if sc["type"] == "onecycle":
        for k in ("max_lr", "pct_start", "div_factor", "final_div_factor"):
            sc[k] = _f(sc.get(k), 0.0)
        sc["anneal_strategy"] = sc.get("anneal_strategy", "cosine").lower()
    elif sc["type"] == "cawr":
        sc["T_0"] = int(sc.get("T_0", 5))
        sc["T_mult"] = int(sc.get("T_mult", 1))
        sc["eta_min"] = _f(sc.get("eta_min", 0.0))
    else:
        sc["T_max"] = int(sc.get("T_max", tr["epochs"]))
        sc["eta_min"] = _f(sc.get("eta_min", 0.0))

    # curriculum
    cur = cfg.setdefault("curriculum", {"enabled": False})
    cur["enabled"] = bool(cur.get("enabled", False))
    cur["start_snr_db"] = _f(cur.get("start_snr_db", 0.0))
    cur["end_snr_db"] = _f(cur.get("end_snr_db", 0.0))
    cur["pace_epochs"] = int(cur.get("pace_epochs", 1))
    cur["mask_val"] = bool(cur.get("mask_val", False))

    tr["use_torch_compile"] = bool(tr.get("use_torch_compile", False))
    return cfg


def build_optim_and_sched(net, tr, steps_per_epoch):
    opt = optim.AdamW(net.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    sc = tr["scheduler"]
    if sc["type"] == "onecycle":
        total = steps_per_epoch * tr["epochs"]
        sched = optim.lr_scheduler.OneCycleLR(
            opt, max_lr=sc["max_lr"], total_steps=total,
            pct_start=sc["pct_start"], div_factor=sc["div_factor"],
            final_div_factor=sc["final_div_factor"],
            anneal_strategy="cos" if sc["anneal_strategy"].startswith("cos") else "linear"
        )
        per_batch = True
    elif sc["type"] == "cawr":
        sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=sc["T_0"], T_mult=sc["T_mult"], eta_min=sc["eta_min"]
        )
        per_batch = False
    else:
        sched = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=sc["T_max"], eta_min=sc["eta_min"]
        )
        per_batch = False
    return opt, sched, per_batch


def get_state_dict_for_save(model):
    if hasattr(model, "_orig_mod"):
        return model._orig_mod.state_dict()
    return model.state_dict()


def ensure_metrics_file(path, n_classes, snr_vals):
    if not path.exists():
        header = ["epoch", "train_loss", "train_acc",
                  "masked_val_acc", "full_val_acc",
                  "macro_f1", "top1", "top3", "top5"]
        header += [f"f1_c{c}" for c in range(n_classes)]
        header += [f"acc_snr_{s}" for s in snr_vals]
        header += [f"f1_snr_{s}" for s in snr_vals]
        path.write_text(",".join(header) + "\n")


def append_metrics(path, row):
    with path.open("a") as f:
        f.write(",".join(map(str, row)) + "\n")


class EMA:
    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }
        self.backup = None

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow and v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)

    def apply_to(self, model):
        self.backup = {
            k: v.clone()
            for k, v in model.state_dict().items()
            if k in self.shadow
        }
        model.load_state_dict({**model.state_dict(), **self.shadow}, strict=False)

    def restore(self, model):
        model.load_state_dict({**model.state_dict(), **self.backup}, strict=False)
        self.backup = None


def tta_logits(net, tm, spec, cwt, scat, snr, seg_feats, eng_feats, n_rolls: int):
    outs = []
    for _ in range(n_rolls):
        shifts = torch.randint(0, tm.size(1), (tm.size(0),), device=tm.device)
        tm_s = torch.stack([torch.roll(tm[i], int(shifts[i]), dims=0)
                            for i in range(tm.size(0))], dim=0)
        logits, _ = net(tm_s, spec, cwt, scat, snr, seg_feats=seg_feats, eng_feats=eng_feats)
        outs.append(logits)
    return torch.stack(outs, dim=0).mean(dim=0)


def main() -> None:
    print("🚀 starting training…")
    C = load_cfg()
    tr = C["training"]
    cur = C["curriculum"]
    n_classes = C["model"]["num_classes"]
    recon_weight = tr["recon_loss_weight"]
    use_tta = bool(C.get("eval", {}).get("tta", {}).get("enabled", False))
    tta_n = int(C.get("eval", {}).get("tta", {}).get("n_rolls", 5))
    device = torch.device(tr["device"] if torch.cuda.is_available() else "cpu")
    Path(tr["save_dir"]).mkdir(parents=True, exist_ok=True)

    # ── datasets ────────────────────────────────────────────
    # The dataset now handles all filtering internally based on the config
    full_train = RadioMLDataset(C["data"]["processed_dir"], "train", augment=True)
    full_test = RadioMLDataset(C["data"]["processed_dir"], "test", augment=False)

    if not len(full_train) or not len(full_test):
        print("❌ Error: Datasets are empty after filtering. Check your config.")
        return

    # ── loaders ─────────────────────────────────────
    dl_tr = DataLoader(full_train, batch_size=tr["batch_size"],
                       shuffle=True,
                       drop_last=True, pin_memory=True,
                       num_workers=tr["num_workers"],
                       persistent_workers=tr["num_workers"] > 0,
                       prefetch_factor=tr["prefetch_factor"])
    dl_va = DataLoader(full_test, batch_size=tr["test_batch_size"],
                       shuffle=False, pin_memory=True,
                       num_workers=tr["num_workers"],
                       persistent_workers=tr["num_workers"] > 0,
                       prefetch_factor=tr["prefetch_factor"])

    # ── model ─────────────────────────────────────────────────
    net = GNNModClassifier(C).to(device)

    if tr["use_torch_compile"] and hasattr(torch, "compile"):
        print("✅ Compiling model with torch.compile()...")
        try:
            net = torch.compile(net, mode="reduce-overhead", fullgraph=False, dynamic=True)
        except Exception as e:
            print(f"⚠️ torch.compile disabled: {e}")

    opt, sched, per_batch = build_optim_and_sched(net, tr, len(dl_tr))
    scaler = GradScaler()
    ema = EMA(net, tr["ema_decay"]) if tr.get("ema_decay", 0.0) > 0 else None
    best_full, best_full_ema = 0.0, 0.0

    def _clean_feats(x, desired_dim):
        if x is None or x.numel() == 0: return None
        return x if x.ndim > 1 and x.size(-1) == desired_dim else None

    # ── evaluation function ─────────────────────────────────────────
    def run_eval(model_to_eval, dataloader, use_tta_flag=False):
        model_to_eval.eval()
        corr, tot = 0, 0
        all_preds, all_trues, all_snrs, all_logits = [], [], [], []
        with torch.no_grad(), autocast():
            for batch in tqdm(dataloader, desc="  [Validation]", leave=False):
                tm = batch["tm"].to(device, non_blocking=True)
                spec = batch["spec"].to(device, non_blocking=True)
                cwt = batch["cwt"].to(device, non_blocking=True)
                scat = batch["scat"].to(device, non_blocking=True)
                snr = batch["snr"].to(device, non_blocking=True)
                y = batch["y"].to(device, non_blocking=True)
                seg_feats_raw = batch.get("seg_feats")
                seg_feats = _clean_feats(seg_feats_raw, C["model"]["seg_feat_dim"])
                if seg_feats is not None: seg_feats = seg_feats.to(device)
                eng_feats = batch["eng_feats"].to(device, non_blocking=True)

                if use_tta_flag:
                    logits = tta_logits(model_to_eval, tm, spec, cwt, scat, snr, seg_feats, eng_feats, tta_n)
                else:
                    logits, _ = model_to_eval(tm, spec, cwt, scat, snr, seg_feats=seg_feats, eng_feats=eng_feats)

                pred = logits.argmax(1)
                corr += (pred == y).sum().item()
                tot += y.size(0)
                all_preds.append(pred.cpu().numpy())
                all_trues.append(y.cpu().numpy())
                all_snrs.append(snr.cpu().numpy())
                all_logits.append(logits.cpu().numpy())
        full_acc = 100 * corr / tot if tot > 0 else 0
        return (full_acc,
                np.concatenate(all_preds) if all_preds else np.array([]),
                np.concatenate(all_trues) if all_trues else np.array([]),
                np.concatenate(all_snrs) if all_snrs else np.array([]),
                np.concatenate(all_logits) if all_logits else np.array([]))

    # ── training loop ────────────────────────────────────────
    for ep in range(1, tr["epochs"] + 1):
        if ep == tr.get("freeze_after_epoch", -1):
            print("🔒 Freezing experts, bumping LR")
            # freeze_experts_and_ae(net) # This function is not defined in the provided code
            opt, sched, per_batch = build_optim_and_sched(net, tr, len(dl_tr))
            for g in opt.param_groups: g["lr"] = tr["lr"] * 2
        if ep == tr.get("unfreeze_epoch", -1):
            print("🔓 Unfreezing all modules, low‑LR polish")
            # unfreeze_all(net) # This function is not defined in the provided code
            opt, sched, per_batch = build_optim_and_sched(net, tr, len(dl_tr))
            for g in opt.param_groups: g["lr"] = tr["unfreeze_lr"]

        net.train()
        tot_loss, tot_samp, correct = 0, 0, 0

        for batch in tqdm(dl_tr, desc=f"train {ep:02d}", leave=False):
            tm = batch["tm"].to(device, non_blocking=True)
            spec = batch["spec"].to(device, non_blocking=True)
            cwt = batch["cwt"].to(device, non_blocking=True)
            scat = batch["scat"].to(device, non_blocking=True)
            snr = batch["snr"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            seg_feats_raw = batch.get("seg_feats")
            seg_feats = _clean_feats(seg_feats_raw, C["model"]["seg_feat_dim"])
            if seg_feats is not None: seg_feats = seg_feats.to(device)
            eng_feats = batch["eng_feats"].to(device, non_blocking=True)

            with autocast():
                logits, recon = net(tm, spec, cwt, scat, snr, seg_feats=seg_feats, eng_feats=eng_feats)
                weights = (snr < 0).float() * tr["snr_loss_weight_low"] + (snr >= 0).float() * tr[
                    "snr_loss_weight_high"]
                ce = F.cross_entropy(logits, y, reduction="none", label_smoothing=tr["label_smoothing"])
                if tr["focal_gamma"] > 0:
                    pt = torch.exp(-ce)
                    cls_loss = ((1 - pt) ** tr["focal_gamma"] * ce) * weights
                else:
                    cls_loss = ce * weights
                cls_loss = cls_loss.mean()
                loss = cls_loss + recon_weight * F.mse_loss(recon, tm) if recon_weight > 0 else cls_loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if tr["grad_clip_norm"] > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), tr["grad_clip_norm"])
            scaler.step(opt)
            scaler.update()
            if ema: ema.update(net)
            if per_batch: sched.step()
            tot_loss += loss.item() * y.size(0)
            tot_samp += y.size(0)
            correct += (logits.argmax(1) == y).sum().item()

        if not per_batch:
            sched.step(ep if sc["type"] == "cawr" else None)
        train_loss = tot_loss / tot_samp if tot_samp > 0 else 0
        train_acc = 100 * correct / tot_samp if tot_samp > 0 else 0
        print(f"epoch {ep:02d} → loss {train_loss:.4f} · train acc {train_acc:.2f}%")

        # ── CONSOLIDATED VALIDATION & METRICS ─────────────────────────────────
        full_acc, preds, trues, snrs, logits_arr = run_eval(net, dl_va, use_tta)
        print(f"           full val acc   {full_acc:.2f}%")

        if len(trues) == 0:
            print("           Skipping metrics calculation due to empty validation set.")
            continue

        top1 = full_acc
        top3 = (torch.from_numpy(logits_arr).topk(3, dim=1)[1].eq(torch.from_numpy(trues)[:, None]).any(
            1).float().mean().item() * 100)
        top5 = (torch.from_numpy(logits_arr).topk(5, dim=1)[1].eq(torch.from_numpy(trues)[:, None]).any(
            1).float().mean().item() * 100)
        class_f1 = f1_score(trues, preds, average=None, labels=np.arange(n_classes), zero_division=0)
        macro_f1 = class_f1.mean() * 100
        snr_round = np.rint(snrs).astype(int)
        if ep == 1:
            snr_vals = sorted(np.unique(snr_round).tolist())
            ensure_metrics_file(Path(tr["save_dir"]) / "epoch_metrics.csv", n_classes, snr_vals)
        snr_acc, snr_f1 = {}, {}
        for s in snr_vals:
            mask = snr_round == s
            if mask.any():
                snr_acc[s] = float((preds[mask] == trues[mask]).mean() * 100)
                snr_f1[s] = float(f1_score(trues[mask], preds[mask], average="macro", zero_division=0) * 100)
            else:
                snr_acc[s], snr_f1[s] = 0.0, 0.0
        row = [ep, f"{train_loss:.6f}", f"{train_acc:.4f}", full_acc, full_acc, f"{macro_f1:.4f}", f"{top1:.4f}",
               f"{top3:.4f}", f"{top5:.4f}"]
        row += [f"{x * 100:.4f}" for x in class_f1]
        row += [f"{snr_acc.get(s, 0):.4f}" for s in snr_vals]
        row += [f"{snr_f1.get(s, 0):.4f}" for s in snr_vals]
        append_metrics(Path(tr["save_dir"]) / "epoch_metrics.csv", row)
        with open(Path(tr["save_dir"]) / "epoch_metrics.jsonl", "a") as jf:
            jf.write(json.dumps({"epoch": ep, "train_loss": train_loss, "train_acc": train_acc,
                                 "masked_val_acc": full_acc, "full_val_acc": full_acc, "top1": top1,
                                 "top3": top3, "top5": top5, "macro_f1": macro_f1,
                                 "class_f1": (class_f1 * 100).tolist(),
                                 "snr_acc": {str(k): v for k, v in snr_acc.items()},
                                 "snr_macro_f1": {str(k): v for k, v in snr_f1.items()}}
                                ) + "\n")
        if full_acc > best_full:
            best_full = full_acc
            torch.save(get_state_dict_for_save(net), os.path.join(tr["save_dir"], "best.pth"))
            print(f"           🏅 new best full-range model ({best_full:.2f}%)")
        if ema:
            ema.apply_to(net)
            ema_acc, *_ = run_eval(net, dl_va, use_tta)
            ema.restore(net)
            if ema_acc > best_full_ema:
                best_full_ema = ema_acc
                torch.save(ema.shadow, os.path.join(tr["save_dir"], "best_ema.pth"))
                print(f"           ⭐ new best EMA model ({ema_acc:.2f}%)")
    print(f"\n✅ done — best full val acc = {best_full:.2f}% | best EMA = {best_full_ema:.2f}%")


if __name__ == "__main__":
    main()
