# eval.py
# Standalone evaluation for ACM-CGN style project
# SPDX-License-Identifier: MIT

import os
import argparse
import json
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from dataset import RadioMLDataset
from model import MultiBranchGNNClassifier as GNNModClassifier


# ─────────────────────────── helpers ───────────────────────────

def _tqdm(it, total=None, desc="eval", leave=False, position=0):
    return tqdm(it, total=total, desc=desc, dynamic_ncols=True, leave=leave, position=position)

def load_cfg(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # Minimal defaults used here
    tr = cfg.setdefault("training", {})
    tr.setdefault("test_batch_size", 4096)
    tr.setdefault("channels_last", True)
    mdl = cfg.setdefault("model", {})
    assert "num_classes" in mdl, "configs/config.yaml: model.num_classes is required"
    return cfg

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_csv(path: Path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)

def compute_prf_from_conf(conf: np.ndarray):
    """Return per-class precision, recall, f1, support from confusion matrix (C x C)."""
    C = conf.shape[0]
    tp = np.diag(conf).astype(np.float64)
    support = conf.sum(axis=1).astype(np.float64)          # true counts per class
    pred_sum = conf.sum(axis=0).astype(np.float64)         # predicted counts per class

    precision = np.divide(tp, np.maximum(1.0, pred_sum), where=(pred_sum > 0))
    recall    = np.divide(tp, np.maximum(1.0, support), where=(support > 0))
    f1 = np.zeros(C, dtype=np.float64)
    denom = precision + recall
    mask = denom > 0
    f1[mask] = 2.0 * precision[mask] * recall[mask] / denom[mask]
    return precision, recall, f1, support

def metrics_from_conf(conf: np.ndarray):
    """Overall metrics from a confusion matrix."""
    C = conf.shape[0]
    total = conf.sum()
    correct = np.trace(conf)
    acc = float(correct) / float(total) if total > 0 else 0.0

    precision, recall, f1, support = compute_prf_from_conf(conf)
    macro_f1_strict = float(f1.mean())
    valid = support > 0
    macro_f1_clean = float(f1[valid].mean()) if valid.any() else 0.0
    weighted_f1 = float((f1 * support).sum() / np.maximum(1.0, support.sum()))
    return {
        "acc": acc,
        "macro_f1_strict": macro_f1_strict,
        "macro_f1_clean": macro_f1_clean,
        "weighted_f1": weighted_f1,
        "per_class": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
            "support": support.tolist(),
        },
    }

def topk_counts_from_logits(logits: torch.Tensor, labels: torch.Tensor, kset=(1,3,5)):
    with torch.no_grad():
        fl = logits.float()
        maxk = max(kset)
        _, pred = fl.topk(maxk, dim=1)
        eq = pred.eq(labels.view(-1, 1).expand_as(pred))
        ret = {}
        B = labels.size(0)
        for k in kset:
            ret[f"top{k}"] = float(eq[:, :k].any(dim=1).sum().item()) / float(B)
        return ret

def _roll_for_tta(batch, seg_cfg, seg_shift_units):
    if seg_shift_units == 0:
        return batch
    step = int(seg_cfg["segment_len"] * (1 - seg_cfg["overlap"]))
    samp_shift = int(seg_shift_units * step)
    tm  = batch["tm"].roll(shifts=samp_shift, dims=1)
    sp  = batch["spec"].roll(shifts=seg_shift_units, dims=2)  # (B,F,Tseg,2)
    cwt = batch["cwt"]
    if isinstance(cwt, torch.Tensor) and cwt.numel() > 0:
        cwt = cwt.roll(shifts=samp_shift, dims=2)
    return {"tm": tm, "spec": sp, "cwt": cwt, "snr": batch["snr"], "y": batch["y"]}

def logits_tta(model, batch, tta_shifts, seg_cfg, num_classes, micro_bs: int = 0):
    """Return logits for batch, averaged over symmetric segment shifts."""
    if tta_shifts <= 1:
        return logits_mb(model, batch, num_classes, micro_bs)
    half = tta_shifts // 2
    if tta_shifts % 2 == 1:
        shifts = list(range(-half, half + 1))  # e.g., 3 → [-1,0,1]
    else:
        shifts = [s for s in range(-half, 0)] + [s for s in range(1, half + 1)]
    acc = None
    for s in shifts:
        b = _roll_for_tta(batch, seg_cfg, s)
        out = logits_mb(model, b, num_classes, micro_bs)
        acc = out if acc is None else (acc + out)
    return acc / float(len(shifts))

def logits_mb(model, batch, num_classes, micro_bs: int = 0):
    tm, sp, cwt, snr = batch["tm"], batch["spec"], batch["cwt"], batch["snr"]
    y = batch["y"]
    B = y.size(0)
    outs = []
    with torch.inference_mode(), autocast():
        if micro_bs <= 0 or micro_bs >= B:
            out = model(tm, sp, cwt, snr).detach()
            assert out.size(1) == num_classes
            return out
        for st in range(0, B, micro_bs):
            ed = min(B, st + micro_bs)
            out = model(tm[st:ed], sp[st:ed], cwt[st:ed], snr[st:ed]).detach()
            outs.append(out)
    return torch.cat(outs, dim=0)


# ─────────────────────────── evaluation ───────────────────────────

def evaluate(cfg_path, ckpt_path, out_dir,
             batch_size=None, device="cuda", tta_shifts=1,
             micro_batch_size=0, shuffle=False, val_max_batches=0):
    cfg = load_cfg(cfg_path)
    seg_cfg = cfg["segmentation"]
    mdl_cfg = cfg["model"]
    tr_cfg  = cfg["training"]

    num_classes = int(mdl_cfg["num_classes"])
    device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")

    # Data
    test_ds = RadioMLDataset(
        cfg["data"]["processed_dir"],
        split="test",
        augment=False,
        seg_cfg=seg_cfg,
        aug_cfg=cfg.get("augmentation", {}),
        min_snr_db=None, max_snr_db=None
    )
    bs = int(batch_size or tr_cfg.get("test_batch_size", 4096))
    dl = DataLoader(
        test_ds,
        batch_size=bs,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=max(0, int(tr_cfg.get("num_workers", 8)) // 2),
        persistent_workers=bool(tr_cfg.get("persistent_workers", True)),
        prefetch_factor=max(2, int(tr_cfg.get("prefetch_factor", 6)) // 2),
        drop_last=False,
    )

    # Model
    net = GNNModClassifier(cfg).to(device)
    if bool(tr_cfg.get("channels_last", True)):
        net = net.to(memory_format=torch.channels_last)
    sd = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if len(missing) or len(unexpected):
        print(f"⚠️  Loaded with strict=False. Missing={len(missing)}, Unexpected={len(unexpected)}")

    net.eval()

    # Accumulators
    conf_all = np.zeros((num_classes, num_classes), dtype=np.int64)

    # SNR binning
    edges = np.array([-20, -10, 0, 10, 20, 30], dtype=np.float64)
    bins  = ["[-20,-10)", "[-10,0)", "[0,10)", "[10,20)", "[20,30]"]
    conf_bins = {b: np.zeros((num_classes, num_classes), dtype=np.int64) for b in bins}

    # Per-SNR integer accuracy
    per_snr = {}  # {snr_int: {"correct": x, "total": y}}

    # Top-k counts
    total_seen = 0
    topk_hits = {"top1": 0, "top3": 0, "top5": 0}

    # Optional cap on batches
    max_batches = int(val_max_batches or 0)
    processed_batches = 0

    # Eval loop
    for batch in _tqdm(dl, total=(len(dl) if max_batches == 0 else max_batches), desc="eval"):
        # to device
        for k in ("tm", "spec", "cwt", "snr", "y"):
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device, non_blocking=True)

        logits = logits_tta(net, batch, tta_shifts=tta_shifts, seg_cfg=seg_cfg,
                            num_classes=num_classes, micro_bs=int(micro_batch_size or 0))
        preds = logits.argmax(dim=1)
        y     = batch["y"]
        snr   = batch["snr"]

        # top-k
        tk = topk_counts_from_logits(logits, y, kset=(1,3,5))
        B = y.size(0)
        total_seen += B
        topk_hits["top1"] += int(tk["top1"] * B)
        topk_hits["top3"] += int(tk["top3"] * B)
        topk_hits["top5"] += int(tk["top5"] * B)

        # confusion (overall)
        p_np = preds.cpu().numpy().astype(np.int64)
        t_np = y.cpu().numpy().astype(np.int64)
        for t, p in zip(t_np, p_np):
            conf_all[t, p] += 1

        # bins + per-integer-SNR
        s_np = snr.detach().cpu().numpy()
        for t, p, s in zip(t_np, p_np, s_np):
            # bin
            idx = np.searchsorted(edges, s, side="right") - 1
            idx = max(0, min(idx, len(bins) - 1))
            conf_bins[bins[idx]][t, p] += 1
            # per integer SNR (rounded)
            s_int = int(np.round(s))
            d = per_snr.setdefault(s_int, {"correct": 0, "total": 0})
            d["total"] += 1
            d["correct"] += int(t == p)

        processed_batches += 1
        if max_batches > 0 and processed_batches >= max_batches:
            break

    # ── Metrics
    overall = metrics_from_conf(conf_all)
    topk = {
        "top1": float(topk_hits["top1"]) / max(1, total_seen),
        "top3": float(topk_hits["top3"]) / max(1, total_seen),
        "top5": float(topk_hits["top5"]) / max(1, total_seen),
    }

    # Per-bin metrics
    per_bin_rows = []
    per_bin_summary = {}
    for b in bins:
        res = metrics_from_conf(conf_bins[b])
        per_bin_summary[b] = res
        per_bin_rows.append([
            b,
            res["acc"],
            res["macro_f1_strict"],
            res["macro_f1_clean"],
            res["weighted_f1"],
            int(conf_bins[b].sum())
        ])

    # Per-class overall table
    pc = overall["per_class"]
    per_class_rows = [["class", "precision", "recall", "f1", "support"]]
    for i in range(num_classes):
        per_class_rows.append([i, pc["precision"][i], pc["recall"][i], pc["f1"][i], int(pc["support"][i])])

    # Per-class per-bin table
    per_class_per_bin_rows = [["bin", "class", "precision", "recall", "f1", "support"]]
    for b in bins:
        p_b = metrics_from_conf(conf_bins[b])["per_class"]
        for i in range(num_classes):
            per_class_per_bin_rows.append([
                b, i, p_b["precision"][i], p_b["recall"][i], p_b["f1"][i], int(p_b["support"][i])
            ])

    # Per-SNR integer accuracy rows
    per_snr_rows = [["snr_db", "accuracy", "correct", "total"]]
    for s in sorted(per_snr.keys()):
        d = per_snr[s]
        acc = float(d["correct"]) / max(1, d["total"])
        per_snr_rows.append([s, acc, d["correct"], d["total"]])

    # ── Save
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    # Summary JSON
    summary = {
        "config": cfg_path,
        "checkpoint": ckpt_path,
        "samples_evaluated": int(conf_all.sum()),
        "overall": {
            "accuracy": overall["acc"],
            "macro_f1_strict": overall["macro_f1_strict"],
            "macro_f1_clean": overall["macro_f1_clean"],
            "weighted_f1": overall["weighted_f1"],
            **topk
        },
        "per_bin": {
            b: {
                "acc": per_bin_summary[b]["acc"],
                "macro_f1_strict": per_bin_summary[b]["macro_f1_strict"],
                "macro_f1_clean": per_bin_summary[b]["macro_f1_clean"],
                "weighted_f1": per_bin_summary[b]["weighted_f1"],
                "support": int(conf_bins[b].sum()),
            } for b in bins
        }
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # CSVs
    save_csv(out_dir / "per_class_metrics.csv",
             ["class", "precision", "recall", "f1", "support"],
             per_class_rows[1:])

    save_csv(out_dir / "per_bin_metrics.csv",
             ["bin", "acc", "macro_f1_strict", "macro_f1_clean", "weighted_f1", "support"],
             per_bin_rows)

    save_csv(out_dir / "per_class_per_bin_metrics.csv",
             ["bin", "class", "precision", "recall", "f1", "support"],
             per_class_per_bin_rows[1:])

    save_csv(out_dir / "per_snr_integer_accuracy.csv",
             ["snr_db", "accuracy", "correct", "total"],
             per_snr_rows[1:])

    # Confusion matrices
    np.save(out_dir / "confusion_overall.npy", conf_all)
    np.savetxt(out_dir / "confusion_overall.csv", conf_all, fmt="%d", delimiter=",")
    for b in bins:
        np.save(out_dir / f"confusion_{b}.npy", conf_bins[b])
        np.savetxt(out_dir / f"confusion_{b}.csv", conf_bins[b], fmt="%d", delimiter=",")

    # Pretty print summary
    print("\n==== Evaluation Summary ====")
    print(f"Samples evaluated: {int(conf_all.sum())}")
    print(f"Top-1: {topk['top1']*100:.2f}% | Top-3: {topk['top3']*100:.2f}% | Top-5: {topk['top5']*100:.2f}%")
    print(f"Overall Acc: {overall['acc']*100:.2f}%")
    print(f"Macro-F1 (strict): {overall['macro_f1_strict']:.4f}")
    print(f"Macro-F1 (clean):  {overall['macro_f1_clean']:.4f}")
    print(f"Weighted-F1:       {overall['weighted_f1']:.4f}")
    print("\nPer-bin (acc | macroF1_clean | weightedF1 | support):")
    for b in bins:
        res = per_bin_summary[b]
        sup = int(conf_bins[b].sum())
        print(f"  {b:>10} → {res['acc']*100:6.2f}% | {res['macro_f1_clean']:.4f} | {res['weighted_f1']:.4f} | {sup}")

    print(f"\nArtifacts saved to: {out_dir.resolve()}")


# ─────────────────────────── CLI ───────────────────────────

def main():
    p = argparse.ArgumentParser("Standalone evaluator")
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--ckpt", type=str, required=True, help="Path to model .pth")
    p.add_argument("--out-dir", type=str, default="eval_out")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--micro-batch-size", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")  # "cuda" or "cpu"
    p.add_argument("--tta-shifts", type=int, default=1, help="Use symmetric segment shifts (e.g., 3 = {-1,0,+1})")
    p.add_argument("--shuffle", action="store_true", help="Shuffle eval loader (usually false)")
    p.add_argument("--val-max-batches", type=int, default=0, help="0 = full sweep; otherwise cap")
    args = p.parse_args()

    evaluate(
        cfg_path=args.config,
        ckpt_path=args.ckpt,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        device=args.device,
        tta_shifts=args.tta_shifts,
        micro_batch_size=args.micro_batch_size,
        shuffle=args.shuffle,
        val_max_batches=args.val_max_batches
    )

if __name__ == "__main__":
    main()
