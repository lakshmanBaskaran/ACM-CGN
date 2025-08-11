# SPDX-License-Identifier: MIT
"""
Fuse NN logits with SVM probabilities using SNR-aware weights and (optionally) segment voting.

- Loads your trained GNN model and an SVM pipeline trained on deep embeddings.
- For each batch, computes:
    p_nn  = softmax(logits)
    p_svm = svm.predict_proba(embeddings)
    w_svm = sigmoid(k*(snr - snr0))  # per-sample weight (configurable)
    p_fused = (1 - w_svm) * p_nn + w_svm * p_svm

- Reports accuracy on:
    * all samples
    * SNR 0–30 dB
    * SNR 10–30 dB

Optional segment vote (if your dataset emits 'ex_id' or similar grouping): averages
probabilities across segments with the same id before argmax.
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast
from tqdm import tqdm
import joblib

# import project
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset import RadioMLDataset  # noqa: E402
from src.model import MultiBranchGNNClassifier as GNNModClassifier  # noqa: E402


def find_last_linear(module: nn.Module) -> nn.Linear | None:
    last = None
    for m in module.modules():
        if isinstance(m, nn.Linear):
            last = m
    return last


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@torch.no_grad()
def fuse_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # config
    import yaml
    with open(args.config, "r") as f:
        C = yaml.safe_load(f)

    # data
    ds = RadioMLDataset(
        C["data"]["processed_dir"],
        args.split,
        augment=False,
        seg_cfg=C.get("segmentation", {}),
        aug_cfg=C.get("augmentation", {}),
        min_snr_db=None,
        max_snr_db=None,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True, drop_last=False
    )

    # model
    net = GNNModClassifier(C)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    try:
        net.load_state_dict(ckpt)
    except Exception:
        new_state = {}
        for k, v in ckpt.items():
            if k.startswith("module."):
                new_state[k[len("module."):]] = v
            else:
                new_state[k] = v
        net.load_state_dict(new_state, strict=False)
    net.to(device).eval()

    # embedding capture (return_embed preferred; fallback hook)
    supports_return = False
    if hasattr(net, "forward"):
        try:
            net.forward.__code__.co_varnames.index("return_embed")
            supports_return = True
        except Exception:
            supports_return = False

    captured = {"feat": []}

    def hook_fn(_module, inputs, _output):
        x = inputs[0]
        captured["feat"].append(x.detach())

    hook = None
    if not supports_return:
        last_fc = find_last_linear(net)
        if last_fc is None:
            raise RuntimeError("Could not find final nn.Linear to hook for embeddings.")
        hook = last_fc.register_forward_hook(hook_fn)

    # SVM
    svm = joblib.load(args.svm)
    print(f"Loaded SVM from {args.svm}")

    all_probs = []
    all_labels = []
    all_snrs = []
    all_ids = []

    autocast_enabled = device.type == "cuda"

    for batch in tqdm(loader, desc="fuse-eval"):
        tm = batch.get("tm");  spec = batch.get("spec");  cwt = batch.get("cwt")
        snr = batch.get("snr"); y   = batch.get("y")
        ex_id = batch.get("id", None) or batch.get("ex_id", None)  # optional grouping id

        if isinstance(tm, torch.Tensor):  tm = tm.to(device, non_blocking=True)
        if isinstance(spec, torch.Tensor): spec = spec.to(device, non_blocking=True)
        if isinstance(cwt, torch.Tensor):  cwt = cwt.to(device, non_blocking=True)
        if isinstance(snr, torch.Tensor):  snr = snr.to(device, non_blocking=True)
        if isinstance(y, torch.Tensor):    y   = y.to(device, non_blocking=True)

        with autocast(enabled=autocast_enabled):
            if supports_return:
                logits, embed = net(tm, spec, cwt, snr, return_embed=True)
            else:
                logits = net(tm, spec, cwt, snr)

        if supports_return:
            feats = embed.detach().cpu().numpy()
        else:
            feats = torch.cat(captured["feat"], dim=0).cpu().numpy()
            captured["feat"].clear()

        p_nn = F.softmax(logits.float(), dim=1).cpu().numpy()
        p_svm = svm.predict_proba(feats)

        snr_cpu = snr.detach().cpu().numpy().astype(np.float32)
        # SNR-weighted fusion:
        # weight SVM more as SNR rises: w = sigmoid(k*(snr - snr0))
        k = float(args.snr_k)
        snr0 = float(args.snr_0)
        w_svm = sigmoid(k * (snr_cpu - snr0))  # shape (B,)
        w_svm = w_svm[:, None]  # (B, 1)

        p_fused = (1.0 - w_svm) * p_nn + w_svm * p_svm

        all_probs.append(p_fused)
        all_labels.append(y.detach().cpu().numpy())
        all_snrs.append(snr_cpu)
        if ex_id is None:
            all_ids.append(np.arange(p_fused.shape[0]))
        else:
            all_ids.append(np.asarray(ex_id))

    if hook is not None:
        hook.remove()

    P = np.concatenate(all_probs, axis=0)
    Y = np.concatenate(all_labels, axis=0)
    S = np.concatenate(all_snrs, axis=0)
    G = np.concatenate(all_ids, axis=0)

    # optional segment vote: average prob per group id
    if args.segment_vote:
        uniq = np.unique(G)
        P_agg = []
        Y_agg = []
        S_agg = []
        for gid in uniq:
            m = (G == gid)
            P_agg.append(P[m].mean(axis=0))
            # for label/SNR, take majority label (or first) and avg SNR
            Y_agg.append(np.bincount(Y[m]).argmax())
            S_agg.append(S[m].mean())
        P = np.stack(P_agg, axis=0)
        Y = np.asarray(Y_agg, dtype=np.int64)
        S = np.asarray(S_agg, dtype=np.float32)

    pred = P.argmax(axis=1)
    acc_all = (pred == Y).mean() * 100.0

    def range_acc(lo: float, hi: float):
        m = (S >= lo) & (S <= hi)
        if m.sum() == 0:
            return float("nan")
        return (pred[m] == Y[m]).mean() * 100.0

    acc_0_30 = range_acc(0.0, 30.0)
    acc_10_30 = range_acc(10.0, 30.0)

    print(f"\nFUSED accuracy: all={acc_all:.2f}% · 0–30 dB={acc_0_30:.2f}% · 10–30 dB={acc_10_30:.2f}%")
    print(f"(snr_k={args.snr_k}, snr_0={args.snr_0}, segment_vote={args.segment_vote})")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(REPO_ROOT / "configs/config.yaml"))
    p.add_argument("--checkpoint", required=True, help="model weights (best.pth)")
    p.add_argument("--svm", required=True, help="joblib from tools/train_svm.py")
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--segment-vote", action="store_true", help="average probs per example id if available")
    # SNR fusion parameters
    p.add_argument("--snr-k", type=float, default=0.35, help="slope in sigmoid weight")
    p.add_argument("--snr-0", type=float, default=8.0, help="snr midpoint (dB) where w_svm=0.5")
    return p.parse_args()


if __name__ == "__main__":
    fuse_eval(parse_args())
