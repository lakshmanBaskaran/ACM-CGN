# SPDX-License-Identifier: MIT
"""
Extract penultimate embeddings, labels, and SNR from a trained model.

This tries `model(..., return_embed=True)` first. If the model doesn't support it,
it registers a forward hook on the LAST nn.Linear module and captures its input
as the embedding vector.

Saves: an .npz with arrays: embeddings (N,D), labels (N,), snr (N,), idx (N,)
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm

# Make "src" importable when running from repo root
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


@torch.no_grad()
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Load config
    import yaml
    with open(args.config, "r") as f:
        C = yaml.safe_load(f)

    # Dataset
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

    # Model
    net = GNNModClassifier(C)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    # handle DP/compile wrappers:
    try:
        net.load_state_dict(ckpt)
    except Exception:
        # Try to strip "module." prefixes if needed
        new_state = {}
        for k, v in ckpt.items():
            if k.startswith("module."):
                new_state[k[len("module."):]] = v
            else:
                new_state[k] = v
        net.load_state_dict(new_state, strict=False)

    net.to(device).eval()

    # Prepare embedding capture
    captured = {"feat": []}

    def hook_fn(_module, inputs, _output):
        # capture the input to classifier as embedding
        # inputs is a tuple; we take the first
        x = inputs[0]
        captured["feat"].append(x.detach())

    hook = None
    # prefer built-in path return_embed
    supports_return = False
    if hasattr(net, "forward"):
        try:
            # dry-run with dummy to see if signature supports return_embed (we won't actually run)
            net.forward.__code__.co_varnames.index("return_embed")
            supports_return = True
        except Exception:
            supports_return = False

    # fallback: hook last linear
    if not supports_return:
        last_fc = find_last_linear(net)
        if last_fc is None:
            raise RuntimeError("Could not find a final nn.Linear to hook. "
                               "Consider adding `return_embed` to your model forward.")
        hook = last_fc.register_forward_hook(hook_fn)

    all_embeds = []
    all_labels = []
    all_snr = []
    all_idx = []

    idx_counter = 0
    autocast_enabled = device.type == "cuda"

    for batch in tqdm(loader, desc="extract"):
        # move to device
        tm = batch.get("tm");  spec = batch.get("spec");  cwt = batch.get("cwt")
        snr = batch.get("snr"); y   = batch.get("y")
        if isinstance(tm, torch.Tensor):  tm = tm.to(device, non_blocking=True)
        if isinstance(spec, torch.Tensor): spec = spec.to(device, non_blocking=True)
        if isinstance(cwt, torch.Tensor):  cwt = cwt.to(device, non_blocking=True)
        if isinstance(snr, torch.Tensor):  snr = snr.to(device, non_blocking=True)
        if isinstance(y, torch.Tensor):    y   = y.to(device, non_blocking=True)

        with autocast(enabled=autocast_enabled):
            if supports_return:
                logits, embed = net(tm, spec, cwt, snr, return_embed=True)
            else:
                _ = net(tm, spec, cwt, snr)  # hook will fill captured["feat"]

        if supports_return:
            feats = embed.detach()
        else:
            feats = torch.cat(captured["feat"], dim=0)
            captured["feat"].clear()

        all_embeds.append(feats.cpu())
        all_labels.append(y.cpu())
        all_snr.append(snr.cpu())
        # synthetic incremental indices (or use actual if dataset provides)
        all_idx.append(torch.arange(idx_counter, idx_counter + feats.size(0)))
        idx_counter += feats.size(0)

    if hook is not None:
        hook.remove()

    E = torch.cat(all_embeds, dim=0).numpy()
    L = torch.cat(all_labels, dim=0).numpy()
    S = torch.cat(all_snr, dim=0).numpy()
    I = torch.cat(all_idx, dim=0).numpy()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, embeddings=E, labels=L, snr=S, idx=I)
    print(f"saved: {out}  -> embeddings {E.shape}, labels {L.shape}, snr {S.shape}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(REPO_ROOT / "configs/config.yaml"))
    p.add_argument("--checkpoint", required=True, help="Path to model weights (e.g., save_dir/best.pth)")
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out", default=str(REPO_ROOT / "artifacts/embeddings_test.npz"))
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
