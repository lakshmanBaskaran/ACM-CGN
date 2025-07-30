#!/usr/bin/env python3
# scripts/preprocess.py
# SPDX-License-Identifier: MIT

import sys, os, math, yaml, h5py, numpy as np
import torch, torchaudio
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_ROOT)

from dsp_utils import (
    bandpass_filter, wiener_filter, matched_filter, cwt_spec, wavelet_denoise, dwt_spec,
    estimate_snr, hoc_features, scd_matrix, wpt_energy, bispectrum, memmap
)
from kymatio.torch import Scattering1D as TorchScattering1D

with open("configs/config.yaml", "r") as f: C = yaml.safe_load(f)
P, OUT_DIR, SEG = C["preprocess"], C["data"]["processed_dir"], C["segmentation"]
S, L = SEG["num_segments"], SEG["segment_len"]
step = int(L * (1 - SEG.get("overlap", 0.5)))
ENG_DIM = C["model"]["eng_feat_dim"]
cpu_workers = P.get("cpu_workers", max(1, multiprocessing.cpu_count() - 1))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
spec_cfg = P["spectrogram"]
stft_transform = torchaudio.transforms.Spectrogram(
    n_fft=spec_cfg["n_fft"], hop_length=spec_cfg["hop_length"],
    window_fn=torch.hann_window, power=None, normalized=False
).to(device)
F_spec = stft_transform(torch.zeros(1, 1024, device=device)).shape[-2]

# STABILITY FIX: Changed J=4 to J=3 to resolve Kymatio warning
scat_torch = TorchScattering1D(J=3, shape=L).to(device)
C_s = scat_torch(torch.zeros(1, L, device=device)).shape[1]


def extract_cpu_features(iq_pair, label):
    iq = iq_pair.copy()
    iq /= np.sqrt((iq ** 2).mean()) + 1e-9
    dsp = P.get("dsp", {})
    if dsp.get("bandpass", {}).get("enabled"):
        bp = dsp["bandpass"]
        iq[:, :2] = bandpass_filter(iq[:, :2], bp["low"], bp["high"], bp["fs"], bp.get("order", 4))
    if dsp.get("wiener", {}).get("enabled"):
        iq[:, :2] = wiener_filter(iq[:, :2], dsp["wiener"].get("mysize", 5))
    snr = estimate_snr(iq)
    if snr < P["denoise"]["threshold_db"]:
        iq = wavelet_denoise(iq, P["denoise"]["wavelet"], P["denoise"]["level"])
    I, Q = iq[:, 0], iq[:, 1];
    z = I + 1j * Q;
    amp = np.abs(z);
    ph = np.angle(z)
    tm_feat = np.stack([I, Q, amp, ph], axis=1).astype("float32")
    m2 = (amp ** 2).mean();
    m4 = (amp ** 4).mean();
    C40 = np.abs(np.mean(z ** 4)) - 3 * (np.abs(np.mean(z ** 2)) ** 2)
    C42 = m4 - np.abs(np.mean(z ** 2)) ** 2 - 2 * m2 ** 2;
    a_m, a_v = amp.mean(), amp.var()
    a_k = np.mean((amp - a_m) ** 4) / (a_v ** 2 + 1e-12) if a_v > 1e-12 else 0;
    dphi = np.diff(ph)
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi;
    p_v = dphi.var()
    p_k = np.mean((dphi - dphi.mean()) ** 4) / (p_v ** 2 + 1e-12) if p_v > 1e-12 else 0;
    T = len(z)
    Zf = np.fft.fft(z);
    psd = (np.abs(Zf) ** 2) / T;
    topk_indices = np.argsort(psd)[-5:][::-1]
    vals, idxs_ = psd[topk_indices], topk_indices / T
    eng_feat = np.concatenate([[C40], [C42], [a_m], [a_v], [a_k], [p_v], [p_k], vals, idxs_]).astype("float32")
    other_feats = {}
    if P.get("save_dtype", {}).get("cwt"): other_feats['cw'] = cwt_spec(z, P["cwt"]["scales"])
    if P.get("save_dtype", {}).get("dwt"): other_feats['dw'] = dwt_spec(z, P["dwt"]["wavelet"], P["dwt"]["level"])
    if P.get("hoc", {}).get("enabled"): other_feats['hoc'] = hoc_features(z, P["hoc"])
    if P.get("scd", {}).get("enabled"): other_feats['scd'] = scd_matrix(z, P["scd"])
    if P.get("wpt", {}).get("enabled"): other_feats['wpt'] = wpt_energy(z, P["wpt"])
    if P.get("bispec", {}).get("enabled"): other_feats['bsp'] = bispectrum(z, P["bispec"])
    return tm_feat, eng_feat, snr, other_feats, int(label)


def main():
    """Main function to run the preprocessing."""
    hf = h5py.File(C["data"]["raw_path"], "r")
    X_raw, labels = hf["X"], np.argmax(hf["Y"][:], axis=1)
    N = X_raw.shape[0]
    tr_idx, te_idx = train_test_split(np.arange(N), train_size=P["train_ratio"], random_state=P["random_seed"],
                                      stratify=labels)

    for split, idxs in (("train", tr_idx), ("test", te_idx)):
        base = os.path.join(OUT_DIR, split)
        os.makedirs(base, exist_ok=True)
        num_shards = math.ceil(len(idxs) / P["shard_size"])
        for s_idx in range(num_shards):
            chunk_indices = np.sort(idxs[s_idx * P["shard_size"]:(s_idx + 1) * P["shard_size"]])
            m = len(chunk_indices)
            raw_shard = X_raw[chunk_indices].astype("float32")

            X_tm = memmap(f"{base}/X_tm_{s_idx}.npy", (m, 1024, 4), "float32")
            X_eng = memmap(f"{base}/X_eng_{s_idx}.npy", (m, ENG_DIM), "float32")
            y_mm = memmap(f"{base}/y_{s_idx}.npy", (m,), "int64")
            X_snr = memmap(f"{base}/X_snr_{s_idx}.npy", (m,), "float32")
            X_sp = memmap(f"{base}/X_spec_{s_idx}.npy", (m, F_spec, S, 2), "float32")
            X_scat = memmap(f"{base}/X_scat_{s_idx}.npy", (m, S, C_s), "float32")
            feat_maps = {}
            if P["save_dtype"].get("cwt"): feat_maps["cw"] = memmap(f"{base}/X_cwt_{s_idx}.npy",
                                                                    (m, len(P["cwt"]["scales"]), 1024), "float32")
            if P["save_dtype"].get("dwt"): feat_maps["dw"] = memmap(f"{base}/X_dwt_{s_idx}.npy",
                                                                    (m, P["dwt"]["level"], 1024), "float32")
            if P.get("hoc", {}).get("enabled"): feat_maps["hoc"] = memmap(f"{base}/X_hoc_{s_idx}.npy", (
            m, sum(2 if c in {20, 40, 41} else 1 for c in P["hoc"]["orders"])), "float32")
            if P.get("scd", {}).get("enabled"): feat_maps["scd"] = memmap(f"{base}/X_scd_{s_idx}.npy", (
            m, len(P["scd"]["alphas"]), P["scd"]["n_fft"]), "float32")
            if P.get("wpt", {}).get("enabled"): feat_maps["wpt"] = memmap(f"{base}/X_wpt_{s_idx}.npy",
                                                                          (m, 2 ** P["wpt"]["level"] * 2), "float32")
            if P.get("bispec", {}).get("enabled"): feat_maps["bsp"] = memmap(f"{base}/X_bsp_{s_idx}.npy", (
            m, int(P["bispec"]["n_fft"] / P["bispec"].get("downsample", 2)),
            int(P["bispec"]["n_fft"] / P["bispec"].get("downsample", 2))), "float32")

            print(f"Processing {split} shard {s_idx + 1}/{num_shards}...")
            args = [(raw_shard[i], labels[chunk_indices[i]]) for i in range(m)]
            with ProcessPoolExecutor(max_workers=cpu_workers) as exe:
                futures = {exe.submit(extract_cpu_features, a[0], a[1]): i for i, a in enumerate(args)}
                for future in tqdm(as_completed(futures), total=m, desc=f"  [CPU Features]", leave=False):
                    i = futures[future]
                    tm_feat, eng_feat, snr, other_feats, lbl = future.result()
                    X_tm[i], X_eng[i], X_snr[i], y_mm[i] = tm_feat, eng_feat, snr, lbl
                    for key, data in other_feats.items():
                        if key in feat_maps: feat_maps[key][i] = data

            clean_tm_shard = np.array(X_tm)
            B = 512
            for start in tqdm(range(0, m, B), desc=f"  [GPU Transforms]", leave=False):
                end = min(start + B, m)
                batch_I, batch_amp = clean_tm_shard[start:end, :, 0], clean_tm_shard[start:end, :, 2]
                t_I, t_amp = torch.from_numpy(batch_I).to(device), torch.from_numpy(batch_amp).to(device)
                spec_c = stft_transform(t_I)
                X_sp[start:end] = torch.stack([spec_c.real, spec_c.imag], -1)[..., :S, :].cpu().numpy()
                segs = t_amp.unfold(1, L, step)
                scat = scat_torch(segs.contiguous().view(-1, L))
                X_scat[start:end] = scat.mean(dim=-1).view(end - start, S, -1).cpu().numpy()

            all_maps = [X_tm, X_eng, y_mm, X_snr, X_sp, X_scat] + list(feat_maps.values())
            for arr in all_maps:
                if hasattr(arr, 'flush'): arr.flush()

    print("Preprocessing complete.")
    hf.close()


if __name__ == "__main__":
    main()
