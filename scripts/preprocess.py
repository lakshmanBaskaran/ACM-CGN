# SPDX-License-Identifier: MIT
import os
import math
import yaml
import h5py
import numpy as np
import pywt
from sklearn.model_selection import train_test_split
from scipy.signal import stft, get_window
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# ─── Helpers ────────────────────────────────────────────────────────────────
def stft_spec(z, p):
    win = get_window(p["window"], p["n_fft"], fftbins=True)
    _, _, Z = stft(
        z,
        nperseg=p["n_fft"],
        noverlap=p["n_fft"] - p["hop_length"],
        window=win,
        return_onesided=False,
        padded=False,
    )
    return np.stack([np.abs(Z), np.angle(Z)], -1).astype("float32")

def cwt_spec(z, scales):
    coeffs, _ = pywt.cwt(z, scales, "morl", axis=0)
    return np.abs(coeffs).astype("float32")

def wavelet_denoise(x, wavelet, level, mode="soft"):
    coeffs = pywt.wavedec(x, wavelet, axis=0, level=level)
    sigma  = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh= sigma * np.sqrt(2 * np.log(x.shape[0]))
    denoised = [coeffs[0]]
    for c in coeffs[1:]:
        denoised.append(pywt.threshold(c, uthresh, mode=mode))
    return pywt.waverec(denoised, wavelet, axis=0)

def dwt_spec(x, wavelet, level):
    coeffs = pywt.swt(x, wavelet, level=level)
    details = [np.abs(cD) for (_cA, cD) in coeffs]
    return np.stack(details, axis=0).astype("float32")

def estimate_snr(x):
    I, Q = x[:,0], x[:,1]
    power = (I*I + Q*Q).mean()
    noise = np.var(I - I.mean()) + np.var(Q - Q.mean()) + 1e-12
    return 10 * math.log10(power / noise)

def memmap(path, shape, dtype="float32"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return np.lib.format.open_memmap(path, "w+", dtype=dtype, shape=shape)

# ─── Load config & prepare splits ────────────────────────────────────────────
with open("configs/config.yaml", "r", encoding="utf-8") as f:
    C = yaml.safe_load(f)
P = C["preprocess"]
out_dir = C["data"]["processed_dir"]

with h5py.File(C["data"]["raw_path"], "r") as hf:
    labels = np.argmax(hf["Y"][:], axis=1)
N = labels.shape[0]

all_idxs = np.arange(N)
tr_idx, te_idx = train_test_split(
    all_idxs,
    train_size=P["train_ratio"],
    random_state=P["random_seed"],
    stratify=labels
)

# ─── Per-shard worker ────────────────────────────────────────────────────────
def process_shard(args):
    split, shard_id, idxs = args
    hf = h5py.File(C["data"]["raw_path"], "r")
    X_raw = hf["X"]
    lbls  = np.argmax(hf["Y"][:], axis=1)

    base = os.path.join(out_dir, split)
    os.makedirs(base, exist_ok=True)

    m = len(idxs)
    X_tm = memmap(f"{base}/X_tm_{shard_id}.npy", (m, 1024, 4), C["preprocess"]["save_dtype"]["tm"])
    F, _, _ = stft_spec(np.zeros(1024), P["spectrogram"]).shape
    X_sp = memmap(
        f"{base}/X_spec_{shard_id}.npy",
        (m, F, C["segmentation"]["num_segments"], 2),
        C["preprocess"]["save_dtype"]["spec"]
    )
    has_cwt = C["preprocess"]["save_dtype"]["cwt"] is not None
    if has_cwt:
        X_cw = memmap(
            f"{base}/X_cwt_{shard_id}.npy",
            (m, len(P["cwt"]["scales"]), 1024),
            C["preprocess"]["save_dtype"]["cwt"]
        )
    X_dw = memmap(
        f"{base}/X_dwt_{shard_id}.npy",
        (m, P["dwt"]["level"], 1024),
        C["preprocess"]["save_dtype"]["dwt"]
    )
    y_mm = memmap(f"{base}/y_{shard_id}.npy", (m,), "int64")

    for i, idx in enumerate(idxs):
        iq = X_raw[idx].astype("float32")
        iq /= (np.sqrt((iq**2).mean()) + 1e-9)

        snr = estimate_snr(iq)
        if snr < P["denoise"]["threshold_db"]:
            iq = wavelet_denoise(iq, P["denoise"]["wavelet"], P["denoise"]["level"])

        I, Q = iq[:,0], iq[:,1]
        amp  = np.sqrt(I*I + Q*Q)
        ph   = np.arctan2(Q, I)
        X_tm[i] = np.stack([I, Q, amp, ph], axis=1)

        spec = stft_spec(I + 1j*Q, P["spectrogram"])
        segs = C["segmentation"]["num_segments"]
        if spec.shape[1] >= segs:
            X_sp[i] = spec[:, :segs, :]
        else:
            pad = np.zeros((F, segs - spec.shape[1], 2), dtype=spec.dtype)
            X_sp[i] = np.concatenate([spec, pad], axis=1)

        if has_cwt:
            X_cw[i] = cwt_spec(I + 1j*Q, P["cwt"]["scales"])

        X_dw[i] = dwt_spec(I + 1j*Q, P["dwt"]["wavelet"], P["dwt"]["level"])
        y_mm[i] = lbls[idx]

    for arr in (X_tm, X_sp, *( [X_cw] if has_cwt else [] ), X_dw, y_mm):
        arr.flush()

    hf.close()

# ─── Run in parallel ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    tasks = []
    for split, idxs in (("train", tr_idx), ("test", te_idx)):
        num_shards = math.ceil(len(idxs) / P["shard_size"])
        for s in range(num_shards):
            chunk = idxs[s*P["shard_size"] : (s+1)*P["shard_size"]]
            tasks.append((split, s, chunk))

    with ProcessPoolExecutor() as exe:
        list(tqdm(exe.map(process_shard, tasks), total=len(tasks), desc="All shards"))

    print("Preprocessing complete on **full** SNR range.")
