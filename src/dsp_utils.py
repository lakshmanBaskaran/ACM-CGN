# SPDX-License-Identifier: MIT
import os
import math
import numpy as np
import pywt
from scipy.signal import stft, get_window, butter, sosfilt, wiener, convolve

def bandpass_filter(iq: np.ndarray, low: float, high: float, fs: float, order: int = 4) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter to the I/Q channels.
    iq: (T,2) array of I/Q samples
    """
    sos = butter(order, [low, high], btype='band', fs=fs, output='sos')
    return sosfilt(sos, iq, axis=0)

def wiener_filter(iq: np.ndarray, mysize) -> np.ndarray:
    """
    Apply Wiener filter to each I/Q channel independently.
    mysize: window size (scalar or tuple)
    """
    out = np.zeros_like(iq)
    for c in range(iq.shape[1]):
        out[:, c] = wiener(iq[:, c], mysize=mysize)
    return out

def matched_filter(iq: np.ndarray, pulse: np.ndarray) -> np.ndarray:
    """
    Matched filter by convolving with the time-reversed pulse shape.
    iq: (T,2), pulse: (L,)
    """
    kernel = pulse[::-1]
    out = np.zeros_like(iq)
    for c in range(iq.shape[1]):
        out[:, c] = convolve(iq[:, c], kernel, mode='same')
    return out

def stft_spec(z, p):
    win = get_window(p["window"], p["n_fft"], fftbins=True)
    _, _, Z = stft(
        z, nperseg=p["n_fft"],
        noverlap=p["n_fft"] - p["hop_length"],
        window=win, return_onesided=False, padded=False,
    )
    return np.stack([np.abs(Z), np.angle(Z)], -1).astype("float32")

def cwt_spec(z, scales):
    coeffs, _ = pywt.cwt(z, scales, "morl", axis=0)
    return np.abs(coeffs).astype("float32")

def wavelet_denoise(x, wavelet, level, mode="soft"):
    coeffs = pywt.wavedec(x, wavelet, axis=0, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(x.shape[0]))
    den = [coeffs[0]]
    for c in coeffs[1:]:
        den.append(pywt.threshold(c, uthresh, mode=mode))
    return pywt.waverec(den, wavelet, axis=0)

def dwt_spec(x, wavelet, level):
    coeffs = pywt.swt(x, wavelet, level=level)
    details = [np.abs(cD) for (_cA, cD) in coeffs]
    return np.stack(details, axis=0).astype("float32")

def estimate_snr(x):
    power = np.mean(x[:,0]**2 + x[:,1]**2)
    noise_power = np.var(np.concatenate((x[:,0], x[:,1])))
    if noise_power < 1e-12:
        return 50.0 # High SNR if noise is negligible
    return 10 * math.log10(power / noise_power)

def hoc_features(z, cfg):
    orders = cfg["orders"]
    pwr = np.mean(np.abs(z)**2) + 1e-12
    m2, m4 = np.mean(z**2), np.mean(z**4)
    r2, r4, r6 = np.mean(np.abs(z)**2), np.mean(np.abs(z)**4), np.mean(np.abs(z)**6)
    C20, C21 = m2, r2
    C40 = m4 - 3*(m2**2)
    C41 = np.mean((np.abs(z)**2)*(z**2)) - 2*m2*r2
    C42 = r4 - np.abs(m2)**2 - 2*(r2**2)
    C63 = r6 - 9*r2*r4 + 12*(r2**3)
    lookup = {20:C20, 21:C21, 40:C40, 41:C41, 42:C42, 63:C63}
    feats = []
    for code in orders:
        v = lookup[code]
        if cfg.get("normalize", True):
            v = v / (pwr**(code//20))
        if np.iscomplexobj(v):
            feats.extend([v.real, v.imag])
        else:
            feats.append(v)
    return np.asarray(feats, dtype="float32")

def scd_matrix(z, cfg):
    nfft = int(cfg["n_fft"])
    alphas = cfg["alphas"]
    X = np.fft.fft(z, nfft)
    F = X.shape[0]
    out = []
    for a in alphas:
        shift = int(round(a * F/2.0))
        S = np.roll(X, +shift) * np.conj(np.roll(X, -shift))
        if cfg.get("take_abs", True):
            out.append(np.abs(S).astype("float32"))
        else:
            out.append(np.stack([S.real, S.imag], -1).astype("float32"))
    return np.stack(out, 0)

def wpt_energy(z, cfg):
    from pywt import WaveletPacket
    level, wavelet, agg = int(cfg["level"]), cfg["wavelet"], cfg.get("agg","energy").lower()
    wp = WaveletPacket(data=z, wavelet=wavelet, mode="symmetric", maxlevel=level)
    leaves = [n.path for n in wp.get_level(level, order='freq')]
    feats = []
    for p in leaves:
        data = wp[p].data
        if not np.iscomplexobj(data): data = data.astype(np.complex64)
        if agg == "energy":
            feats.extend([np.sum(np.real(data)**2), np.sum(np.imag(data)**2)])
        else: # mean
            feats.extend([np.mean(np.real(data)), np.mean(np.imag(data))])
    return np.asarray(feats, dtype="float32")

def bispectrum(z, cfg):
    nfft, ds = int(cfg["n_fft"]), int(cfg.get("downsample", 2))
    X = np.fft.fft(z, nfft)
    idx = np.arange(0, nfft, ds)
    Fd = len(idx)
    B = np.zeros((Fd, Fd), dtype=np.complex64)
    for i, f1 in enumerate(idx):
        for j, f2 in enumerate(idx):
            k = (f1 + f2) % nfft
            B[i, j] = X[f1] * X[f2] * np.conj(X[k])
    if cfg.get("take_abs", True):
        return np.abs(B).astype("float32")
    return np.stack([B.real.astype("float32"), B.imag.astype("float32")], -1)

def memmap(path, shape, dtype="float32"):
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    return np.lib.format.open_memmap(path, "w+", dtype=dtype, shape=shape)
