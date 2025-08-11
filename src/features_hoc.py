# SPDX-License-Identifier: MIT
"""
Fast higher-order cumulants (HOC) + tiny cyclostationary features for I/Q signals.

This module exposes:
  - extract_hoc_features(iq)          -> ndarray [C20, C21, C40, C41, C42] (real/imag split)
  - extract_cyclo_tiny(iq, alphas)    -> small spectral-correlation energies
  - extract_hoc_cyclo(iq, alphas)     -> concatenated HOC + cyclo vector

Input `iq` can be:
  - numpy array of shape (T, 2) for [I, Q]
  - numpy array of shape (T,) complex64/complex128
  - torch.Tensor in the same shapes (will be moved to CPU for numpy ops)

All features are real-valued; complex cumulants are split into [real, imag].
"""

from __future__ import annotations
import numpy as np
import torch
from typing import Iterable


def _to_complex(iq) -> np.ndarray:
    if isinstance(iq, torch.Tensor):
        iq = iq.detach().cpu().numpy()
    iq = np.asarray(iq)
    if np.iscomplexobj(iq):
        return iq.astype(np.complex64, copy=False)
    # assume last dim has I,Q
    if iq.ndim == 2 and iq.shape[-1] >= 2:
        I = iq[..., 0].astype(np.float32, copy=False)
        Q = iq[..., 1].astype(np.float32, copy=False)
        return (I + 1j * Q).astype(np.complex64, copy=False)
    raise ValueError("`iq` must be complex or [...,2] with I/Q in last dimension.")


def _centralize(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x)


def _moments(x: np.ndarray):
    # moments up to 4th order
    x1 = np.mean(x)
    x2 = np.mean(x * x)
    x3 = np.mean(x * x * x)
    x4 = np.mean(x * x * x * x)
    return x1, x2, x3, x4


def _cum_complex(iq: np.ndarray):
    """
    Common complex cumulants for mod-rec:
      C20 = cum2(x, x)          (non-conjugate 2nd-order)
      C21 = cum2(x, x*)         (conjugate 2nd-order == power if centered)
      C40, C41, C42 variants (normalized-ish)

    We estimate via sample moments on centralized data.
    """
    x = _centralize(iq.astype(np.complex64, copy=False))
    N = x.size
    if N < 16:
        raise ValueError("Signal too short for cumulants")

    # 2nd order
    C20 = np.mean(x * x)
    C21 = np.mean(x * np.conj(x))  # = power

    # 4th order cumulants (simplified unbiased-ish estimators)
    # refs: Nikias & Mendel, "Signal processing with HOC"
    m20 = C20
    m21 = C21
    m40 = np.mean((x ** 4))
    m41 = np.mean((x ** 3) * np.conj(x))
    m42 = np.mean((x ** 2) * (np.conj(x) ** 2))

    # cumulant relationships
    cum40 = m40 - 3 * (m20 ** 2)
    cum41 = m41 - 3 * m20 * m21
    cum42 = m42 - (abs(m20) ** 2) - 2 * (m21 ** 2)

    return C20, C21, cum40, cum41, cum42


def extract_hoc_features(iq) -> np.ndarray:
    """
    Returns a real-valued vector:
      [Re(C20), Im(C20), Re(C21), Im(C21),
       Re(C40), Im(C40), Re(C41), Im(C41), Re(C42), Im(C42)]
    """
    x = _to_complex(iq)
    C20, C21, C40, C41, C42 = _cum_complex(x)
    feats = np.array([
        np.real(C20), np.imag(C20),
        np.real(C21), np.imag(C21),
        np.real(C40), np.imag(C40),
        np.real(C41), np.imag(C41),
        np.real(C42), np.imag(C42),
    ], dtype=np.float32)
    # rudimentary normalization for scale invariance
    scale = np.linalg.norm(feats) + 1e-8
    return feats / scale


def extract_cyclo_tiny(iq, alphas: Iterable[float] = (0.25, 0.5, 1.0), fft_len: int | None = None) -> np.ndarray:
    """
    Super tiny cyclostationary proxy:
    - compute |FFT|^2 (instantaneous power) then take FFT again to see periodicities.
    - sample a few bins near alpha * N (if symbol rate roughly known, alpha in [0..1] w.r.t. unknown fs)

    This is intentionally light-weight; not a full SCD. Good enough as a weak feature.
    """
    x = _to_complex(iq)
    N = x.size
    if fft_len is None:
        fft_len = int(1 << int(np.ceil(np.log2(max(64, N)))))

    X = np.fft.fft(x, n=fft_len)
    P = np.abs(X) ** 2
    C = np.fft.fft(P, n=fft_len)

    mags = []
    for a in alphas:
        bin_idx = int(round((a % 1.0) * fft_len))
        mags.append(np.abs(C[bin_idx]))
    mags = np.array(mags, dtype=np.float32)
    mags /= (np.linalg.norm(mags) + 1e-8)
    return mags


def extract_hoc_cyclo(iq, alphas: Iterable[float] = (0.25, 0.5, 1.0)) -> np.ndarray:
    hoc = extract_hoc_features(iq)
    cyc = extract_cyclo_tiny(iq, alphas=alphas)
    return np.concatenate([hoc, cyc], axis=0).astype(np.float32)
