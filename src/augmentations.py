# SPDX-License-Identifier: MIT
import torch
import numpy as np
import math

def random_phase_shift(signal: torch.Tensor, max_phase: float = np.pi) -> torch.Tensor:
    B, C, T = signal.shape # C==2
    theta = (torch.rand(B, 1, 1, device=signal.device) * 2 - 1) * max_phase
    cos, sin = torch.cos(theta), torch.sin(theta)
    R = torch.stack([cos, -sin, sin, cos], dim=-1).view(B, 2, 2)
    return torch.einsum('bij,bjt->bit', R, signal)

def frequency_shift(signal: torch.Tensor, max_shift: float = 0.05) -> torch.Tensor:
    B, C, T = signal.shape
    t = torch.arange(T, device=signal.device, dtype=torch.float32).view(1, 1, -1)
    delta = (torch.rand(B, 1, 1, device=signal.device) * 2 - 1) * max_shift
    phi = 2 * math.pi * delta * t
    cos, sin = torch.cos(phi), torch.sin(phi)
    I, Q = signal[:, 0:1], signal[:, 1:2] # Keep channel dim
    real = I * cos - Q * sin
    imag = I * sin + Q * cos
    return torch.cat([real, imag], dim=1)

def time_shift(signal: torch.Tensor, max_shift: int = 128) -> torch.Tensor:
    B, C, T = signal.shape
    out = torch.empty_like(signal)
    for i in range(B):
        s = torch.randint(0, max_shift, (1,), device=signal.device).item()
        out[i] = torch.roll(signal[i], shifts=s, dims=1)
    return out

def add_noise(signal: torch.Tensor, std_max: float = 0.1, prob: float = 0.5) -> torch.Tensor:
    if torch.rand(1).item() < prob:
        noise_std = torch.rand(1).item() * std_max
        noise = torch.randn_like(signal) * noise_std
        return signal + noise
    return signal

def sigaugment(signal: torch.Tensor, aug_cfg: dict) -> torch.Tensor:
    # signal: (B, 2, T)
    if aug_cfg.get("phase_rotate_prob", 0) > 0 and torch.rand(1).item() < aug_cfg["phase_rotate_prob"]:
        signal = random_phase_shift(signal)
    if aug_cfg.get("freq_shift_prob", 0) > 0 and torch.rand(1).item() < aug_cfg["freq_shift_prob"]:
        signal = frequency_shift(signal, max_shift=aug_cfg.get("freq_shift_max", 0.05))
    if aug_cfg.get("time_shift_prob", 0) > 0 and torch.rand(1).item() < aug_cfg["time_shift_prob"]:
        signal = time_shift(signal, max_shift=aug_cfg.get("time_shift_max", 128))
    signal = add_noise(signal,
                       std_max=aug_cfg.get("noise_std_max", 0.1),
                       prob=aug_cfg.get("noise_prob", 0.0))
    return signal
