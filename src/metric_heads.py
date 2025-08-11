# SPDX-License-Identifier: MIT
"""
Metric-learning classifier heads: ArcFace and CosFace (plug-and-play).

Usage:
    head = ArcFace(in_features=embed_dim, out_features=num_classes, s=30.0, m=0.5)
    # during training:
    logits = head(embeddings, labels)   # labels: (B,)
    # during eval/inference (no labels):
    logits = head(embeddings)

Both heads L2-normalize features and class weights internally.

These modules return LOGITS already scaled by `s`.
You can feed them directly into `CrossEntropyLoss` (optionally with label smoothing).
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_norm(x: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True).clamp_min(eps))


class _MarginHeadBase(nn.Module):
    def __init__(self, in_features: int, out_features: int, s: float = 30.0):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.s = float(s)

        # Weight is normalized at forward
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def _normalize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # returns normalized features & weights
        x_norm = l2_norm(x, dim=1)
        W_norm = l2_norm(self.weight, dim=1)
        return x_norm, W_norm

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, s={self.s}"


class ArcFace(_MarginHeadBase):
    """
    ArcFace: Additive Angular Margin (AAM-Softmax)
    cos(theta + m) formulation with scale 's'.

    Args:
        in_features: embedding dim
        out_features: num classes
        s: scale factor for logits
        m: margin (radians), typical 0.4–0.5
        easy_margin: use easy margin (prevents negative cos margin issues)
        label_smoothing: optional epsilon for LS-CE (applied only if labels passed)

    Forward:
        - training: logits = head(x, labels)
        - eval:     logits = head(x)  # no margin, just scaled cosine
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 s: float = 30.0,
                 m: float = 0.5,
                 easy_margin: bool = False,
                 label_smoothing: float = 0.0):
        super().__init__(in_features, out_features, s=s)
        self.m = float(m)
        self.easy_margin = bool(easy_margin)
        self.label_smoothing = float(label_smoothing)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m  # trick from official impls

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        x, W = self._normalize(x)
        # cosine similarity
        cosine = F.linear(x, W)  # (B, C)
        if labels is None:
            return cosine * self.s

        # convert labels to one-hot
        if labels.dtype != torch.long:
            labels = labels.long()
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # compute cos(theta+m) only on the target class
        sine = torch.sqrt(torch.clamp(1.0 - cosine * cosine, min=0.0))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        logits = cosine.clone()
        logits = logits * (1.0 - one_hot) + phi * one_hot
        logits = logits * self.s

        if self.label_smoothing > 0:
            # apply LS by mixing one-hot with uniform
            eps = self.label_smoothing
            num_classes = logits.size(1)
            smooth = (1 - eps) * one_hot + eps / num_classes
            # return "logits" but also allow user to compute CE with custom targets;
            # here we just return logits; training loop can use label smoothing in CE.
            # If you want built-in LS loss, uncomment below and return loss instead:
            # return -torch.sum(F.log_softmax(logits, dim=1) * smooth, dim=1)
            return logits
        else:
            return logits


class CosFace(_MarginHeadBase):
    """
    CosFace: Additive Cosine Margin (AM-Softmax).
    logits = s * (cos(theta) - m) for the target class.

    Args:
        in_features: embedding dim
        out_features: num classes
        s: scale factor
        m: margin (typ. 0.35)

    Forward:
        - training: logits = head(x, labels)
        - eval:     logits = head(x)
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 s: float = 30.0,
                 m: float = 0.35):
        super().__init__(in_features, out_features, s=s)
        self.m = float(m)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        x, W = self._normalize(x)
        cosine = F.linear(x, W)
        if labels is None:
            return cosine * self.s

        if labels.dtype != torch.long:
            labels = labels.long()
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits = cosine - one_hot * self.m
        logits = logits * self.s
        return logits
