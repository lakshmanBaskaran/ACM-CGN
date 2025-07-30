# SPDX-License-Identifier: MIT
import torch
import torch.nn as nn

class MultiTaskLoss(nn.Module):
    """
    Combines classification (CE) with optional reconstruction (MSE) loss.
    """
    def __init__(self, alpha=1.0):
        super(MultiTaskLoss, self).__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def forward(self, logits, labels, denoised=None, clean_input=None):
        loss = self.ce(logits, labels)
        if denoised is not None and clean_input is not None:
            loss = loss + self.alpha * self.mse(denoised, clean_input)
        return loss
