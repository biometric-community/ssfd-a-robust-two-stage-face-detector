"""Context Agglomeration Module (CAM; Fig. 4, Sec. 3.3)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextAgglomerationModule(nn.Module):
    """Two dilated conv branches (r=2, r=3) + 1×1 convs + concat."""

    def __init__(self, channels: int = 512) -> None:
        super().__init__()
        self.dila2 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.dila3 = nn.Conv2d(channels, channels, kernel_size=3, padding=3, dilation=3, bias=True)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.out_channels = channels * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = F.relu(self.conv2(F.relu(self.dila2(x))), inplace=True)
        b = F.relu(self.conv3(F.relu(self.dila3(x))), inplace=True)
        return torch.cat([a, b], dim=1)
