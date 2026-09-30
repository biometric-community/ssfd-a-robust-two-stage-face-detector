"""VGG16 backbone for SSFD+ (Sec. 3.1–3.2): remove pool 8→16, keep stride 8."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm


class VGG16Backbone(nn.Module):
    """VGG16 without the final maxpool (stride stays 8 at conv5)."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = tvm.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = tvm.vgg16(weights=weights)
        feats = list(vgg.features.children())
        # Drop maxpool after conv4 (index 23 in standard VGG16 features)
        self.body = nn.Sequential(*feats[:23], *feats[24:30])
        self.out_channels = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)
