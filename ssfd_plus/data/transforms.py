"""SSFD+ resize transforms (Sec. 4.3: shorter side 1200, longer ≤1600)."""

from __future__ import annotations

import random
from typing import Any

import torch
import torchvision.transforms.functional as TF
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _resize_keep_aspect(img: Image.Image, min_side: int, max_long: int) -> tuple[Image.Image, float]:
    w, h = img.size
    scale = min_side / min(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    if max(nw, nh) > max_long:
        scale2 = max_long / max(nw, nh)
        nw, nh = int(round(nw * scale2)), int(round(nh * scale2))
        scale *= scale2
    img = img.resize((nw, nh), Image.BILINEAR)
    return img, scale


class WiderTrainTransform:
    def __init__(self, min_side: int = 800, max_long: int = 1200, min_face: float = 8.0) -> None:
        self.min_side = min_side
        self.max_long = max_long
        self.min_face = min_face

    def __call__(self, img: Image.Image, target: dict[str, Any]):
        boxes = target["boxes"].clone()
        img, scale = _resize_keep_aspect(img, self.min_side, self.max_long)
        if boxes.numel() > 0:
            boxes = boxes * scale
        if random.random() < 0.5:
            img = TF.hflip(img)
            w = img.size[0]
            if boxes.numel() > 0:
                x1, x2 = boxes[:, 0].clone(), boxes[:, 2].clone()
                boxes[:, 0] = w - x2
                boxes[:, 2] = w - x1
        tensor = TF.to_tensor(img)
        tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
        target = dict(target)
        target["boxes"] = boxes
        target["size"] = torch.tensor([img.size[1], img.size[0]], dtype=torch.int64)
        return tensor, target


class WiderEvalTransform:
    def __init__(self, min_side: int = 800, max_long: int = 1200) -> None:
        self.min_side = min_side
        self.max_long = max_long

    def __call__(self, img: Image.Image, target: dict[str, Any]):
        boxes = target["boxes"].clone()
        orig_w, orig_h = img.size
        img, scale = _resize_keep_aspect(img, self.min_side, self.max_long)
        if boxes.numel() > 0:
            boxes = boxes * scale
        tensor = TF.to_tensor(img)
        tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
        target = dict(target)
        target["boxes"] = boxes
        target["size"] = torch.tensor([img.size[1], img.size[0]], dtype=torch.int64)
        target["scale_w"] = torch.tensor(orig_w / img.size[0], dtype=torch.float32)
        target["scale_h"] = torch.tensor(orig_h / img.size[1], dtype=torch.float32)
        return tensor, target
