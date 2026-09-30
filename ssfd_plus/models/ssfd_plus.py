"""SSFD+ two-stage detector (Fig. 4–5, Sec. 3)."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import VGG16Backbone
from .cam import ContextAgglomerationModule


def box_iou(a: torch.Tensor, b: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
    """Pairwise IoU; chunks over `a` rows to limit peak memory."""
    if a.shape[0] <= chunk:
        return _box_iou_dense(a, b)
    parts = [_box_iou_dense(a[i : i + chunk], b) for i in range(0, a.shape[0], chunk)]
    return torch.cat(parts, dim=0)


def _box_iou_dense(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-6)


def match_anchors(
    anchors: torch.Tensor,
    gt_boxes: torch.Tensor,
    pos_iou: float,
    neg_iou: float,
    chunk: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return max_iou, gt_idx, pos, neg without storing the full IoU matrix."""
    n_gt = gt_boxes.shape[0]
    n_a = anchors.shape[0]
    device = anchors.device
    max_iou = torch.zeros(n_a, device=device)
    gt_idx = torch.zeros(n_a, dtype=torch.long, device=device)
    best_iou = torch.full((n_gt,), -1.0, device=device)
    best_anchor = torch.zeros(n_gt, dtype=torch.long, device=device)
    for start in range(0, n_a, chunk):
        end = min(start + chunk, n_a)
        iou = _box_iou_dense(anchors[start:end], gt_boxes)
        mv, mi = iou.max(dim=1)
        max_iou[start:end] = mv
        gt_idx[start:end] = mi
        gmax, garg = iou.max(dim=0)
        improve = gmax > best_iou
        best_iou = torch.where(improve, gmax, best_iou)
        best_anchor = torch.where(improve, start + garg, best_anchor)
    pos = max_iou >= pos_iou
    pos[best_anchor] = True
    neg = max_iou < neg_iou
    return max_iou, gt_idx, pos, neg


def nms(boxes: torch.Tensor, scores: torch.Tensor, thresh: float) -> torch.Tensor:
    try:
        from torchvision.ops import nms as tv_nms

        return tv_nms(boxes, scores, thresh)
    except Exception:
        order = scores.argsort(descending=True)
        keep = []
        while order.numel() > 0:
            i = order[0]
            keep.append(i)
            if order.numel() == 1:
                break
            rest = order[1:]
            iou = box_iou(boxes[i].unsqueeze(0), boxes[rest]).squeeze(0)
            order = rest[iou <= thresh]
        return torch.tensor(keep, device=boxes.device, dtype=torch.long)


class SSFDPlus(nn.Module):
    """Two-stage face detector: VGG16 → TransConv → CAM → RPN → ROI head."""

    ANCHOR_SCALES = (4, 8, 16, 32, 64, 128, 256, 512)
    ANCHOR_RATIOS = (0.5, 1.0, 2.0)
    STRIDE = 4  # after TransConv (Sec. 3.2)

    def __init__(self, pretrained_backbone: bool = True, use_cam: bool = True) -> None:
        super().__init__()
        self.use_cam = use_cam
        self.backbone = VGG16Backbone(pretrained=pretrained_backbone)
        self.transconv = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=True)
        self.cam = ContextAgglomerationModule(512)
        feat_ch = self.cam.out_channels if use_cam else 512
        n_anchors = len(self.ANCHOR_SCALES) * len(self.ANCHOR_RATIOS)
        self.rpn_cls = nn.Conv2d(feat_ch, n_anchors * 2, kernel_size=3, padding=1)
        self.rpn_reg = nn.Conv2d(feat_ch, n_anchors * 4, kernel_size=3, padding=1)
        self.roi_fc = nn.Sequential(
            nn.Linear(feat_ch * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        self.roi_cls = nn.Linear(4096, 2)
        self.roi_reg = nn.Linear(4096, 4)
        self.n_anchors = n_anchors
        self._init_heads()

    def _init_heads(self) -> None:
        for m in [self.rpn_cls, self.rpn_reg, self.transconv]:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.roi_cls.weight, std=0.01)
        nn.init.constant_(self.roi_cls.bias, 0)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)
        f = F.relu(self.transconv(f), inplace=True)
        if self.use_cam:
            f = self.cam(f)
        return f

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.extract_features(x)
        rpn_cls = self.rpn_cls(feat)
        rpn_reg = self.rpn_reg(feat)
        return {"feat": feat, "rpn_cls": rpn_cls, "rpn_reg": rpn_reg}

    def roi_forward(self, feat: torch.Tensor, rois: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ROI align + fc head (stage 2). rois: (N,5) [batch_idx,x1,y1,x2,y2]."""
        from torchvision.ops import roi_align

        pooled = roi_align(feat, rois, output_size=(7, 7), spatial_scale=1.0 / self.STRIDE)
        flat = pooled.flatten(start_dim=1)
        h = self.roi_fc(flat)
        return self.roi_cls(h), self.roi_reg(h)

    def generate_anchors(self, fh: int, fw: int, device) -> torch.Tensor:
        """Vectorized anchors: (fh*fw*A, 4) in xyxy on the image canvas."""
        scales = torch.tensor(self.ANCHOR_SCALES, device=device, dtype=torch.float32)
        ratios = torch.tensor(self.ANCHOR_RATIOS, device=device, dtype=torch.float32)
        # base widths/heights for each (scale, ratio)
        s = scales[:, None]  # (S,1)
        r = ratios[None, :]  # (1,R)
        ws = (s * torch.sqrt(r)).reshape(-1)  # (A,)
        hs = (s / torch.sqrt(r)).reshape(-1)
        shift_x = (torch.arange(fw, device=device, dtype=torch.float32) + 0.5) * self.STRIDE
        shift_y = (torch.arange(fh, device=device, dtype=torch.float32) + 0.5) * self.STRIDE
        cy, cx = torch.meshgrid(shift_y, shift_x, indexing="ij")
        centers = torch.stack([cx.reshape(-1), cy.reshape(-1)], dim=1)  # (H*W, 2)
        # broadcast: (H*W, A, 4)
        cx_ = centers[:, None, 0]
        cy_ = centers[:, None, 1]
        boxes = torch.stack(
            [cx_ - ws / 2, cy_ - hs / 2, cx_ + ws / 2, cy_ + hs / 2],
            dim=-1,
        )
        return boxes.reshape(-1, 4)

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        score_thresh: float = 0.5,
        nms_thresh: float = 0.5,
        top_k: int = 300,
    ) -> list[dict[str, torch.Tensor]]:
        self.eval()
        out = self.forward(images)
        feat, rpn_cls, rpn_reg = out["feat"], out["rpn_cls"], out["rpn_reg"]
        _, _, fh, fw = rpn_cls.shape
        device = images.device
        h_img, w_img = images.shape[-2:]
        results = []
        for b in range(images.shape[0]):
            anchors = self.generate_anchors(fh, fw, device)
            cls = rpn_cls[b].permute(1, 2, 0).reshape(-1, 2)
            reg = rpn_reg[b].permute(1, 2, 0).reshape(-1, 4)
            prob = torch.softmax(cls, dim=1)[:, 1]
            # decode boxes
            acx = (anchors[:, 0] + anchors[:, 2]) / 2
            acy = (anchors[:, 1] + anchors[:, 3]) / 2
            aw = (anchors[:, 2] - anchors[:, 0]).clamp(min=1e-6)
            ah = (anchors[:, 3] - anchors[:, 1]).clamp(min=1e-6)
            dx, dy, dw, dh = reg[:, 0], reg[:, 1], reg[:, 2], reg[:, 3]
            pcx = dx * aw + acx
            pcy = dy * ah + acy
            pw = torch.exp(dw.clamp(max=4)) * aw
            ph = torch.exp(dh.clamp(max=4)) * ah
            boxes = torch.stack([pcx - pw / 2, pcy - ph / 2, pcx + pw / 2, pcy + ph / 2], dim=1)
            mask = prob > score_thresh
            if not mask.any():
                results.append({"boxes": torch.zeros((0, 4), device=device), "scores": torch.zeros(0, device=device)})
                continue
            boxes, scores = boxes[mask], prob[mask]
            # Pre-filter top candidates before NMS (avoids OOM on dense anchors)
            pre_nms = min(1000, scores.numel())
            if scores.numel() > pre_nms:
                top = scores.topk(pre_nms).indices
                boxes, scores = boxes[top], scores[top]
            boxes[:, 0::2].clamp_(0, w_img - 1)
            boxes[:, 1::2].clamp_(0, h_img - 1)
            # drop degenerate boxes
            valid = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
            boxes, scores = boxes[valid], scores[valid]
            if boxes.numel() == 0:
                results.append({"boxes": torch.zeros((0, 4), device=device), "scores": torch.zeros(0, device=device)})
                continue
            keep = nms(boxes, scores, nms_thresh)[:top_k]
            results.append({"boxes": boxes[keep], "scores": scores[keep]})
        return results
