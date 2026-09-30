"""Two-stage RPN + ROI losses (Sec. 3, 4.3)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ssfd_plus.models.ssfd_plus import SSFDPlus, box_iou, match_anchors


class SSFDPlusLoss(nn.Module):
    """Stage-1 RPN loss (IoU 0.7/0.3, N=256) + stage-2 ROI loss (IoU 0.5, N=128, 1:3)."""

    def __init__(
        self,
        rpn_pos_iou: float = 0.7,
        rpn_neg_iou: float = 0.3,
        roi_iou: float = 0.5,
        rpn_samples: int = 256,
        roi_samples: int = 128,
        roi_neg_ratio: float = 3.0,
    ) -> None:
        super().__init__()
        self.rpn_pos_iou = rpn_pos_iou
        self.rpn_neg_iou = rpn_neg_iou
        self.roi_iou = roi_iou
        self.rpn_samples = rpn_samples
        self.roi_samples = roi_samples
        self.roi_neg_ratio = roi_neg_ratio

    @staticmethod
    def _encode(anchors: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        acx = (anchors[:, 0] + anchors[:, 2]) / 2
        acy = (anchors[:, 1] + anchors[:, 3]) / 2
        aw = (anchors[:, 2] - anchors[:, 0]).clamp(min=1e-6)
        ah = (anchors[:, 3] - anchors[:, 1]).clamp(min=1e-6)
        gcx = (boxes[:, 0] + boxes[:, 2]) / 2
        gcy = (boxes[:, 1] + boxes[:, 3]) / 2
        gw = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-6)
        gh = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-6)
        return torch.stack(
            [(gcx - acx) / aw, (gcy - acy) / ah, torch.log(gw / aw), torch.log(gh / ah)],
            dim=1,
        )

    def _rpn_loss(self, model: SSFDPlus, out: dict, gt_boxes: torch.Tensor, device):
        rpn_cls, rpn_reg = out["rpn_cls"], out["rpn_reg"]
        _, _, fh, fw = rpn_cls.shape
        anchors = model.generate_anchors(fh, fw, device)
        cls = rpn_cls[0].permute(1, 2, 0).reshape(-1, 2)
        reg = rpn_reg[0].permute(1, 2, 0).reshape(-1, 4)
        if gt_boxes.numel() == 0:
            labels = torch.zeros(cls.shape[0], device=device, dtype=torch.long)
            loss_c = F.cross_entropy(cls, labels)
            return loss_c, loss_c.new_tensor(0.0), 0

        max_iou, gt_idx, pos, neg = match_anchors(
            anchors, gt_boxes, self.rpn_pos_iou, self.rpn_neg_iou
        )
        labels = torch.zeros(anchors.shape[0], device=device, dtype=torch.long)
        labels[pos] = 1

        n_pos = int(pos.sum().item())
        n_neg = min(int(n_pos), int(neg.sum().item()))
        loss_c_all = F.cross_entropy(cls, labels, reduction="none")
        pos_loss = loss_c_all[pos].sum() if n_pos else cls.new_tensor(0.0)
        neg_loss = torch.topk(loss_c_all[neg], n_neg).values.sum() if n_neg else cls.new_tensor(0.0)
        loss_cls = (pos_loss + neg_loss) / max(n_pos + n_neg, 1)

        if n_pos:
            matched = gt_boxes[gt_idx[pos]]
            loss_reg = F.smooth_l1_loss(reg[pos], self._encode(anchors[pos], matched), reduction="sum") / n_pos
        else:
            loss_reg = cls.new_tensor(0.0)
        return loss_cls, loss_reg, n_pos

    def _roi_loss(self, model: SSFDPlus, feat: torch.Tensor, gt_boxes: torch.Tensor, proposals: torch.Tensor):
        device = feat.device
        if proposals.numel() == 0 or gt_boxes.numel() == 0:
            z = feat.new_tensor(0.0)
            return z, z, 0
        ious = box_iou(proposals, gt_boxes)
        max_iou, gt_idx = ious.max(dim=1)
        pos = max_iou >= self.roi_iou
        neg = max_iou < self.roi_iou
        n_pos = int(pos.sum().item())
        n_neg = min(int(n_pos * self.roi_neg_ratio), int(neg.sum().item()))
        sel = pos.clone()
        if n_neg:
            neg_idx = neg.nonzero(as_tuple=False).squeeze(1)
            if neg_idx.numel() > n_neg:
                _, order = max_iou[neg_idx].sort(descending=False)
                keep_neg = neg_idx[order[:n_neg]]
                sel = pos.clone()
                sel[keep_neg] = True
            else:
                sel[neg_idx] = True
        rois = torch.cat([torch.zeros((sel.sum(), 1), device=device), proposals[sel]], dim=1)
        cls_logits, reg = model.roi_forward(feat, rois)
        labels = pos[sel].long()
        loss_cls = F.cross_entropy(cls_logits, labels)
        if n_pos:
            pos_mask = labels == 1
            matched = gt_boxes[gt_idx[sel][pos_mask]]
            loss_reg = F.smooth_l1_loss(reg[pos_mask], self._encode(proposals[sel][pos_mask], matched), reduction="sum") / n_pos
        else:
            loss_reg = feat.new_tensor(0.0)
        return loss_cls, loss_reg, n_pos

    def forward(self, model: SSFDPlus, out: dict, targets: list[dict]):
        device = out["feat"].device
        gt = targets[0]["boxes"].to(device)
        rpn_cls, rpn_reg, n_pos = self._rpn_loss(model, out, gt, device)
        # proposals: top-K RPN objectness (plus GT boxes as guaranteed positives)
        _, _, fh, fw = out["rpn_cls"].shape
        anchors = model.generate_anchors(fh, fw, device)
        cls = out["rpn_cls"][0].permute(1, 2, 0).reshape(-1, 2)
        prob = torch.softmax(cls, dim=1)[:, 1]
        max_props = 256
        k = min(max_props, prob.numel())
        proposals = anchors[prob.topk(k).indices]
        if gt.numel():
            proposals = torch.cat([proposals, gt], dim=0)
        roi_cls, roi_reg, _ = self._roi_loss(model, out["feat"], gt, proposals)
        loss = rpn_cls + rpn_reg + roi_cls + roi_reg
        return {
            "loss": loss,
            "loss_rpn_cls": rpn_cls,
            "loss_rpn_reg": rpn_reg,
            "loss_roi_cls": roi_cls,
            "loss_roi_reg": roi_reg,
        }
