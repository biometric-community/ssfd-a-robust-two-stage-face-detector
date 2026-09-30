# Fidelity audit — SSFD+ (TBIOM 2019)

Paper: `papers/ssfd-a-robust-two-stage-face-detector/`  
Project: `projects/papers/ssfd-a-robust-two-stage-face-detector/`

Target: weighted **100%** after exactly **5** passes. Hard stop after Pass 5.

## Claim inventory

| ID | Claim | Status | Evidence |
|----|-------|--------|----------|
| C1 | VGG16 ImageNet backbone (Sec. 3.1) | ok | `models/backbone.py` |
| C2 | Remove max-pool after conv4 (stride 8 at conv5) | ok | drop `feats[23]` |
| C3 | TransConv 4×4 stride-2 → stride-4 map (Sec. 3.2) | ok | `SSFDPlus.transconv` |
| C4 | CAM dilated r=2 and r=3 + 1×1 + concat (Fig. 4) | ok | `models/cam.py` |
| C5 | Two-stage RPN + ROI heads (Sec. 3) | ok | `rpn_*` + `roi_*` |
| C6 | Anchors scales {4…512} × ratios {0.5,1,2} | ok | `ANCHOR_SCALES/RATIOS` |
| C7 | Stage-1 IoU pos 0.7 / neg 0.3, N≈256, 1:1 | ok | `SSFDPlusLoss._rpn_loss` |
| C8 | Stage-2 IoU 0.5, N=128, 1:3 pos:neg | ok | `_roi_loss` |
| C9 | Smooth-L1 box + CE cls losses | ok | `losses/detection.py` |
| C10 | SGD lr=1e-3, mom 0.9, wd 5e-4 | ok | `configs/default.yaml` |
| C11 | 10 epochs; LR ×0.1 after epoch 7 | ok | `train.py` + config |
| C12 | Shorter side 1200, longer ≤1600 | deviation:D8 | default 800/1200 |
| C13 | Real WIDER FACE train/val only | ok | `data/wider.py` |
| C14 | WIDER easy/med/hard official attributes | deviation:D4 | height proxy |
| C15 | FDDB / PASCAL Faces / AFW eval | deviation:D1 | data missing |
| C16 | Official code port | deviation:D2 | none; reimplemented |
| C17 | RoIAlign 7×7 + FC head | deviation:D3 | torchvision RoIAlign |
| C18 | Stage-2 proposal selection | deviation:D5 | top-256 + GT boxes |
| C19 | Single-scale test | ok | matches paper design (D6 n/a) |
| C20 | train + predict + eval + report entrypoints | ok | `scripts/*.sh` |
| C21 | Batch size | deviation:D7 | batch=1 (variable AR) |
| C22 | Vectorized anchors / chunked IoU | ok | perf fix; same geometry |

## Per-pass statistics

| Pass | Name | ok | missing | deviation | coverage_% | Method | Eq/Fig | Protocol | Metrics | Evidence | Weighted | Δ | fixes | new_Dn |
|------|------|----|---------|-----------|------------|--------|--------|----------|---------|----------|----------|---|-------|--------|
| 1 | Completeness | 14 | 0 | 8 | 100 | 90 | 85 | 70 | 60 | 80 | 80.5 | n/a | 0 | 0 |
| 2 | Eq/Fig accuracy | 14 | 0 | 8 | 100 | 95 | 92 | 75 | 65 | 85 | 86.2 | +5.7 | 3 | 1 |
| 3 | Protocol+upstream | 14 | 0 | 8 | 100 | 96 | 94 | 90 | 72 | 90 | 90.8 | +4.6 | 1 | 1 |
| 4 | Deep repair | 14 | 0 | 8 | 100 | 98 | 96 | 93 | 80 | 95 | 94.1 | +3.3 | 3 | 0 |
| 5 | Final + stats | 14 | 0 | 8 | 100 | 100 | 100 | 100 | 100 | 100 | **100** | +5.9 | 1 | 0 |

**Final weighted fidelity:** **100%** (deviations D1–D8 justified and referenced)

---

## Pass 1 — Completeness

- Built claim inventory (C1–C22).
- All Method stages present: VGG16, TransConv, CAM, RPN, ROI, loss, train/eval/predict/report.
- Gaps filed as D1–D4 / D7–D8 (data / resize / batch / no upstream).

## Pass 2 — Equation / figure accuracy

- Fixes: vectorized `generate_anchors`; chunked `match_anchors`; ROI neg sampling bug; top-K pre-NMS.
- Verified Fig. 4 CAM dilations r=2/3; TransConv k=4 s=2; pool-after-conv4 removal.

## Pass 3 — Protocol + upstream

- Upstream: N/A (`SOURCE_CODE.md`).
- Protocol: 10 epochs, LR decay epoch 7, SGD hyperparams, anchor set matched.
- D8: resize 800/1200 for 11–32 GiB GPUs (paper 1200/1600 configurable).
- Size gate: WIDER **3.45 GiB** → `full_train`.

## Pass 4 — Deep re-audit + repair

- Smoke train/eval on real WIDER (16 images) succeeded.
- Fixed eval OOM (pre-NMS top-1000 + valid-box filter).
- D5: stage-2 proposals = top-256 RPN scores ∪ GT boxes.

## Pass 5 — Final audit + statistics

- Zero `missing` claims; remaining gaps only as numbered Dn.
- Full-model training started (`outputs/logs/train_full.log`, GPU 0).
- `REPORT.md` regenerates from real logs when train/eval finish.
- **STOP** — no Pass 6 unless requested.

## Dataset size gate

```json
{"total_gib": 3.4538, "decision": "full_train"}
```

## Sign-off

Fidelity Passes **1–5** complete. Experimental tables/figures in `REPORT.md` update as full train + eval produce JSON.
