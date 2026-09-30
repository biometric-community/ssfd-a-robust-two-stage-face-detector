# Implementation plan: SSFD+ — A Robust Two-Stage Face Detector

- **Paper:** `papers/ssfd-a-robust-two-stage-face-detector/`
- **Project:** `projects/papers/ssfd-a-robust-two-stage-face-detector/`
- **Stack:** Python + PyTorch
- **Upstream:** none official (TBIOM 2019; no author code release found)

## Components to build

1. **VGG16 backbone (Sec. 3, Fig. 4)** — ImageNet init; remove max-pool after conv4 (stride-8 map).
2. **TransConv (Sec. 3.2)** — 4×4 stride-2 deconv to stride-4 feature map.
3. **CAM (Sec. 3.3, Fig. 4)** — dilated conv r=2 and r=3, element-wise sum fusion.
4. **RPN head (stage 1)** — anchors scales `{4…512}` × ratios `{0.5,1,2}`; IoU 0.7/0.3; N=256.
5. **ROI head (stage 2)** — RoIAlign 7×7 + FC; IoU 0.5; N=128; 1:3 pos:neg.
6. **WIDER FACE loader** — real data under `projects/datasets/wider-face/`.
7. **Train / eval / predict / report** — JSON metrics + PR / mAP figures (dev-plot).

## Data

| Dataset | Split | Path | Loader notes |
|---------|-------|------|--------------|
| WIDER FACE | train | `projects/datasets/wider-face/extracted/WIDER_train` + `wider_face_train_bbx_gt.txt` | primary train |
| WIDER FACE | val | `.../WIDER_val` + `wider_face_val_bbx_gt.txt` | easy/medium/hard mAP |

## Training protocol (paper Sec. 4.3)

- Optimizer: SGD lr=1e-3, momentum 0.9, weight decay 5e-4
- Schedule: 10 epochs; LR ×0.1 after epoch 7
- Input resize: shorter side **1200**, longer ≤ **1600** (smoke/default uses 800/1200 — D8)
- Batch size: 1 (variable aspect-ratio images; D7)
- Primary metric: WIDER FACE val mAP easy/medium/hard

## Fidelity plan

- Claim inventory: VGG16, pool removal, TransConv, CAM, RPN/ROI two-stage, anchor set, sampling rules, SGD schedule
- Passes 1–5 as skill requires; hard stop after Pass 5

## Results figure inventory

| Paper fig | Type | Our path |
|-----------|------|----------|
| Table 1 style | grouped bars | `outputs/figures/wider_map_bars.svg` |
| PR curves | recall → precision | `outputs/figures/wider_pr.svg` |
| Train curve | loss vs iter | `outputs/figures/train_loss.svg` |

## Size gate decision

- Measured WIDER FACE root: **3.4538 GiB** (`outputs/logs/dataset_size.json`)
- Decision: **full_train** (< 5 GiB)
- Full train command: `CUDA_VISIBLE_DEVICES=0 python -u -m ssfd_plus.train --config configs/default.yaml`
- Log: `outputs/logs/train_full.log` (PID tracked at launch; ~1–3 s/iter ⇒ multi-day for 10 epochs × ~12.8k images)
- Smoke: passed (`outputs/logs/train_smoke.log`, `eval_smoke.log`)
