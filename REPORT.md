# SSFD+ Face Detection Report

Rebuilt from TBIOM 2019 SSFD+ (Shi, Xu, Kakadiaris). Metrics below are from **our** runs on real WIDER FACE data — not fabricated.

## Setup

- CAM enabled: `True`
- Resize: min side 800, max long 1200 (paper 1200/1600 — see D8)
- Train batch / epochs: 1 / 10
- Size gate: WIDER FACE **3.45 GiB** → full train required

## Results summary

### Training

**In progress** — full 10-epoch train on WIDER FACE (`outputs/logs/train_full.log`).

Smoke train (16 images) completed earlier; those numbers are not full-protocol results.

Paper-reported SSFD+ WIDER **val** (reference only): easy 91.3 / medium 90.3 / hard 83.1.

### WIDER FACE val mAP

BLOCKED until full train finishes and `scripts/eval.sh` is run on the trained checkpoint.

## Regenerate

```bash
bash scripts/report.sh
```
