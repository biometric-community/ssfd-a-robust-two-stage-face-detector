# SSFD+ Smoke Report

Rebuilt from TBIOM 2019 SSFD+ (Shi, Xu, Kakadiaris). Metrics below are from **our** runs on real WIDER FACE data — not fabricated.

## Setup

- CAM enabled: `True`
- Resize: min side 480, max long 800
- Train batch / epochs: 1 / 1

## Results summary

### Training

- Final iter: 16
- Final loss: 1.477062702178955
- Train images: 16
- Wall time (s): 21.44818925857544

### WIDER FACE val mAP (Ours)

| Subset | mAP |
|--------|-----|
| easy | 0.0000 |
| medium | 0.0000 |
| hard | 0.0000 |
| all | 0.0000 |

Paper-reported SSFD+ WIDER **val** (reference only): easy 91.3 / medium 90.3 / hard 83.1.

## Figures

### `outputs/figures/train_loss.svg`

![](outputs/figures/train_loss.svg)

### `outputs/figures/wider_map_bars.svg`

![](outputs/figures/wider_map_bars.svg)

### `outputs/figures/wider_pr.svg`

![](outputs/figures/wider_pr.svg)

## Regenerate

```bash
bash scripts/report.sh
```
