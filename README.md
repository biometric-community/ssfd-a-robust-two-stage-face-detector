# SSFD+ — A Robust Two-Stage Face Detector (TBIOM 2019)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

PyTorch rebuild of **SSFD+** (VGG16 + TransConv + CAM two-stage face detector) from:

`papers/ssfd-a-robust-two-stage-face-detector/`

**Repository:** [biometric-community/ssfd-a-robust-two-stage-face-detector](https://github.com/biometric-community/ssfd-a-robust-two-stage-face-detector)  
**Monorepo path:** `projects/papers/ssfd-a-robust-two-stage-face-detector` in [tbiom](https://github.com/biometric-community/tbiom) (git submodule).

## Setup

```bash
bash scripts/setup_env.sh
```

Uses the monorepo shared `.venv` (no project-local venv).

## Data

Requires real **WIDER FACE** under:

`projects/datasets/wider-face/extracted/{WIDER_train,WIDER_val,wider_face_split}`

## Train / eval / predict / report

```bash
bash scripts/check_dataset_size.sh ../../datasets/wider-face
bash scripts/train_full.sh
bash scripts/eval.sh
bash scripts/predict.sh --input /path/to/image.jpg
bash scripts/report.sh
```

Smoke (not for REPORT claims):

```bash
bash scripts/train.sh configs/smoke.yaml
bash scripts/eval.sh configs/smoke.yaml
```

## Layout

- `ssfd_plus/models/` — VGG16 backbone, CAM, two-stage SSFD+
- `ssfd_plus/data/` — WIDER loader + aspect-ratio resize
- `ssfd_plus/losses/` — RPN + ROI losses
- `configs/default.yaml` — paper protocol (10 epochs)
- `REPORT.md` — generated from real logs
- `DEVIATIONS.md`, `FIDELITY_AUDIT.md`, `SOURCE_CODE.md`

## Citation

Shi, Xu, Kakadiaris. SSFD+: A Robust Two-Stage Face Detector. *IEEE TBIOM*, 2019. DOI: [10.1109/TBIOM.2019.2928118](https://doi.org/10.1109/TBIOM.2019.2928118)
