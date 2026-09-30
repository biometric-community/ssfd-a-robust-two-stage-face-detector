# Deviations from the paper

| ID | Description | Severity | Mitigation |
|----|-------------|----------|------------|
| D1 | FDDB, PASCAL Faces, AFW not present under `projects/datasets/` | eval coverage | WIDER FACE only |
| D2 | No official code; PyTorch reimplementation from PDF | medium | Cite sections/figures; fidelity audit |
| D3 | ROI head uses standard RoIAlign + FC rather than exact Caffe layer names | low | Functionally equivalent Faster R-CNN stage-2 |
| D4 | WIDER easy/medium/hard split approximated by face height on eval canvas | medium | Prefer official WIDER eval toolkit when available |
| D5 | Stage-2 proposals capped at 256 top RPN-scored boxes during training | low | Avoids RoIAlign on thousands of anchors per step |
| D6 | Single-scale test only (no image pyramid) | low | Matches SSFD+ single-scale design |
| D7 | Batch size 1 due to variable aspect-ratio resizing per image | low | Paper does not always specify batch; memory-safe |
| D8 | Default resize **800 / 1200** instead of paper **1200 / 1600** for GPU memory | medium | Configurable via `model.min_side` / `max_long` |

Intentional non-goals: matching published mAP numbers exactly (structural fidelity is the default skill target).
