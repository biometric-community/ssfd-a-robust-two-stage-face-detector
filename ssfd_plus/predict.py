"""Run SSFD+ inference on images and save detections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw

from ssfd_plus.data.transforms import IMAGENET_MEAN, IMAGENET_STD, _resize_keep_aspect
from ssfd_plus.engine.config import load_config
from ssfd_plus.models import SSFDPlus


@torch.no_grad()
def predict_image(
    model: SSFDPlus,
    path: Path,
    min_side: int,
    max_long: int,
    device,
    score_thresh: float,
    nms_thresh: float,
):
    img = Image.open(path).convert("RGB")
    orig_w, orig_h = img.size
    img_r, scale = _resize_keep_aspect(img, min_side, max_long)
    tensor = TF.normalize(TF.to_tensor(img_r), IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0).to(device)
    pred = model.predict(tensor, score_thresh=score_thresh, nms_thresh=nms_thresh)[0]
    boxes = pred["boxes"].cpu()
    scores = pred["scores"].cpu()
    boxes = boxes / scale
    boxes[:, [0, 2]].clamp_(0, orig_w - 1)
    boxes[:, [1, 3]].clamp_(0, orig_h - 1)
    return img, boxes, scores


def main() -> None:
    ap = argparse.ArgumentParser(description="SSFD+ predict")
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--input", type=str, required=True, help="Image file or directory")
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(project_root / args.config)
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    model = SSFDPlus(pretrained_backbone=False, use_cam=cfg["model"].get("use_cam", True)).to(device)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else project_root / cfg["paths"]["checkpoint_dir"] / "latest.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    inp = Path(args.input)
    paths = sorted(inp.glob("*")) if inp.is_dir() else [inp]
    paths = [p for p in paths if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    if not paths:
        raise FileNotFoundError(f"No images under {inp}")
    out_dir = Path(args.out_dir) if args.out_dir else project_root / cfg["paths"]["pred_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    min_side = cfg["model"]["min_side"]
    max_long = cfg["model"]["max_long"]
    score_thresh = cfg["eval"].get("score_thresh", 0.05)
    nms_thresh = cfg["eval"].get("nms_thresh", 0.5)

    manifest = []
    for p in paths:
        img, boxes, scores = predict_image(model, p, min_side, max_long, device, score_thresh, nms_thresh)
        draw = ImageDraw.Draw(img)
        dets = []
        for box, sc in zip(boxes.tolist(), scores.tolist()):
            draw.rectangle(box, outline=(0, 51, 204), width=3)
            dets.append({"box": box, "score": sc})
        out_img = out_dir / f"{p.stem}_det.jpg"
        img.save(out_img)
        out_json = out_dir / f"{p.stem}_det.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"image": str(p), "detections": dets}, f, indent=2)
        manifest.append({"image": str(p), "out_image": str(out_img), "n": len(dets)})
        print(f"{p.name}: {len(dets)} faces -> {out_img}")
    with open(out_dir / "predict_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
