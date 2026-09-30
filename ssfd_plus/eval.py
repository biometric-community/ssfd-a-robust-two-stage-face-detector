"""Evaluate SSFD+ on WIDER FACE validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ssfd_plus.data import WiderEvalTransform, WiderFaceDataset, resolve_wider_paths, wider_collate
from ssfd_plus.engine.config import load_config, resolve_data_root
from ssfd_plus.metrics import average_precision, split_wider_difficulty
from ssfd_plus.models import SSFDPlus


@torch.no_grad()
def evaluate(cfg: dict, project_root: Path, checkpoint: str | None = None) -> dict:
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    paths = resolve_wider_paths(resolve_data_root(cfg, project_root))
    ds = WiderFaceDataset(
        paths["val_images"],
        paths["val_gt"],
        transforms=WiderEvalTransform(cfg["model"]["min_side"], cfg["model"]["max_long"]),
        max_samples=cfg["data"].get("max_val_samples"),
    )
    loader = DataLoader(
        ds,
        batch_size=cfg["eval"].get("batch_size", 1),
        shuffle=False,
        num_workers=cfg["train"].get("num_workers", 4),
        collate_fn=wider_collate,
    )
    model = SSFDPlus(
        pretrained_backbone=False,
        use_cam=cfg["model"].get("use_cam", True),
    ).to(device)
    ckpt_path = Path(checkpoint) if checkpoint else project_root / cfg["paths"]["checkpoint_dir"] / "latest.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Train first.")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    score_thresh = cfg["eval"].get("score_thresh", 0.05)
    nms_thresh = cfg["eval"].get("nms_thresh", 0.5)
    buckets = {k: {"pb": [], "ps": [], "gb": []} for k in ("easy", "medium", "hard", "all")}

    for batch in loader:
        images = batch["images"].to(device)
        preds = model.predict(images, score_thresh=score_thresh, nms_thresh=nms_thresh)
        for pred, tgt in zip(preds, batch["targets"]):
            pb = pred["boxes"].cpu().numpy()
            ps = pred["scores"].cpu().numpy()
            gb = tgt["boxes"].cpu().numpy().reshape(-1, 4)
            buckets["all"]["pb"].append(pb)
            buckets["all"]["ps"].append(ps)
            buckets["all"]["gb"].append(gb)
            split = split_wider_difficulty(gb)
            for name in ("easy", "medium", "hard"):
                buckets[name]["pb"].append(pb)
                buckets[name]["ps"].append(ps)
                buckets[name]["gb"].append(split[name])

    results = {}
    pr_curves = {}
    for name, b in buckets.items():
        ap = average_precision(b["pb"], b["ps"], b["gb"], iou_thresh=0.5)
        results[name] = ap["ap"]
        pr_curves[name] = {"recall": ap["recall"], "precision": ap["precision"], "n_gt": ap["n_gt"]}

    payload = {
        "checkpoint": str(ckpt_path),
        "num_images": len(ds),
        "map": results,
        "pr": pr_curves,
        "note": "Difficulty split by face height proxy (D4) unless WIDER attributes wired.",
    }
    log_dir = project_root / cfg["paths"]["log_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(json.dumps(results, indent=2))
    print("Wrote", out_path)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate SSFD+")
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--checkpoint", type=str, default=None)
    args = ap.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = project_root / cfg_path
    evaluate(load_config(cfg_path), project_root, args.checkpoint)


if __name__ == "__main__":
    main()
