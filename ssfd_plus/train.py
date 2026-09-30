"""Train SSFD+ on WIDER FACE (Sec. 4.3)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ssfd_plus.data import WiderFaceDataset, WiderTrainTransform, resolve_wider_paths, wider_collate
from ssfd_plus.engine.config import load_config, resolve_data_root
from ssfd_plus.losses import SSFDPlusLoss
from ssfd_plus.models import SSFDPlus


def train(cfg: dict, project_root: Path) -> None:
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    paths = resolve_wider_paths(resolve_data_root(cfg, project_root))
    ds = WiderFaceDataset(
        paths["train_images"],
        paths["train_gt"],
        transforms=WiderTrainTransform(cfg["model"]["min_side"], cfg["model"]["max_long"]),
        max_samples=cfg["data"].get("max_train_samples"),
    )
    batch_size = int(cfg["train"]["batch_size"])
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg["train"].get("num_workers", 4),
        collate_fn=wider_collate,
        pin_memory=True,
    )
    model = SSFDPlus(
        pretrained_backbone=cfg["model"].get("pretrained_backbone", True),
        use_cam=cfg["model"].get("use_cam", True),
    ).to(device)
    criterion = SSFDPlusLoss()
    optim = torch.optim.SGD(
        model.parameters(),
        lr=cfg["train"]["lr"],
        momentum=cfg["train"].get("momentum", 0.9),
        weight_decay=cfg["train"].get("weight_decay", 5e-4),
    )
    epochs = int(cfg["train"]["epochs"])
    lr_decay_epoch = int(cfg["train"].get("lr_decay_epoch", 7))
    log_every = int(cfg["train"].get("log_every", 50))
    ckpt_dir = project_root / cfg["paths"]["checkpoint_dir"]
    log_dir = project_root / cfg["paths"]["log_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    use_amp = bool(cfg["train"].get("use_amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history = []
    it = 0
    t0 = time.time()
    print(f"Train device={device} epochs={epochs} batch={batch_size} amp={use_amp}", flush=True)

    for epoch in range(epochs):
        if epoch == lr_decay_epoch:
            for g in optim.param_groups:
                g["lr"] *= cfg["train"].get("lr_gamma", 0.1)
        model.train()
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            targets = batch["targets"]
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(images)
                losses = criterion(model, out, targets)
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optim)
            scaler.update()
            it += 1
            if it % log_every == 0 or it == 1:
                row = {
                    "iter": it,
                    "epoch": epoch + 1,
                    "loss": float(losses["loss"].detach()),
                    "loss_rpn_cls": float(losses["loss_rpn_cls"].detach()),
                    "loss_rpn_reg": float(losses["loss_rpn_reg"].detach()),
                    "loss_roi_cls": float(losses["loss_roi_cls"].detach()),
                    "loss_roi_reg": float(losses["loss_roi_reg"].detach()),
                    "lr": optim.param_groups[0]["lr"],
                    "time_s": time.time() - t0,
                }
                history.append(row)
                print(
                    f"epoch {epoch+1}/{epochs} iter {it} loss={row['loss']:.4f} lr={row['lr']:.6f}",
                    flush=True,
                )
                with open(log_dir / "train_history.json", "w", encoding="utf-8") as f:
                    json.dump(history, f, indent=2)
        ckpt = {"epoch": epoch + 1, "iter": it, "model": model.state_dict(), "optim": optim.state_dict(), "cfg": cfg}
        torch.save(ckpt, ckpt_dir / f"ssfd_epoch{epoch+1:02d}.pt")
        torch.save(ckpt, ckpt_dir / "latest.pt")

    summary = {
        "epochs": epochs,
        "final_iter": it,
        "final_loss": history[-1]["loss"] if history else None,
        "num_train_images": len(ds),
        "seconds": time.time() - t0,
        "use_cam": cfg["model"].get("use_cam", True),
    }
    with open(log_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print("Wrote", log_dir / "train_summary.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train SSFD+")
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    args = ap.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = project_root / cfg_path
    train(load_config(cfg_path), project_root)


if __name__ == "__main__":
    main()
