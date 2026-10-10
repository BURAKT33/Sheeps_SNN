"""Train multi-sheep SNN detector; save best.pt."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CFG, Config
from dataset import SheepYoloSpikeDataset, collate_spikes
from models.snn_detector import SNNDetector, build_model, decode_detections, reset_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def draw_gaussian_heatmap(
    boxes: torch.Tensor,
    mask: torch.Tensor,
    hs: int,
    ws: int,
    sigma: float = 1.5,
) -> torch.Tensor:
    """boxes (N,4) cxcywh normalized -> heatmap (1,Hs,Ws)."""
    heat = torch.zeros(1, hs, ws, device=boxes.device, dtype=boxes.dtype)
    ys = torch.arange(hs, device=boxes.device, dtype=boxes.dtype).view(hs, 1)
    xs = torch.arange(ws, device=boxes.device, dtype=boxes.dtype).view(1, ws)
    for i in range(boxes.shape[0]):
        if not bool(mask[i]):
            continue
        cx, cy, _, _ = boxes[i]
        x0 = cx * (ws - 1)
        y0 = cy * (hs - 1)
        g = torch.exp(-((xs - x0) ** 2 + (ys - y0) ** 2) / (2 * sigma**2))
        heat[0] = torch.maximum(heat[0], g)
    return heat


def detection_loss(
    heatmap: torch.Tensor,
    class_map: torch.Tensor,
    size_map: torch.Tensor,
    tgt_boxes: torch.Tensor,
    tgt_labels: torch.Tensor,
    tgt_mask: torch.Tensor,
    alpha: float,
    beta: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    bsz, _, hs, ws = heatmap.shape
    device = heatmap.device

    heat_gt = torch.zeros(bsz, 1, hs, ws, device=device)
    for b in range(bsz):
        heat_gt[b] = draw_gaussian_heatmap(tgt_boxes[b], tgt_mask[b], hs, ws)

    # Focal-style BCE on heatmap
    pred = torch.sigmoid(heatmap).clamp(1e-4, 1 - 1e-4)
    pos = heat_gt.eq(1)
    # soft positives from gaussian > 0.5
    pos = heat_gt >= 0.5
    neg = ~pos
    pos_loss = -torch.log(pred) * torch.pow(1 - pred, 2) * pos
    neg_loss = -torch.log(1 - pred) * torch.pow(pred, 2) * neg * torch.pow(1 - heat_gt, 4)
    num_pos = pos.sum().clamp(min=1.0)
    loss_heat = (pos_loss.sum() + neg_loss.sum()) / num_pos

    loss_cls = torch.tensor(0.0, device=device)
    loss_wh = torch.tensor(0.0, device=device)
    n_obj = 0
    for b in range(bsz):
        for i in range(tgt_boxes.shape[1]):
            if not bool(tgt_mask[b, i]):
                continue
            cx, cy, bw, bh = tgt_boxes[b, i]
            x = int(torch.round(cx * (ws - 1)).item())
            y = int(torch.round(cy * (hs - 1)).item())
            x = min(max(x, 0), ws - 1)
            y = min(max(y, 0), hs - 1)
            loss_cls = loss_cls + F.cross_entropy(
                class_map[b : b + 1, :, y, x], tgt_labels[b : b + 1, i]
            )
            loss_wh = loss_wh + F.smooth_l1_loss(
                size_map[b, :, y, x], tgt_boxes[b, i, 2:]
            )
            n_obj += 1
    if n_obj > 0:
        loss_cls = loss_cls / n_obj
        loss_wh = loss_wh / n_obj

    total = alpha * (loss_heat + loss_cls) + beta * loss_wh
    return total, {
        "loss": float(total.detach().item()),
        "loss_heat": float(loss_heat.detach().item()),
        "loss_cls": float(loss_cls.detach().item()),
        "loss_bbox": float(loss_wh.detach().item()),
    }


@torch.no_grad()
def evaluate(model: SNNDetector, loader: DataLoader, cfg: Config, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for batch in loader:
        spikes = batch["spikes"].to(device)
        reset_model(model)
        out = model(spikes)
        loss, parts = detection_loss(
            out["heatmap"],
            out["class_map"],
            out["size_map"],
            batch["tgt_boxes"].to(device),
            batch["tgt_labels"].to(device),
            batch["tgt_mask"].to(device),
            cfg.alpha_cls,
            cfg.beta_bbox,
        )
        dets = decode_detections(
            out["heatmap"],
            out["class_map"],
            out["size_map"],
            conf_threshold=0.2,
            max_det=1,
            nms_kernel=5,
        )
        pred = dets["labels"][:, 0]
        gt = batch["class_id"].to(device)
        acc = (pred == gt).float().mean().item()
        bs = gt.shape[0]
        total_loss += parts["loss"] * bs
        total_acc += acc * bs
        n += bs
        reset_model(model)
    if n == 0:
        return {"loss": 0.0, "acc": 0.0}
    return {"loss": total_loss / n, "acc": total_acc / n}


def train(cfg: Config) -> Path:
    set_seed(cfg.seed)
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device if cfg.device != "cuda" or torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = SheepYoloSpikeDataset(cfg.train_images, cfg.train_labels, cfg=cfg, augment=True)
    valid_ds = SheepYoloSpikeDataset(cfg.valid_images, cfg.valid_labels, cfg=cfg, augment=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_spikes,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_spikes,
    )

    model = build_model(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        base_channels=cfg.base_channels,
        time_steps=cfg.time_steps,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    best_acc = -1.0
    best_path = cfg.best_checkpoint

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = {"loss": 0.0, "loss_heat": 0.0, "loss_cls": 0.0, "loss_bbox": 0.0}
        seen = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}")
        for batch in pbar:
            spikes = batch["spikes"].to(device)
            reset_model(model)
            out = model(spikes)
            loss, parts = detection_loss(
                out["heatmap"],
                out["class_map"],
                out["size_map"],
                batch["tgt_boxes"].to(device),
                batch["tgt_labels"].to(device),
                batch["tgt_mask"].to(device),
                cfg.alpha_cls,
                cfg.beta_bbox,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            reset_model(model)

            bs = spikes.shape[1]
            seen += bs
            for k in running:
                running[k] += parts[k] * bs
            pbar.set_postfix(
                loss=f"{running['loss']/seen:.3f}",
                heat=f"{running['loss_heat']/seen:.3f}",
                cls=f"{running['loss_cls']/seen:.3f}",
            )

        val = evaluate(model, valid_loader, cfg, device)
        print(f"Val  acc@top1={val['acc']:.3f}  loss={val['loss']:.4f}")

        ckpt = {
            "model": model.state_dict(),
            "cfg": {
                "time_steps": cfg.time_steps,
                "image_size": cfg.image_size,
                "in_channels": cfg.in_channels,
                "num_classes": cfg.num_classes,
                "class_names": cfg.class_names,
                "base_channels": cfg.base_channels,
                "use_frame_diff": cfg.use_frame_diff,
                "multi_object": True,
            },
            "epoch": epoch,
            "val_acc": val["acc"],
        }
        torch.save(ckpt, cfg.checkpoint_dir / "last.pt")
        if val["acc"] >= best_acc:
            best_acc = val["acc"]
            torch.save(ckpt, best_path)
            print(f"Saved best checkpoint -> {best_path} (acc={best_acc:.3f})")

    return best_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train multi-sheep SNN detector")
    p.add_argument("--epochs", type=int, default=CFG.epochs)
    p.add_argument("--batch-size", type=int, default=CFG.batch_size)
    p.add_argument("--lr", type=float, default=CFG.learning_rate)
    p.add_argument("--device", type=str, default=CFG.device)
    p.add_argument("--image-size", type=int, default=CFG.image_size)
    p.add_argument("--time-steps", type=int, default=CFG.time_steps)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    CFG.epochs = args.epochs
    CFG.batch_size = args.batch_size
    CFG.learning_rate = args.lr
    CFG.device = args.device
    CFG.image_size = args.image_size
    CFG.time_steps = args.time_steps
    CFG.num_workers = args.num_workers
    path = train(CFG)
    print(f"Done. Best weights: {path}")
