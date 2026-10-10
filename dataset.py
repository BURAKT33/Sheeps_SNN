"""YOLO (bbox or segmentation polygon) dataset with on-the-fly spike encoding."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset

from config import CFG, YOLO_NAME_TO_MODEL, Config
from encoder import SpikeEncoder


def polygon_to_yolo_bbox(coords: Sequence[float]) -> Tuple[float, float, float, float]:
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = max(x_max - x_min, 1e-6)
    h = max(y_max - y_min, 1e-6)
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    return (
        float(np.clip(cx, 0.0, 1.0)),
        float(np.clip(cy, 0.0, 1.0)),
        float(np.clip(w, 0.0, 1.0)),
        float(np.clip(h, 0.0, 1.0)),
    )


def parse_yolo_label_line(parts: List[str]) -> Tuple[int, Tuple[float, float, float, float]]:
    cls_id = int(float(parts[0]))
    nums = [float(x) for x in parts[1:]]
    if len(nums) == 4:
        cx, cy, w, h = nums
        return cls_id, (
            float(np.clip(cx, 0.0, 1.0)),
            float(np.clip(cy, 0.0, 1.0)),
            float(np.clip(w, 0.0, 1.0)),
            float(np.clip(h, 0.0, 1.0)),
        )
    if len(nums) >= 6 and len(nums) % 2 == 0:
        return cls_id, polygon_to_yolo_bbox(nums)
    raise ValueError(f"Unsupported YOLO label format with {len(nums)} values")


def load_yaml_class_map(data_yaml: Path) -> Dict[int, int]:
    if not data_yaml.is_file():
        return {0: 0, 1: 1, 2: 2}

    with data_yaml.open("r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    names = meta.get("names", [])
    if isinstance(names, dict):
        ordered = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    else:
        ordered = list(names)

    mapping: Dict[int, int] = {}
    for src_idx, name in enumerate(ordered):
        key = str(name).strip().lower()
        if key in YOLO_NAME_TO_MODEL:
            mapping[src_idx] = YOLO_NAME_TO_MODEL[key]
    return mapping


def load_all_targets(
    label_path: Path, class_map: Dict[int, int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return class_ids (N,) and boxes (N,4)."""
    classes: List[int] = []
    boxes: List[Tuple[float, float, float, float]] = []
    text = label_path.read_text(encoding="utf-8").strip()
    if text:
        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                src_cls, box = parse_yolo_label_line(parts)
            except ValueError:
                continue
            if src_cls not in class_map:
                continue
            classes.append(class_map[src_cls])
            boxes.append(box)
    if not classes:
        return (
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, 4, dtype=torch.float32),
        )
    return (
        torch.tensor(classes, dtype=torch.long),
        torch.tensor(boxes, dtype=torch.float32),
    )


class SheepYoloSpikeDataset(Dataset):
    """YOLO labels -> spikes + all instance targets (multi-sheep)."""

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        cfg: Config = CFG,
        class_map: Optional[Dict[int, int]] = None,
        augment: bool = False,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.cfg = cfg
        self.augment = augment
        self.class_map = class_map or load_yaml_class_map(cfg.data_root / "data.yaml")

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.samples: List[Tuple[Path, Path]] = []
        for img_path in sorted(self.images_dir.iterdir()):
            if img_path.suffix.lower() not in exts:
                continue
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            if not label_path.is_file():
                continue
            self.samples.append((img_path, label_path))

        if not self.samples:
            raise FileNotFoundError(
                f"No image/label pairs under {self.images_dir} and {self.labels_dir}"
            )

        self.encoder = SpikeEncoder(
            time_steps=cfg.time_steps,
            image_size=cfg.image_size,
            use_frame_diff=cfg.use_frame_diff,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        img_path, label_path = self.samples[index]
        with Image.open(img_path) as im:
            rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0

        class_ids, boxes = load_all_targets(label_path, self.class_map)

        if self.augment and np.random.rand() < 0.5:
            rgb = np.ascontiguousarray(rgb[:, ::-1, :])
            if boxes.numel() > 0:
                boxes = boxes.clone()
                boxes[:, 0] = 1.0 - boxes[:, 0]

        frames = self.encoder.synthesize_sequence(
            rgb, motion_scale=0.04 if self.augment else 0.02
        )
        spikes = self.encoder.encode_sequence(frames)

        # Primary (largest) kept for simple metrics.
        if boxes.numel() > 0:
            areas = boxes[:, 2] * boxes[:, 3]
            primary = int(areas.argmax().item())
            primary_cls = class_ids[primary]
            primary_box = boxes[primary]
        else:
            primary_cls = torch.tensor(1, dtype=torch.long)
            primary_box = torch.tensor([0.5, 0.5, 0.2, 0.2], dtype=torch.float32)

        return {
            "spikes": spikes,
            "class_ids": class_ids,
            "boxes": boxes,
            "class_id": primary_cls,
            "bbox": primary_box,
            "path": str(img_path),
        }


def collate_spikes(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    spikes = torch.stack([b["spikes"] for b in batch], dim=1)
    class_ids = torch.stack([b["class_id"] for b in batch], dim=0)
    bboxes = torch.stack([b["bbox"] for b in batch], dim=0)

    max_n = max(int(b["boxes"].shape[0]) for b in batch)
    max_n = max(max_n, 1)
    B = len(batch)
    boxes_pad = torch.zeros(B, max_n, 4, dtype=torch.float32)
    labels_pad = torch.zeros(B, max_n, dtype=torch.long)
    tgt_mask = torch.zeros(B, max_n, dtype=torch.bool)
    for i, b in enumerate(batch):
        n = int(b["boxes"].shape[0])
        if n == 0:
            continue
        boxes_pad[i, :n] = b["boxes"]
        labels_pad[i, :n] = b["class_ids"]
        tgt_mask[i, :n] = True

    return {
        "spikes": spikes,
        "class_id": class_ids,
        "bbox": bboxes,
        "tgt_boxes": boxes_pad,
        "tgt_labels": labels_pad,
        "tgt_mask": tgt_mask,
    }
