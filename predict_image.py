"""Quick image test: multi-sheep boxes with clear class labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from config import CFG, MODEL_CLASS_NAMES
from dataset import load_all_targets, load_yaml_class_map
from encoder import SpikeEncoder
from inference_rtsp import load_checkpoint, yolo_to_xyxy
from models.snn_detector import CLASS_COLORS_RGB, decode_detections, reset_model


def main() -> None:
    p = argparse.ArgumentParser(description="Test multi-sheep SNN .pt on one image")
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default=str(CFG.best_checkpoint))
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--max-det", type=int, default=30)
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    image_path = Path(args.image)
    ckpt_path = Path(args.checkpoint)
    if not image_path.is_file():
        raise SystemExit(f"Image not found: {image_path}")
    if not ckpt_path.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    model, meta = load_checkpoint(ckpt_path, device)
    class_names = list(meta.get("class_names", MODEL_CLASS_NAMES))
    encoder = SpikeEncoder(
        time_steps=int(meta.get("time_steps", CFG.time_steps)),
        image_size=int(meta.get("image_size", CFG.image_size)),
        use_frame_diff=bool(meta.get("use_frame_diff", CFG.use_frame_diff)),
    )

    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    frames = encoder.synthesize_sequence(rgb, motion_scale=0.02)
    spikes = encoder.encode_sequence(frames).unsqueeze(1).to(device)

    reset_model(model)
    out = model(spikes)
    reset_model(model)

    if "heatmap" not in out:
        raise SystemExit(
            "This checkpoint is single-object (old). Retrain with: python3 train.py --epochs 30 --device cpu"
        )

    dets = decode_detections(
        out["heatmap"],
        out["class_map"],
        out["size_map"],
        conf_threshold=args.conf,
        max_det=args.max_det,
        nms_kernel=5,
    )

    print(f"Image      : {image_path}")
    print(f"Checkpoint : {ckpt_path}")
    n = int(dets["mask"][0].sum().item())
    print(f"Detections : {n}")
    for i in range(n):
        cls_i = int(dets["labels"][0, i].item())
        name = class_names[cls_i] if cls_i < len(class_names) else str(cls_i)
        sc = float(dets["scores"][0, i].item())
        box = dets["boxes"][0, i].cpu().numpy()
        print(f"  #{i+1}  CLASS={name.upper():5s}  conf={sc:.3f}  box={np.round(box,3).tolist()}")

    label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    class_map = load_yaml_class_map(CFG.data_root / "data.yaml")
    gt_cls, gt_boxes = load_all_targets(label_path, class_map)
    print(f"GT count   : {int(gt_boxes.shape[0])}")

    im = Image.open(image_path).convert("RGB")
    w, h = im.size
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    # GT in thin green
    for i in range(gt_boxes.shape[0]):
        box = gt_boxes[i].numpy()
        x1, y1, x2, y2 = yolo_to_xyxy(*box, w, h)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 180, 0), width=2)
        gname = class_names[int(gt_cls[i])]
        draw.text((x1, max(0, y1 - 16)), f"GT:{gname}", fill=(0, 180, 0), font=font)

    # Predictions with class color + big label
    for i in range(n):
        cls_i = int(dets["labels"][0, i].item())
        name = class_names[cls_i] if cls_i < len(class_names) else str(cls_i)
        sc = float(dets["scores"][0, i].item())
        box = dets["boxes"][0, i].cpu().numpy()
        x1, y1, x2, y2 = yolo_to_xyxy(*box, w, h)
        color = CLASS_COLORS_RGB.get(name, (255, 0, 0))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"#{i+1} {name.upper()} {sc:.2f}"
        # label background
        tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        ty = max(0, y1 - th - 2)
        draw.rectangle([x1, ty, x1 + tw + 4, ty + th + 2], fill=color)
        draw.text((x1 + 2, ty), label, fill=(0, 0, 0), font=font)

    if args.out:
        out_path = Path(args.out)
    else:
        out_dir = Path("outputs/test_vis")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"pred_{image_path.stem}.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)
    print(f"Saved      : {out_path.resolve()}")


if __name__ == "__main__":
    main()
