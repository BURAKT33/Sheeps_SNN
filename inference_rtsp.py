"""Live RTSP / webcam / video-file inference with multi-sheep SNN detector."""

from __future__ import annotations

import argparse
import csv
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np
import torch

from camera import LatestFrameStream
from config import CFG, Config
from encoder import SpikeEncoder
from models.snn_detector import (
    CLASS_COLORS_RGB,
    SNNDetector,
    build_model,
    decode_detections,
    reset_model,
)

_VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".m4v", ".webm"}


def _is_file_source(source: str) -> bool:
    p = Path(source)
    return p.is_file() and p.suffix.lower() in _VIDEO_EXTS


def load_checkpoint(path: Path, device: torch.device) -> Tuple[SNNDetector, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = ckpt.get("cfg", {})
    model = build_model(
        in_channels=int(meta.get("in_channels", CFG.in_channels)),
        num_classes=int(meta.get("num_classes", CFG.num_classes)),
        base_channels=int(meta.get("base_channels", CFG.base_channels)),
        time_steps=int(meta.get("time_steps", CFG.time_steps)),
    )
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print("Warning: checkpoint missing keys (retrain recommended):", missing[:8])
    model.to(device)
    model.eval()
    return model, meta


def yolo_to_xyxy(
    cx: float, cy: float, w: float, h: float, width: int, height: int
) -> Tuple[int, int, int, int]:
    x1 = int((cx - w / 2.0) * width)
    y1 = int((cy - h / 2.0) * height)
    x2 = int((cx + w / 2.0) * width)
    y2 = int((cy + h / 2.0) * height)
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    return x1, y1, x2, y2


@torch.no_grad()
def run_inference(cfg: Config, source: str, checkpoint: Path) -> None:
    device = torch.device(
        cfg.device if cfg.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    model, meta = load_checkpoint(checkpoint, device)
    class_names = list(meta.get("class_names", cfg.class_names))
    image_size = int(meta.get("image_size", cfg.image_size))
    time_steps = int(meta.get("time_steps", cfg.time_steps))
    use_diff = bool(meta.get("use_frame_diff", cfg.use_frame_diff))

    encoder = SpikeEncoder(
        time_steps=time_steps,
        image_size=image_size,
        use_frame_diff=use_diff,
    )

    cfg.timeseries_dir.mkdir(parents=True, exist_ok=True)
    ts_path = cfg.timeseries_dir / "live_timeseries.csv"
    ts_file = ts_path.open("w", newline="", encoding="utf-8")
    ts_writer = csv.writer(ts_file)
    ts_writer.writerow(["t_sec", "det_id", "state", "cx", "cy", "w", "h", "conf"])

    frame_buf: Deque[np.ndarray] = deque(maxlen=time_steps)
    infer_counter = 0
    t0 = time.time()
    last_dets: Optional[dict] = None

    use_latest_thread = not _is_file_source(source)
    stream: Optional[LatestFrameStream] = None
    file_cap: Optional[cv2.VideoCapture] = None

    if use_latest_thread:
        stream = LatestFrameStream(source, reconnect_sec=cfg.reconnect_sec).start()
        print(f"Latest-frame RTSP/webcam stream: {source}")
    else:
        file_cap = cv2.VideoCapture(source)
        if not file_cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {source}")
        print(f"Sequential file playback: {source}")
    print(f"Temporal buffer size T={time_steps} | multi-sheep decode")

    def read_frame() -> Tuple[bool, Optional[np.ndarray]]:
        if stream is not None:
            return stream.read()
        assert file_cap is not None
        return file_cap.read()

    try:
        while True:
            ok, frame = read_frame()
            if not ok or frame is None:
                if file_cap is not None:
                    break
                time.sleep(0.01)
                continue

            display = frame.copy()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_buf.append(rgb)

            infer_counter += 1
            do_infer = len(frame_buf) >= time_steps and (
                infer_counter % max(cfg.frame_stride, 1) == 0
            )

            if do_infer:
                spikes = encoder.encode_sequence(list(frame_buf)).unsqueeze(1).to(device)
                reset_model(model)
                out = model(spikes)
                reset_model(model)
                if "heatmap" not in out:
                    raise RuntimeError("Old single-object checkpoint. Retrain required.")
                last_dets = decode_detections(
                    out["heatmap"],
                    out["class_map"],
                    out["size_map"],
                    conf_threshold=cfg.conf_threshold,
                    max_det=30,
                    nms_kernel=5,
                )
                t_sec = time.time() - t0
                n = int(last_dets["mask"][0].sum().item())
                for i in range(n):
                    cls_i = int(last_dets["labels"][0, i].item())
                    state = class_names[cls_i] if cls_i < len(class_names) else str(cls_i)
                    sc = float(last_dets["scores"][0, i].item())
                    cx, cy, bw, bh = map(float, last_dets["boxes"][0, i].tolist())
                    ts_writer.writerow(
                        [f"{t_sec:.3f}", i, state, f"{cx:.4f}", f"{cy:.4f}", f"{bw:.4f}", f"{bh:.4f}", f"{sc:.4f}"]
                    )
                ts_file.flush()

            if last_dets is not None:
                h, w = display.shape[:2]
                n = int(last_dets["mask"][0].sum().item())
                for i in range(n):
                    cls_i = int(last_dets["labels"][0, i].item())
                    state = class_names[cls_i] if cls_i < len(class_names) else str(cls_i)
                    sc = float(last_dets["scores"][0, i].item())
                    cx, cy, bw, bh = map(float, last_dets["boxes"][0, i].tolist())
                    x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, bw, bh, w, h)
                    rgb_c = CLASS_COLORS_RGB.get(state, (0, 0, 255))
                    # PIL RGB -> OpenCV BGR
                    color = (rgb_c[2], rgb_c[1], rgb_c[0])
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    label = f"#{i+1} {state.upper()} {sc:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    cv2.rectangle(display, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(
                        display,
                        label,
                        (x1 + 2, max(th + 2, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 0),
                        2,
                    )

            if cfg.show_window:
                cv2.imshow("SNN Sheep Multi", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        if stream is not None:
            stream.stop()
        if file_cap is not None:
            file_cap.release()
        ts_file.close()
        if cfg.show_window:
            cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SNN RTSP / video multi-sheep inference")
    p.add_argument("--source", type=str, default=CFG.rtsp_url)
    p.add_argument("--checkpoint", type=str, default=str(CFG.best_checkpoint))
    p.add_argument("--device", type=str, default=CFG.device)
    p.add_argument("--stride", type=int, default=CFG.frame_stride)
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    CFG.device = args.device
    CFG.frame_stride = args.stride
    CFG.show_window = not args.no_show
    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt}. Train first with: python train.py")
    run_inference(CFG, args.source, ckpt)
