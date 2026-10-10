"""Global hyperparameters for SNN sheep detection training and RTSP inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints"
TIMESERIES_DIR = ROOT / "outputs" / "timeseries"


# Source Roboflow names -> model class ids
# data.yaml: ['drinking', 'eating', 'lying', 'stand']
YOLO_NAME_TO_MODEL: Dict[str, int] = {
    "lying": 0,  # sit
    "stand": 1,
    "eating": 2,  # eat
    "drinking": 2,  # map drink -> eat for 3-class head
}

MODEL_CLASS_NAMES: List[str] = ["sit", "stand", "eat"]


@dataclass
class Config:
    # Paths
    data_root: Path = DATA_ROOT
    checkpoint_dir: Path = CHECKPOINT_DIR
    timeseries_dir: Path = TIMESERIES_DIR

    # Spike / image
    time_steps: int = 4
    image_size: int = 160
    in_channels: int = 2  # intensity + frame-diff channel
    use_frame_diff: bool = True

    # Model
    num_classes: int = 3
    class_names: List[str] = field(default_factory=lambda: list(MODEL_CLASS_NAMES))
    base_channels: int = 32

    # Training
    epochs: int = 30
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 2
    alpha_cls: float = 1.0
    beta_bbox: float = 5.0  # stronger localization (was center-biased)
    device: str = "cpu"  # set to "cuda" when GPU available
    seed: int = 42

    # Inference / RTSP
    rtsp_url: str = "rtsp://user:pass@192.168.1.10:554/stream1"
    frame_stride: int = 10
    conf_threshold: float = 0.35
    reconnect_sec: float = 3.0
    show_window: bool = True

    @property
    def train_images(self) -> Path:
        return self.data_root / "train" / "images"

    @property
    def train_labels(self) -> Path:
        return self.data_root / "train" / "labels"

    @property
    def valid_images(self) -> Path:
        return self.data_root / "valid" / "images"

    @property
    def valid_labels(self) -> Path:
        return self.data_root / "valid" / "labels"

    @property
    def best_checkpoint(self) -> Path:
        return self.checkpoint_dir / "best.pt"

    def input_shape(self) -> Tuple[int, int, int, int]:
        """(T, C, H, W) without batch."""
        return (self.time_steps, self.in_channels, self.image_size, self.image_size)


CFG = Config()
