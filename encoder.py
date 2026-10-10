"""Multi-frame spike encoding: real temporal sequence -> (T, C, H, W)."""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F


ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_float_gray(image: ArrayLike) -> torch.Tensor:
    """Convert BGR/RGB/gray image to gray float tensor in [0, 1], shape (H, W)."""
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 3:
            if arr.shape[2] == 3:
                arr = (
                    0.114 * arr[:, :, 0]
                    + 0.587 * arr[:, :, 1]
                    + 0.299 * arr[:, :, 2]
                )
            else:
                arr = arr[:, :, 0]
        tensor = torch.from_numpy(arr.astype(np.float32))
    else:
        tensor = image.float()
        if tensor.ndim == 3:
            if tensor.shape[0] in (1, 3):
                if tensor.shape[0] == 3:
                    tensor = 0.299 * tensor[0] + 0.587 * tensor[1] + 0.114 * tensor[2]
                else:
                    tensor = tensor[0]
            else:
                tensor = 0.299 * tensor[:, :, 0] + 0.587 * tensor[:, :, 1] + 0.114 * tensor[:, :, 2]

    if tensor.max() > 1.5:
        tensor = tensor / 255.0
    return tensor.clamp(0.0, 1.0)


def resize_gray(gray: torch.Tensor, size: int) -> torch.Tensor:
    x = gray.unsqueeze(0).unsqueeze(0)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return x.squeeze(0).squeeze(0)


class SpikeEncoder:
    """
    Encode a temporal window of frames into spikes.

    Each of the T frames becomes one time step (real video time), not a
    repeated rate-code of a single still image.
    Channels: intensity (+ optional frame-difference).
    """

    def __init__(
        self,
        time_steps: int = 4,
        image_size: int = 160,
        use_frame_diff: bool = True,
        intensity_gain: float = 1.0,
        diff_gain: float = 2.0,
        spike_threshold: float = 0.35,
    ) -> None:
        self.time_steps = time_steps
        self.image_size = image_size
        self.use_frame_diff = use_frame_diff
        self.intensity_gain = intensity_gain
        self.diff_gain = diff_gain
        self.spike_threshold = spike_threshold

    def _features(
        self,
        gray: torch.Tensor,
        prev_gray: Optional[torch.Tensor],
    ) -> torch.Tensor:
        intensity = (gray * self.intensity_gain).clamp(0.0, 1.0)
        if not self.use_frame_diff:
            return intensity.unsqueeze(0)
        if prev_gray is None:
            diff = torch.zeros_like(intensity)
        else:
            diff = ((gray - prev_gray).abs() * self.diff_gain).clamp(0.0, 1.0)
        return torch.stack([intensity, diff], dim=0)

    def encode_frame(
        self,
        image: ArrayLike,
        prev_gray: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode one video frame -> spikes (C,H,W) and resized gray."""
        gray = resize_gray(_to_float_gray(image), self.image_size)
        feats = self._features(gray, prev_gray)
        spikes = (feats >= self.spike_threshold).float()
        return spikes, gray

    def encode_sequence(self, frames: Sequence[ArrayLike]) -> torch.Tensor:
        """
        Encode frames as temporal spikes (T, C, H, W).

        Pads on the left if fewer than time_steps; keeps the last time_steps if more.
        """
        if not frames:
            raise ValueError("encode_sequence requires at least one frame")

        frame_list: List[ArrayLike] = list(frames)
        if len(frame_list) < self.time_steps:
            frame_list = [frame_list[0]] * (self.time_steps - len(frame_list)) + frame_list
        elif len(frame_list) > self.time_steps:
            frame_list = frame_list[-self.time_steps :]

        spikes_t: List[torch.Tensor] = []
        prev: Optional[torch.Tensor] = None
        for frame in frame_list:
            spike, gray = self.encode_frame(frame, prev_gray=prev)
            spikes_t.append(spike)
            prev = gray
        return torch.stack(spikes_t, dim=0)

    def synthesize_sequence(self, image: ArrayLike, motion_scale: float = 0.03) -> List[np.ndarray]:
        """
        Build a T-frame pseudo-sequence from a still image (offline YOLO training).

        Applies small brightness / noise drift so frame-diff carries weak motion cues.
        Live RTSP should use real buffered frames instead.
        """
        if isinstance(image, torch.Tensor):
            base = image.detach().cpu().numpy()
        else:
            base = np.asarray(image)

        base_f = base.astype(np.float32)
        max_v = 255.0 if base_f.max() > 1.5 else 1.0
        frames: List[np.ndarray] = []
        for t in range(self.time_steps):
            drift = 1.0 + motion_scale * (t / max(self.time_steps - 1, 1) - 0.5)
            noise = np.random.randn(*base_f.shape).astype(np.float32) * (motion_scale * max_v)
            frame = np.clip(base_f * drift + noise, 0.0, max_v)
            if max_v == 255.0:
                frame = frame.astype(np.uint8)
            frames.append(frame)
        return frames
