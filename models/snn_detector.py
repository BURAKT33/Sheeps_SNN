"""Spiking Conv detector: multi-sheep CenterNet-style heads."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional, layer, neuron, surrogate


def _conv_lif_block(in_ch: int, out_ch: int, tau: float = 2.0) -> nn.Sequential:
    return nn.Sequential(
        layer.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
        layer.BatchNorm2d(out_ch),
        neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan(), detach_reset=True),
        layer.MaxPool2d(2),
    )


# RGB colors for PIL drawing
CLASS_COLORS_RGB = {
    "sit": (255, 140, 0),
    "stand": (0, 220, 0),
    "eat": (0, 180, 255),
}


class SNNDetector(nn.Module):
    """
    Multi-object SNN detector.

    Outputs dense maps (temporal mean of spike features):
      heatmap:   (B, 1, Hs, Ws) objectness logits
      class_map: (B, num_classes, Hs, Ws) class logits
      size_map:  (B, 2, Hs, Ws) (w, h) in (0, 1)
    """

    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 3,
        base_channels: int = 32,
        time_steps: int = 4,
    ) -> None:
        super().__init__()
        self.time_steps = time_steps
        self.num_classes = num_classes

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 4

        self.backbone = nn.Sequential(
            _conv_lif_block(in_channels, c1),
            _conv_lif_block(c1, c2),
            _conv_lif_block(c2, c3),
            layer.Conv2d(c3, c4, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(c4),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True),
        )

        self.heat_head = nn.Conv2d(c4, 1, kernel_size=1)
        self.class_head = nn.Conv2d(c4, num_classes, kernel_size=1)
        self.size_head = nn.Conv2d(c4, 2, kernel_size=1)

        functional.set_step_mode(self, step_mode="m")

    def forward(self, spikes: torch.Tensor) -> Dict[str, torch.Tensor]:
        if spikes.dim() != 5:
            raise ValueError(f"Expected spikes (T,B,C,H,W), got {tuple(spikes.shape)}")

        x = self.backbone(spikes)  # (T, B, C, Hs, Ws)
        feat = x.mean(dim=0)  # (B, C, Hs, Ws)

        heatmap = self.heat_head(feat)
        class_map = self.class_head(feat)
        size_map = torch.sigmoid(self.size_head(feat)).clamp(0.05, 0.95)
        return {"heatmap": heatmap, "class_map": class_map, "size_map": size_map}


@torch.no_grad()
def decode_detections(
    heatmap: torch.Tensor,
    class_map: torch.Tensor,
    size_map: torch.Tensor,
    conf_threshold: float = 0.25,
    max_det: int = 30,
    nms_kernel: int = 5,
) -> Dict[str, torch.Tensor]:
    """
    Peak decode from dense maps.

    Returns padded:
      boxes (B,max_det,4), scores (B,max_det), labels (B,max_det),
      logits (B,max_det,C), mask (B,max_det)
    """
    bsz, _, hs, ws = heatmap.shape
    device = heatmap.device
    heat = torch.sigmoid(heatmap)

    pad = nms_kernel // 2
    pooled = F.max_pool2d(heat, kernel_size=nms_kernel, stride=1, padding=pad)
    peaks = (heat == pooled) & (heat >= conf_threshold)

    boxes = torch.zeros(bsz, max_det, 4, device=device)
    scores = torch.zeros(bsz, max_det, device=device)
    labels = torch.zeros(bsz, max_det, dtype=torch.long, device=device)
    logits_out = torch.zeros(bsz, max_det, class_map.shape[1], device=device)
    mask = torch.zeros(bsz, max_det, dtype=torch.bool, device=device)

    for b in range(bsz):
        ys, xs = torch.where(peaks[b, 0])
        if ys.numel() == 0:
            flat = heat[b, 0].reshape(-1)
            idx = int(flat.argmax().item())
            ys = torch.tensor([idx // ws], device=device)
            xs = torch.tensor([idx % ws], device=device)

        vals = heat[b, 0, ys, xs]
        order = torch.argsort(vals, descending=True)[:max_det]
        ys, xs, vals = ys[order], xs[order], vals[order]

        for i in range(ys.numel()):
            y = int(ys[i].item())
            x = int(xs[i].item())
            sc = float(vals[i].item())
            cls_logits = class_map[b, :, y, x]
            cls_id = int(cls_logits.argmax().item())
            w = float(size_map[b, 0, y, x].item())
            h = float(size_map[b, 1, y, x].item())
            cx = x / max(ws - 1, 1)
            cy = y / max(hs - 1, 1)
            boxes[b, i] = torch.tensor([cx, cy, w, h], device=device)
            scores[b, i] = sc
            labels[b, i] = cls_id
            logits_out[b, i] = cls_logits
            mask[b, i] = True

    return {
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
        "logits": logits_out,
        "mask": mask,
    }


def reset_model(model: nn.Module) -> None:
    functional.reset_net(model)


def build_model(
    in_channels: int = 2,
    num_classes: int = 3,
    base_channels: int = 32,
    time_steps: int = 4,
) -> SNNDetector:
    return SNNDetector(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        time_steps=time_steps,
    )
