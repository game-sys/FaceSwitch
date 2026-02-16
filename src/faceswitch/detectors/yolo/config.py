from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class YoloDetectorConfig:
    """Configuration for YOLO-based detector."""

    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    model: str = "yolov8n-face.pt"  # can be a path or model name
    device: str = "cpu"            # "cpu", "cuda", "mps"