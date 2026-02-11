from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FaceBox:
    """Face bounding box in xyxy format."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float | None = None
