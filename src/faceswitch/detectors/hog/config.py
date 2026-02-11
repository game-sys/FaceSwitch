from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HogDetectorConfig:
    """Configuration for dlib HOG detector."""

    upsample_times: int = 1
