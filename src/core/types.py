from dataclasses import dataclass


@dataclass(frozen=True)
class FaceBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float | None = None
