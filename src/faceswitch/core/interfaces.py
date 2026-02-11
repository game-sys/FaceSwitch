from __future__ import annotations

from abc import ABC, abstractmethod

from faceswitch.core.types import FaceBox


class FaceDetector(ABC):
    """Contract for face detector implementations."""

    @abstractmethod
    def detect(self, image) -> list[FaceBox]:
        """Return detected faces from a numpy image array."""
        raise NotImplementedError
