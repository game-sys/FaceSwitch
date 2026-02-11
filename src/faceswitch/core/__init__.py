"""Core contracts and types for detector implementations."""

from faceswitch.core.interfaces import FaceDetector
from faceswitch.core.types import FaceBox

__all__ = ["FaceBox", "FaceDetector"]
