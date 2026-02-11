"""Public FaceSwitch library API."""

from faceswitch.core.interfaces import FaceDetector
from faceswitch.core.types import FaceBox
from faceswitch.detectors.hog import HogDetector, HogDetectorConfig

__all__ = ["FaceBox", "FaceDetector", "HogDetector", "HogDetectorConfig"]

__version__ = "0.1.0"