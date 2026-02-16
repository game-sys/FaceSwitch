"""Public FaceSwitch library API."""

from faceswitch.core.interfaces import FaceDetector
from faceswitch.core.types import FaceBox
from faceswitch.detectors.hog import HogDetector, HogDetectorConfig

__all__ = ["FaceBox", "FaceDetector", "HogDetector", "HogDetectorConfig"]

try:
    from faceswitch.detectors import YoloDetector, YoloDetectorConfig  # noqa: F401

    __all__ += ["YoloDetector", "YoloDetectorConfig"]
except ImportError:
    pass

__version__ = "0.2.1"