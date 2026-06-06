"""Detector implementations."""

from faceswitch.detectors.hog import HogDetector, HogDetectorConfig

__all__ = ["HogDetector", "HogDetectorConfig"]

try:
    from faceswitch.detectors.yolo import YoloDetector, YoloDetectorConfig  # noqa: F401

    __all__ += ["YoloDetector", "YoloDetectorConfig"]
except ImportError:
    pass

try:
    from faceswitch.detectors.retinaface import RetinaFaceDetector, RetinaFaceDetectorConfig  # noqa: F401

    __all__ += ["RetinaFaceDetector", "RetinaFaceDetectorConfig"]
except ImportError:
    pass
