from __future__ import annotations


import numpy as np
from numpy.typing import NDArray

from faceswitch.core.interfaces import FaceDetector
from faceswitch.core.types import FaceBox
from faceswitch.detectors.retinaface.config import RetinaFaceDetectorConfig


class RetinaFaceDetector(FaceDetector):
    """RetinaFace face detector."""

    def __init__(self, config: RetinaFaceDetectorConfig | None = None) -> None:
        try:
            from retinaface import RetinaFace as _rf  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "RetinaFaceDetector requires optional dependency 'retinaface'. "
                "Install with: pip install 'faceswitch[retinaface]'"
            ) from exc

        self.config = config or RetinaFaceDetectorConfig()
        self._model = _rf

    def detect(self, image: NDArray[np.uint8]) -> list[FaceBox]:
        """Detect faces and return list of FaceBox (xyxy)."""
        if image is None:
            return []

        results = self._model.detect_faces(image)
        if not isinstance(results, dict):
            return []
        boxes = []
        for face in results.values():
            x1, y1, x2, y2 = face["facial_area"]
            score = float(face.get("score", 1.0))
            boxes.append(FaceBox(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2), confidence=score))
        return boxes
