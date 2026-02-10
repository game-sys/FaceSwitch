from core.interfaces import FaceDetector
from core.types import FaceBox


def _run_hog_detection(image) -> list[tuple[int, int, int, int]]:
    return []


class HogDetector(FaceDetector):
    def detect(self, image) -> list[FaceBox]:
        detections = _run_hog_detection(image)
        return [
            FaceBox(x1=x, y1=y, x2=x + w, y2=y + h, confidence=None)
            for x, y, w, h in detections
        ]
