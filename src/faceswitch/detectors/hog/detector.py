from __future__ import annotations

from numpy.typing import NDArray

from faceswitch.core.interfaces import FaceDetector
from faceswitch.core.types import FaceBox
from faceswitch.detectors.hog.config import HogDetectorConfig


class HogDetector(FaceDetector):
    """HOG-based frontal face detector using dlib."""

    def __init__(self, config: HogDetectorConfig | None = None) -> None:
        """Initialize and cache dlib's frontal face detector."""
        try:
            import dlib  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "HogDetector requires optional dependency 'dlib'. "
                "Install with: pip install 'faceswitch[hog]'"
            ) from exc

        self.config = config or HogDetectorConfig()
        self._detector = dlib.get_frontal_face_detector()

    def detect(self, image: NDArray[np.uint8]) -> list[FaceBox]:
        """Detect faces from a numpy ndarray (uint8, grayscale or BGR/RGB image)."""
        if image is None:
            return []

        if len(image.shape) == 3:
            gray = (
                0.114 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.299 * image[:, :, 2]
            ).astype("uint8")
        else:
            gray = image

        detections = self._detector(gray, self.config.upsample_times)
        return [
            FaceBox(
                x1=int(face.left()),
                y1=int(face.top()),
                x2=int(face.right()),
                y2=int(face.bottom()),
                confidence=None,
            )
            for face in detections
        ]
