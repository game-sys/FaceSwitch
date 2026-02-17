import importlib.util
from pathlib import Path

import pytest

from faceswitch import HogDetector


def test_hog_detector_optional_dependency_behavior() -> None:
    has_dlib = importlib.util.find_spec("dlib") is not None

    if has_dlib:
        detector = HogDetector()
        assert detector is not None
    else:
        try:
            HogDetector()
            assert False, "Expected ImportError when dlib is missing"
        except ImportError as exc:
            assert "faceswitch[hog]" in str(exc)


def test_hog_detector_detects_lenna_face() -> None:
    pytest.importorskip("dlib")
    cv2 = pytest.importorskip("cv2")

    image_path = Path(__file__).parent / "assets" / "Lenna_test_image.png"
    image = cv2.imread(str(image_path))
    assert image is not None, f"Could not read test image: {image_path}"

    detector = HogDetector()
    faces = detector.detect(image)

    assert len(faces) >= 1
    # Validate xyxy contract
    for face in faces:
        assert face.x2 > face.x1, f"Invalid bbox: x2={face.x2} should be > x1={face.x1}"
        assert face.y2 > face.y1, f"Invalid bbox: y2={face.y2} should be > y1={face.y1}"
