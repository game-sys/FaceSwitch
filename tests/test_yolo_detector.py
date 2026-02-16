from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def test_yolo_optional_dependency() -> None:
    has_ultralytics = importlib.util.find_spec("ultralytics") is not None
    if not has_ultralytics:
        from faceswitch.detectors.yolo.detector import YoloDetector

        with pytest.raises(ImportError) as exc:
            YoloDetector()
        assert "faceswitch[yolo]" in str(exc.value)


def test_yolo_detect_lenna_contract() -> None:
    pytest.importorskip("ultralytics")
    cv2 = pytest.importorskip("cv2")

    from faceswitch.detectors.yolo.detector import YoloDetector

    image_path = Path(__file__).parent / "assets" / "Lenna_test_image.png"
    image = cv2.imread(str(image_path))
    assert image is not None

    detector = YoloDetector()
    faces = detector.detect(image)

    assert isinstance(faces, list)
    # may be 0 if model isn't face-tuned; contract test still valid:
    for f in faces:
        assert f.x2 > f.x1
        assert f.y2 > f.y1