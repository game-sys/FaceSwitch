from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import pytest

from faceswitch.detectors.retinaface.detector import RetinaFaceDetector

LENNA = Path(__file__).parent / "assets" / "Lenna_test_image.png"
IMAGE_W, IMAGE_H = 512, 512
RUNS_FOR_SPEED = 10
MAX_SECONDS_PER_RUN = 3.0


def test_retinaface_optional_dependency_behavior() -> None:
    has_dep = importlib.util.find_spec("retinaface") is not None
    if has_dep:
        detector = RetinaFaceDetector()
        assert detector is not None
    else:
        with pytest.raises(ImportError) as exc:
            RetinaFaceDetector()
        assert "faceswitch[retinaface]" in str(exc.value)


def test_retinaface_detects_lenna_face() -> None:
    pytest.importorskip("retinaface")
    cv2 = pytest.importorskip("cv2")

    image = cv2.imread(str(LENNA))
    assert image is not None, f"Could not read test image: {LENNA}"

    detector = RetinaFaceDetector()
    faces = detector.detect(image)

    assert isinstance(faces, list), "detect() must return a list"
    assert len(faces) >= 1, f"Expected at least 1 face on Lenna, got {len(faces)}"

    for f in faces:
        assert f.x2 > f.x1, f"Invalid bbox: x2={f.x2} <= x1={f.x1}"
        assert f.y2 > f.y1, f"Invalid bbox: y2={f.y2} <= y1={f.y1}"
        assert f.x1 >= 0, f"x1={f.x1} is outside image"
        assert f.y1 >= 0, f"y1={f.y1} is outside image"
        assert f.x2 <= IMAGE_W, f"x2={f.x2} exceeds image width {IMAGE_W}"
        assert f.y2 <= IMAGE_H, f"y2={f.y2} exceeds image height {IMAGE_H}"
        if f.confidence is not None:
            assert 0.0 <= f.confidence <= 1.0, f"confidence={f.confidence} out of [0,1]"
        assert (f.x2 - f.x1) >= 10, f"Box too narrow: {f.x2 - f.x1}px"
        assert (f.y2 - f.y1) >= 10, f"Box too short: {f.y2 - f.y1}px"


def test_retinaface_speed_on_lenna() -> None:
    pytest.importorskip("retinaface")
    cv2 = pytest.importorskip("cv2")

    image = cv2.imread(str(LENNA))
    assert image is not None

    detector = RetinaFaceDetector()
    detector.detect(image)  # warm-up, not counted

    times = []
    for _ in range(RUNS_FOR_SPEED):
        t0 = time.perf_counter()
        detector.detect(image)
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    worst = max(times)

    assert worst <= MAX_SECONDS_PER_RUN, (
        f"Slowest run {worst:.3f}s exceeded limit of {MAX_SECONDS_PER_RUN}s"
    )
    print(
        f"\nretinaface speed over {RUNS_FOR_SPEED} runs: "
        f"avg={avg*1000:.1f}ms  worst={worst*1000:.1f}ms"
    )
