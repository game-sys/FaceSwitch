from __future__ import annotations

import os
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from numpy.typing import NDArray

from faceswitch.core.interfaces import FaceDetector
from faceswitch.core.types import FaceBox
from faceswitch.detectors.yolo.config import YoloDetectorConfig


class YoloDetector(FaceDetector):
    """YOLO-based face detector (downloads weights on demand)."""

    # Put the real URL(s) you want to support
    _MODEL_URLS: dict[str, str] = {
        "yolov8n-face.pt": "https://github.com/YapaLab/yolo-face/releases/download/1.0.0/yolov8n-face.pt",
        # "yolov11n-face.pt": "...",
        # "yolov12n-face.pt": "...",
    }

    def __init__(self, config: YoloDetectorConfig | None = None) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "YoloDetector requires optional dependency 'ultralytics'. "
                "Install with: pip install \"faceswitch[yolo]\""
            ) from exc

        self.config = config or YoloDetectorConfig()

        weights_path = self._get_or_download_weights(self.config.model)
        self._model = YOLO(str(weights_path))
        self._device = self.config.device

    def _get_or_download_weights(self, model_name: str) -> Path:
        # Default: user cache (works everywhere)
        target_dir = Path.home() / ".faceswitch" / "weights" / "yolo"
        target_dir.mkdir(parents=True, exist_ok=True)

        model_path = target_dir / model_name
        if model_path.exists() and model_path.stat().st_size > 0:
            return model_path

        url = self._MODEL_URLS.get(model_name)
        if not url:
            raise ValueError(
                f"Unknown YOLO model '{model_name}'. Supported: {list(self._MODEL_URLS)}"
            )

        tmp_path = model_path.with_suffix(model_path.suffix + ".part")
        try:
            urlretrieve(url, tmp_path)
            os.replace(tmp_path, model_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

        return model_path

    def detect(self, image: NDArray[np.uint8]) -> list[FaceBox]:
        if image is None:
            raise ValueError("image cannot be None")

        results = self._model.predict(
            source=image,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            device=self._device,
            verbose=False,
        )

        if not results or results[0].boxes is None:
            return []

        xyxy = results[0].boxes.xyxy
        conf = getattr(results[0].boxes, "conf", None)

        xyxy_np = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else np.asarray(xyxy)
        conf_np = (
            conf.cpu().numpy()
            if (conf is not None and hasattr(conf, "cpu"))
            else None
        )

        out: list[FaceBox] = []
        for i, (x1, y1, x2, y2) in enumerate(xyxy_np):
            score = float(conf_np[i]) if conf_np is not None else None
            out.append(
                FaceBox(
                    x1=int(x1),
                    y1=int(y1),
                    x2=int(x2),
                    y2=int(y2),
                    confidence=score,
                )
            )
        return out