# FaceSwitch

FaceSwitch is a Python library that provides a common interface for multiple face detection backends.

## Features

- Clean detector interface
- Optional backend dependencies via extras
- Pluggable architecture
- Automatic model download for supported backends
- Typed API (`FaceBox`)

## Requirements

- Python 3.10+

## Installation

```bash
pip install faceswitch
```

Install a specific backend via extras:

```bash
pip install "faceswitch[<backend>]"
```

Examples:

```bash
pip install "faceswitch[hog]"
pip install "faceswitch[yolo]"
```

Install demo dependencies (`opencv-python`):

```bash
pip install "faceswitch[examples]"
```

Install all optional dependencies:

```bash
pip install "faceswitch[all]"
```

## Backend Model

- Each detector backend is optional and installed through extras.
- New backends can be added without changing the core detection interface.
- Current backend extras include `hog` and `yolo`.

## Minimal Usage (Backend-Agnostic)

```python
import cv2
from faceswitch.detectors.hog import HogDetector

image = cv2.imread("path/to/image.png")
if image is None:
    raise ValueError("Could not read image")

detector = HogDetector()
faces = detector.detect(image)

print(f"Detected: {len(faces)}")
```

To switch backend later, replace `HogDetector` with another detector class from `faceswitch` (for example, `YoloDetector`) and install its matching extra.

## Detector Interface

```python
faces = detector.detect(image)
```

All detectors return a list of `FaceBox` values:

```python
FaceBox(
    x1=int,  # left
    y1=int,  # top
    x2=int,  # right
    y2=int,  # bottom
    confidence=float | None,
)
```

Some detectors may include backend-specific behavior (for example, model download/caching) documented in their module or config.

## Run Demos

```bash
python examples/demo_hog.py path/to/image.png
python examples/demo_yolo.py path/to/image.png
```

## Adding New Backends

FaceSwitch is designed to grow with more detector implementations. For contribution workflow and detector contract requirements, see `CONTRIBUTING.md` and `ARCHITECTURE.md`.
