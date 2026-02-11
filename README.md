# FaceSwitch

FaceSwitch is an installable Python library for face detection backends.

## Install

```bash
pip install -e .
```

Install the HOG backend dependency:

```bash
pip install -e '.[hog]'
```

## Minimal Usage

```python
import cv2
from faceswitch import HogDetector

image = cv2.imread("tests/Lenna_test_image.png")
detector = HogDetector()
faces = detector.detect(image)
print(f"Detected: {len(faces)}")
```

## Example Script

```bash
pip install -e '.[hog,examples]'
python examples/demo_hog.py tests/Lenna_test_image.png
```
