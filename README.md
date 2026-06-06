# FaceSwitch

**Detect faces in any image with one line of Python — swap the detector without changing your code.**

FaceSwitch is a lightweight Python library that wraps multiple face detection engines behind a single, consistent interface. Pick the detector that suits your needs, swap it later without rewriting anything, and only install what you actually use.

---

## What does it do?

You give it an image. It tells you where the faces are.

```python
from faceswitch.detectors.yolo import YoloDetector
import cv2

image = cv2.imread("photo.jpg")
detector = YoloDetector()
faces = detector.detect(image)

for face in faces:
    print(f"Face at ({face.x1}, {face.y1}) → ({face.x2}, {face.y2})")
```

Every detector returns the same thing — a list of face boxes — so switching from YOLO to HOG (or any other detector) is just changing one import line.

---

## Installation

You need Python 3.10 or newer.

**Step 1 — install the base library:**

```bash
pip install faceswitch
```

**Step 2 — install the detector you want to use:**

| Detector | What it is | Install command |
|---|---|---|
| HOG | Classic CPU-based detector (fast, lightweight, no GPU needed) | `pip install "faceswitch[hog]"` |
| YOLO | Deep learning detector (more accurate, works best with GPU) | `pip install "faceswitch[yolo]"` |
| RetinaFace | High-accuracy ResNet detector (best for difficult angles and small faces) | `pip install "faceswitch[retinaface]"` |

**Or install everything at once:**

```bash
pip install "faceswitch[all]"
```

---

## Quick start — 3 lines to detect faces

```python
import cv2
from faceswitch.detectors.hog import HogDetector   # swap this line to switch detectors

image = cv2.imread("photo.jpg")
faces = HogDetector().detect(image)

print(f"Found {len(faces)} face(s)")
```

To use a different detector, change only the import:

```python
from faceswitch.detectors.yolo import YoloDetector         # YOLO
from faceswitch.detectors.retinaface import RetinaFaceDetector  # RetinaFace
```

Everything else stays exactly the same.

---

## Understanding the results

`detector.detect(image)` always returns a list of `FaceBox` objects. Each one looks like this:

```
FaceBox(x1=120, y1=45, x2=210, y2=160, confidence=0.97)
```

| Field | Meaning |
|---|---|
| `x1`, `y1` | Top-left corner of the face box (pixels) |
| `x2`, `y2` | Bottom-right corner of the face box (pixels) |
| `confidence` | How sure the detector is (0.0 to 1.0). Some detectors don't provide this (`None`). |

**Draw the boxes on your image:**

```python
import cv2
from faceswitch.detectors.yolo import YoloDetector

image = cv2.imread("photo.jpg")
faces = YoloDetector().detect(image)

for face in faces:
    cv2.rectangle(image, (face.x1, face.y1), (face.x2, face.y2), (0, 255, 0), 2)

cv2.imwrite("result.jpg", image)
print(f"Saved result.jpg with {len(faces)} face(s) highlighted")
```

---

## Choosing the right detector

| | HOG | YOLO | RetinaFace |
|---|---|---|---|
| **Speed** | Fast | Medium | Slower |
| **Accuracy** | Basic | High | Very high |
| **GPU needed?** | No | Optional | No |
| **Best for** | Quick scripts, low-power machines | General use, real-time video | Difficult angles, small faces, production use |
| **Install size** | Small (~80MB) | Large (~500MB with torch) | Large (~200MB with TensorFlow) |

**Not sure?** Start with HOG. If it misses faces, switch to YOLO or RetinaFace — your code won't change.

---

## Available detectors

### HOG — `pip install "faceswitch[hog]"`

Uses dlib's Histogram of Oriented Gradients detector. Works entirely on CPU, very fast, good for frontal faces.

```python
from faceswitch.detectors.hog import HogDetector
faces = HogDetector().detect(image)
```

### YOLO — `pip install "faceswitch[yolo]"`

Uses Ultralytics YOLOv8. Deep learning-based, excellent accuracy on varied poses and lighting. Downloads a model on first use (~6MB).

```python
from faceswitch.detectors.yolo import YoloDetector
faces = YoloDetector().detect(image)
```

### RetinaFace — `pip install "faceswitch[retinaface]"`

Uses the serengil/retinaface ResNet+FPN model. Best accuracy on small faces, side profiles, and crowded images. Downloads model weights on first use (~120MB).

```python
from faceswitch.detectors.retinaface import RetinaFaceDetector
faces = RetinaFaceDetector().detect(image)
```

---

## Common questions

**Do I need a GPU?**
No. All detectors run on CPU. YOLO and RetinaFace are faster with a GPU but don't require one.

**I get "ImportError: ... install faceswitch[hog]"**
You installed the base library but not the detector dependency. Run the install command from the table above.

**The detector downloads a model on first use — is that normal?**
Yes. YOLO (~6MB) and RetinaFace (~120MB) download their model weights automatically the first time you use them. After that, they're cached locally.

**Can I use my own image loading library instead of OpenCV?**
Yes, as long as the image is a NumPy array with shape `(height, width, 3)` and `uint8` dtype (standard BGR or RGB image). `cv2.imread()` gives you this automatically.

---

## More detectors coming

FaceSwitch is actively maintained. New detectors are added regularly — run `pip install --upgrade faceswitch` to get the latest.

Current version: **0.3.2** | [GitHub](https://github.com/game-sys/FaceSwitch) | [PyPI](https://pypi.org/project/faceswitch/)
