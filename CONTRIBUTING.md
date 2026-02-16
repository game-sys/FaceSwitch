# Contributing to FaceSwitch

Thank you for your interest in contributing to FaceSwitch! This document provides guidelines and steps for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Adding a New Detector](#adding-a-new-detector)
- [Testing Guidelines](#testing-guidelines)
- [Code Standards](#code-standards)
- [Pull Request Process](#pull-request-process)
- [Project Structure](#project-structure)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive experience for everyone. We expect all contributors to:

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

### Unacceptable Behavior

- Harassment, trolling, or discriminatory comments
- Publishing others' private information without permission
- Other conduct that could reasonably be considered inappropriate

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- Basic understanding of face detection concepts

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/FaceSwitch.git
   cd FaceSwitch
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode**
   ```bash
   pip install -e '.[dev,hog,examples]'
   ```

4. **Run tests to verify setup**
   ```bash
   pytest tests/
   ```

5. **Create a new branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Adding a New Detector

This is the most common contribution type. Follow these steps to add a new face detector (e.g., YOLO, MTCNN, RetinaFace).

### Phase 1: Planning

1. **Create an issue** describing the detector you want to add
   - Detector name and backend library
   - Expected dependencies
   - Any special requirements or considerations

2. **Get feedback** from maintainers before starting implementation

### Phase 2: Implementation

#### Step 1: Create Detector Structure

Create a new folder under `src/faceswitch/detectors/`:

```
src/faceswitch/detectors/
└── yolo/                    # Replace 'yolo' with your detector name
    ├── __init__.py
    ├── config.py
    └── detector.py
```

#### Step 2: Implement Configuration (`config.py`)

Create a frozen dataclass with detector-specific parameters:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class YoloDetectorConfig:
    """Configuration for YOLO detector."""
    
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    model_path: str | None = None
    device: str = "cpu"
```

#### Step 3: Implement Detector (`detector.py`)

Follow this template:

```python
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from faceswitch.core.interfaces import FaceDetector
from faceswitch.core.types import FaceBox
from faceswitch.detectors.yolo.config import YoloDetectorConfig


class YoloDetector(FaceDetector):
    """YOLO-based face detector."""

    def __init__(self, config: YoloDetectorConfig | None = None) -> None:
        """Initialize YOLO detector."""
        try:
            import ultralytics  # Replace with your backend
        except ImportError as exc:
            raise ImportError(
                "YoloDetector requires optional dependency 'ultralytics'. "
                "Install with: pip install 'faceswitch[yolo]'"
            ) from exc

        self.config = config or YoloDetectorConfig()
        # Initialize your detector here
        self._model = None  # Load your model

    def detect(self, image: NDArray[np.uint8]) -> list[FaceBox]:
        """Detect faces from a numpy ndarray."""
        if image is None:
            return []

        # 1. Preprocess image if needed
        # 2. Run inference
        # 3. Post-process results
        # 4. Convert to FaceBox with xyxy coordinates
        
        faces = []
        # Your detection logic here
        
        return faces
```

**Critical Requirements:**
- Must inherit from `FaceDetector`
- Must implement `detect(self, image) -> list[FaceBox]`
- Input: `numpy.ndarray` (uint8, grayscale or BGR/RGB)
- Output: `list[FaceBox]` with **xyxy format** (`x1, y1, x2, y2`)
- `confidence` may be `None` if backend doesn't provide scores
- Raise clear `ImportError` when dependencies are missing
- Keep all logic inside class methods (no module-level helpers)

#### Step 4: Update Package Exports

**`src/faceswitch/detectors/yolo/__init__.py`:**
```python
"""YOLO detector package."""

from faceswitch.detectors.yolo.config import YoloDetectorConfig
from faceswitch.detectors.yolo.detector import YoloDetector

__all__ = ["YoloDetector", "YoloDetectorConfig"]
```

**`src/faceswitch/detectors/__init__.py`:**
```python
"""Detector implementations."""

from faceswitch.detectors.hog import HogDetector, HogDetectorConfig
from faceswitch.detectors.yolo import YoloDetector, YoloDetectorConfig

__all__ = ["HogDetector", "HogDetectorConfig", "YoloDetector", "YoloDetectorConfig"]
```

**`src/faceswitch/__init__.py`** (only when stable):
```python
"""Public FaceSwitch library API."""

from faceswitch.core.interfaces import FaceDetector
from faceswitch.core.types import FaceBox
from faceswitch.detectors.hog import HogDetector, HogDetectorConfig
from faceswitch.detectors.yolo import YoloDetector, YoloDetectorConfig

__all__ = [
    "FaceBox",
    "FaceDetector",
    "HogDetector",
    "HogDetectorConfig",
    "YoloDetector",
    "YoloDetectorConfig",
]

__version__ = "0.1.0"
```

### Phase 3: Dependencies

#### Step 5: Update `pyproject.toml`

Add your detector's dependencies to the `[project.optional-dependencies]` section:

```toml
[project.optional-dependencies]
hog = ["dlib>=19.24"]
yolo = ["ultralytics>=8.0.0", "torch>=2.0.0"]
examples = ["opencv-python>=4.8"]
dev = ["pytest>=8.0"]
```

### Phase 4: Testing

#### Step 6: Create Tests

Create `tests/test_yolo_detector.py`:

```python
import importlib.util
from pathlib import Path

import pytest

from faceswitch import YoloDetector


def test_yolo_detector_optional_dependency_behavior() -> None:
    """Test that missing dependencies raise clear ImportError."""
    has_ultralytics = importlib.util.find_spec("ultralytics") is not None

    if has_ultralytics:
        detector = YoloDetector()
        assert detector is not None
    else:
        try:
            YoloDetector()
            assert False, "Expected ImportError when ultralytics is missing"
        except ImportError as exc:
            assert "faceswitch[yolo]" in str(exc)


def test_yolo_detector_detects_lenna_face() -> None:
    """Test that YOLO detector detects faces correctly."""
    pytest.importorskip("ultralytics")
    cv2 = pytest.importorskip("cv2")

    image_path = Path(__file__).parent / "assets" / "Lenna_test_image.png"
    image = cv2.imread(str(image_path))
    assert image is not None, f"Could not read test image: {image_path}"

    detector = YoloDetector()
    faces = detector.detect(image)

    # Verify output contract
    assert isinstance(faces, list)
    assert len(faces) >= 1
    
    # Verify FaceBox structure and xyxy format
    face = faces[0]
    assert hasattr(face, "x1")
    assert hasattr(face, "y1")
    assert hasattr(face, "x2")
    assert hasattr(face, "y2")
    assert face.x2 > face.x1  # Valid xyxy
    assert face.y2 > face.y1  # Valid xyxy
```

#### Step 7: Update Public API Test

Add your detector to `tests/test_public_api.py`:

```python
from faceswitch import FaceBox, FaceDetector, HogDetector, YoloDetector


def test_public_api_exports_classes() -> None:
    assert isinstance(FaceBox, type)
    assert isinstance(FaceDetector, type)
    assert isinstance(HogDetector, type)
    assert isinstance(YoloDetector, type)
```

#### Step 8: Run Tests

```bash
# Install your detector's dependencies
pip install -e '.[yolo]'

# Run tests
pytest tests/ -v

# Run only your detector tests
pytest tests/test_yolo_detector.py -v
```

### Phase 5: Documentation

#### Step 9: Create Example Script

Create `examples/demo_yolo.py`:

```python
import argparse

import cv2

from faceswitch import YoloDetector


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO face detection demo.")
    parser.add_argument("image_path", help="Path to the input image")
    args = parser.parse_args()

    image = cv2.imread(args.image_path)
    if image is None:
        raise ValueError(f"Could not read image: {args.image_path}")

    detector = YoloDetector()
    faces = detector.detect(image)

    for face in faces:
        cv2.rectangle(image, (face.x1, face.y1), (face.x2, face.y2), (0, 255, 0), 2)
        if face.confidence is not None:
            conf_text = f"{face.confidence:.2f}"
            cv2.putText(
                image, conf_text, (face.x1, face.y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

    print(f"Total faces detected: {len(faces)}")
    cv2.imshow("YOLO Face Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

#### Step 10: Update README

Add a section to `README.md` documenting your detector:

```markdown
### YOLO Detector

```python
import cv2
from faceswitch import YoloDetector

image = cv2.imread("path/to/image.jpg")
detector = YoloDetector()
faces = detector.detect(image)
```

Install dependencies:
```bash
pip install 'faceswitch[yolo]'
```

Run example:
```bash
python examples/demo_yolo.py tests/assets/Lenna_test_image.png
```
```

## Testing Guidelines

### Test Requirements

Every detector must have:

1. **Dependency test**: Verify clear `ImportError` when dependencies missing
2. **Detection test**: Verify detector finds faces in test image
3. **Contract test**: Verify output is `list[FaceBox]` with xyxy coordinates
4. **Public API test**: Verify detector is exported in public API

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_yolo_detector.py

# Run with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=faceswitch
```

### Test Data

Place test images in `tests/assets/`. Use `Lenna_test_image.png` for basic face detection tests.

## Code Standards

### Import Order

Follow this order in every module:

1. Future imports (`from __future__ import annotations`)
2. Standard library imports
3. Third-party library imports
4. Local package imports (`faceswitch...`)

Example:
```python
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from faceswitch.core.interfaces import FaceDetector
from faceswitch.core.types import FaceBox
```

### Type Hints

- Use type hints for all public methods
- Use `from __future__ import annotations` for forward references
- Use `numpy.typing.NDArray` for numpy arrays

### Code Style

- Follow PEP 8
- Use dataclasses for configuration
- Use frozen dataclasses when possible
- Keep lines under 88 characters (Black default)
- Use descriptive variable names

### Documentation

- Add docstrings to all public classes and methods
- Use Google-style docstrings
- Document parameters, return values, and exceptions

Example:
```python
def detect(self, image: NDArray[np.uint8]) -> list[FaceBox]:
    """Detect faces from a numpy ndarray.
    
    Args:
        image: Input image as numpy array (uint8, grayscale or BGR/RGB).
        
    Returns:
        List of FaceBox objects containing detected face locations.
        
    Raises:
        ValueError: If image is invalid.
    """
```

### Architecture Rules

**Core Layer:**
- `faceswitch.core` must not depend on detector implementations
- `faceswitch.core` must not depend on heavy backends (dlib, torch, etc.)
- Core defines contracts; detectors implement them

**Detector Layer:**
- Must inherit `FaceDetector` interface
- Must implement `detect(image) -> list[FaceBox]`
- Keep all logic in class methods (no module-level helpers)
- Raise clear `ImportError` for missing dependencies
- Output coordinates must be xyxy format

**Dependencies:**
- Optional dependencies must be declared in `pyproject.toml`
- Use `[detector_name]` extras for optional backends

## Pull Request Process

### Before Submitting

- [ ] Code follows the style guidelines
- [ ] Detector inherits `FaceDetector` and implements contract correctly
- [ ] Output uses `FaceBox` with xyxy coordinates
- [ ] Clear `ImportError` with install instructions
- [ ] Tests pass (`pytest tests/`)
- [ ] Tests added for new functionality
- [ ] Documentation updated (README, examples)
- [ ] Dependencies added to `pyproject.toml`
- [ ] Public API exports updated
- [ ] Commit messages are clear and descriptive

### PR Title Format

- `feat: Add YOLO face detector`
- `fix: Fix HOG detector grayscale conversion`
- `docs: Update installation instructions`
- `test: Add tests for MTCNN detector`
- `refactor: Improve detector base class`

### PR Description Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] New detector
- [ ] Bug fix
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Other (please describe)

## Detector Details (if applicable)
- **Detector Name**: YOLO
- **Backend Library**: ultralytics
- **Dependencies**: ultralytics>=8.0.0, torch>=2.0.0

## Testing
- [ ] Tests pass locally
- [ ] New tests added
- [ ] Tested on sample images

## Checklist
- [ ] Code follows project style guidelines
- [ ] Output uses xyxy coordinate format
- [ ] Clear ImportError for missing dependencies
- [ ] Documentation updated
- [ ] Example script added
```

### Review Process

1. Maintainers will review your PR within a few days
2. Address any requested changes
3. Once approved, maintainers will merge your PR
4. Your contribution will be included in the next release!

## Project Structure

```
FaceSwitch/
├── src/
│   └── faceswitch/
│       ├── __init__.py              # Public API exports
│       ├── core/
│       │   ├── __init__.py
│       │   ├── interfaces.py        # FaceDetector interface
│       │   └── types.py             # FaceBox dataclass
│       └── detectors/
│           ├── __init__.py
│           ├── hog/
│           │   ├── __init__.py
│           │   ├── config.py
│           │   └── detector.py
│           └── yolo/                # Your new detector
│               ├── __init__.py
│               ├── config.py
│               └── detector.py
├── tests/
│   ├── test_public_api.py
│   ├── test_hog_detector.py
│   ├── test_yolo_detector.py       # Your tests
│   └── assets/
│       └── Lenna_test_image.png
├── examples/
│   ├── demo_hog.py
│   └── demo_yolo.py                # Your example
├── pyproject.toml
├── README.md
├── ARCHITECTURE.md
├── CONTRIBUTING.md
└── LICENSE
```

## Questions?

If you have questions or need help:

1. Check existing issues and discussions
2. Read [ARCHITECTURE.md](ARCHITECTURE.md) for design principles
3. Create a new issue with your question
4. Tag it with `question` label

## Recognition

All contributors will be recognized in our README and release notes. Thank you for helping make FaceSwitch better! 🎉
