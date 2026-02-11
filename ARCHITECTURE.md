# Architecture

## 1) High-Level Overview
`faceswitch` is a Python face-detection library organized with `src/` layout.

Architecture is split into two layers:
- `faceswitch.core`: stable contracts (`BaseDetector` interface and `FaceBox` type)
- `faceswitch.detectors`: concrete detector implementations (for example `HogDetector`)

Core defines what detectors must do. Detectors define how detection is performed.

## 2) Layered Dependency Rules
Required dependency direction:
- `faceswitch.detectors` -> `faceswitch.core`
- `faceswitch.core` -> no detector modules
- `faceswitch.core` -> no heavy/optional runtime backends (`opencv`, `dlib`, `torch`)

Rules:
- Core is dependency-light and backend-agnostic.
- Detector modules may depend on backend-specific libraries.
- Cross-detector imports are discouraged unless explicitly shared via core contracts.

## 3) Detector Implementation Rules
Every detector must follow these rules:
- Be a class inheriting `BaseDetector` (the project interface in code is `FaceDetector`).
- Implement `detect(self, image) -> List[FaceBox]`.
- Keep detection logic inside class methods.
- Do not use module-level detection helper functions.
- Input contract: `image` is a `numpy.ndarray`.
- Output contract: list of `FaceBox` objects.
- Bounding boxes must use `xyxy` (`x1, y1, x2, y2`).
- `confidence` may be `None` if backend does not provide scores.

## 4) Public API Rules
Public API is defined in `faceswitch/__init__.py`.

Rules:
- Export only stable, user-facing classes and types.
- Avoid exposing private modules, backend internals, or temporary utilities.
- Adding/removing public exports is a versioned API change.

## 5) Import Order Standard
Use this import order in every module:
1. Future imports (`from __future__ import ...`)
2. Python standard library
3. Third-party libraries
4. Local package imports (`faceswitch...`)

Formatting rules:
- One import block per group.
- Keep imports explicit.
- Avoid wildcard imports.

## 6) Error Handling Standards
Dependency and runtime errors must be explicit.

Rules:
- Optional backend dependencies must raise clear `ImportError` with install guidance.
- Do not silently fallback to a different backend.
- Validate obvious input contract violations early.
- Error messages must identify the failing detector and missing requirement.

## 7) Testing Requirements
Minimum test coverage for each detector/backend:
- Public API export test.
- Dependency behavior test (missing optional dependency -> clear `ImportError`).
- Detector construction test when dependency is present.
- Contract tests: output is `List[FaceBox]` and coordinates are `xyxy`.

General:
- Tests live under `tests/`.
- Use `pytest`.
- Add regression tests for bug fixes.

## 8) Versioning Strategy
Use Semantic Versioning (`MAJOR.MINOR.PATCH`).

Rules:
- `PATCH`: bug fixes without API changes.
- `MINOR`: backward-compatible features (new detectors, new optional extras).
- `MAJOR`: breaking API/contract changes.

Public API and contract changes require explicit version bump rationale.

## 9) Contribution Guidelines for New Detectors
When adding a detector:
1. Add implementation in `src/faceswitch/detectors/`.
2. Implement `BaseDetector`/`FaceDetector` contract.
3. Keep all backend logic in the class.
4. Add optional dependency entry in `pyproject.toml` extras if needed.
5. Raise clear `ImportError` when dependency is unavailable.
6. Add tests for constructor/dependency behavior and output contract.
7. Export detector through public API only when considered stable.

## 10) Folder Structure
Expected structure:

```text
src/
  faceswitch/
    __init__.py            # public API
    core/
      __init__.py
      interfaces.py        # BaseDetector/FaceDetector contract
      types.py             # FaceBox
    detectors/
      __init__.py
      hog/
        __init__.py
        config.py
        detector.py        # HogDetector implementation
examples/
  demo_hog.py              # runnable usage scripts
tests/
  test_public_api.py
  test_hog_detector.py
pyproject.toml
README.md
ARCHITECTURE.md
```

Enforcement note:
- Changes violating layer rules or detector contracts should not be merged.
