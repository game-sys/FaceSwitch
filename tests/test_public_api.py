from faceswitch import FaceBox, FaceDetector, HogDetector


def test_public_api_exports_classes() -> None:
    assert isinstance(FaceBox, type)
    assert isinstance(FaceDetector, type)
    assert isinstance(HogDetector, type)
