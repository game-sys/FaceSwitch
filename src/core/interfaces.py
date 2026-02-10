from abc import ABC, abstractmethod

from core.types import FaceBox


class FaceDetector(ABC):
    @abstractmethod
    def detect(self, image) -> list[FaceBox]:
        pass
