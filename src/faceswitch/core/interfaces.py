from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

from faceswitch.core.types import FaceBox


class FaceDetector(ABC):
    """Contract for face detector implementations."""

    @abstractmethod
    def detect(self, image: "NDArray[np.uint8]") -> list[FaceBox]:
        """Return detected faces from a numpy image array.
        
        Args:
            image: Input image as numpy array (uint8, grayscale or BGR/RGB).
            
        Returns:
            List of FaceBox objects containing detected face locations.
        """
        raise NotImplementedError
