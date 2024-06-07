from typing import Protocol
from abc import abstractmethod
import numpy as np

class Recognizer(Protocol):
    @abstractmethod
    def recognize(self, image: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def recognize_batch(self, data: str) -> np.ndarray:
        raise NotImplementedError