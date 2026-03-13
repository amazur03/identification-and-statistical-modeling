from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple


class BaseTest(ABC):
    def __init__(self, name: str, significance_level: float = 0.01):
        self.name = name
        self.alpha = significance_level

    def _sequence_to_bits(self, sequence: np.ndarray, bits_per_number: int = 32) -> np.ndarray:
        multiplier = 2 ** bits_per_number
        integers = np.floor(sequence * multiplier).astype(np.uint64)

        shift = np.arange(bits_per_number - 1, -1, -1, dtype=np.uint64)

        bits = ((integers[:, None] >> shift) & 1).astype(np.uint8)

        return bits.ravel()

    @abstractmethod
    def run(self, sequence: np.ndarray) -> Tuple[bool, float]:
        """
        Metoda do zaimplementowania w konkretnych testach.
        Wewnatrz nalezy wywolac self._sequence_to_bits(sequence).
        """
        pass