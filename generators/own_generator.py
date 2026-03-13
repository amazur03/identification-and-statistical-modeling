import numpy as np
from .base_generator import BaseGenerator

class OwnGenerator(BaseGenerator):
    def __init__(self, name="OwnGenerator"):
        super().__init__(name)

    def generate(self, n: int) -> np.ndarray:
        results = np.zeros(n)
        x = 0.01

        for i in range(1, n):
            results[i] = (results[i-1] + x) % 1
        return results