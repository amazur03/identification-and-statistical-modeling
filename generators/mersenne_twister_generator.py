import random
import numpy as np
from .base_generator import BaseGenerator

class MersenneTwister(BaseGenerator):
    def __init__(self, name="Mersenne Twister (Python random)"):
        super().__init__(name)

    def generate(self, n: int) -> np.ndarray:
        results = np.zeros(n)
        for i in range(n):
            results[i] = random.random()
        return results