import secrets
import numpy as np
from .base_generator import BaseGenerator

class CryptoGenerator(BaseGenerator):
    def __init__(self, name="Kryptograficzny (Python secrets)"):
        super().__init__(name)
        self.sys_random = secrets.SystemRandom()

    def generate(self, n: int) -> np.ndarray:
        results = np.zeros(n)
        for i in range(n):
            results[i] = self.sys_random.random()
        return results