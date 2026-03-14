import numpy as np
from .base_generator import BaseGenerator

class SecondOwnGenerator(BaseGenerator):
    def __init__(self, name="SecondOwnGenerator"):
        super().__init__(name)
        self.state = 123456789
        self.a = 7
        self.m = 2 ** 31 - 1

    def generate(self, n: int) -> np.ndarray:
        results = np.zeros(n)
        for i in range(n):
            # Klasyczny krok LCG, ale bez '+ c'
            self.state = (self.a * self.state) % self.m

            # Skalowanie do ułamka [0, 1)
            results[i] = self.state / self.m

        return results