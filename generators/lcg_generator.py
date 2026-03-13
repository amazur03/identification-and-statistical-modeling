import numpy as np
from .base_generator import BaseGenerator


class LcgGenerator(BaseGenerator):
    def __init__(self, name="LCG (Linear congruential generator)", a=1664525, c=1013904223, m=2 ** 32, seed=1):
        super().__init__(name)
        self.a = a
        self.c = c
        self.m = m
        self.state = seed

    def generate(self, n: int) -> np.ndarray:
        results = np.zeros(n)

        for i in range(n):
            # 1. Krok matematyczny: X_{n+1} = (a * X_n + c) mod m
            # Obliczamy nowa wartosc i od razu nadpisujemy stary stan
            self.state = (self.a * self.state + self.c) % self.m

            # 2. Skalowanie: dzielimy nowy stan przez 'm',
            # aby uzyskac ulamek z przedzialu [0.0, 1.0) i zapisujemy do tablicy
            results[i] = self.state / self.m

        return results