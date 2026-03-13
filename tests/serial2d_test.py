import numpy as np
from scipy import stats
from .base_test import BaseTest


class Serial2DTest(BaseTest):
    def __init__(self, name="Serial Test 2D", significance_level=0.01, grid_size=20):
        super().__init__(name, significance_level)
        self.grid_size = grid_size

    def run(self, sequence: np.ndarray):
        """
        Serial test 2D.
        Tworzy nieprzesuwane pary (x1, x2), (x3, x4), ...
        i sprawdza rownomiernosc pokrycia kwadratu [0,1)^2.
        """
        n = len(sequence)

        # Jesli liczba elementow jest nieparzysta, odrzucamy ostatni
        if n % 2 != 0:
            sequence = sequence[:-1]
            n -= 1

        # Tworzenie par
        pairs = sequence.reshape(-1, 2)
        m = len(pairs)

        k = self.grid_size

        # Histogram 2D
        counts, _, _ = np.histogram2d(
            pairs[:, 0],
            pairs[:, 1],
            bins=k,
            range=[[0.0, 1.0], [0.0, 1.0]]
        )

        expected = m / (k * k)

        chi_stat = np.sum((counts - expected) ** 2 / expected)

        df = k * k - 1
        p_value = 1 - stats.chi2.cdf(chi_stat, df)


        # 4. Werdykt na podstawie p-value i progu alpha
        lower_bound = self.alpha
        upper_bound = 1 - self.alpha

        is_pass = lower_bound < p_value < upper_bound

        return is_pass, p_value