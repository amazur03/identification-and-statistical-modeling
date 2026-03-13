import numpy as np
from scipy import stats
from .base_test import BaseTest


class AutocorrelationTest(BaseTest):
    def __init__(self, name="Autocorrelation Test", significance_level=0.01, lag=1):
        super().__init__(name, significance_level)
        self.lag = lag

    def run(self, sequence: np.ndarray):
        """
        Test autokorelacji dla sekwencji liczb z przedzialu [0,1).
        Sprawdza, czy probki x_i oraz x_{i+lag} sa liniowo skorelowane.
        """
        n = len(sequence)

        if n <= self.lag:
            return False, {
                "p_value": 0.0,
                "correlation": None,
                "lag": self.lag,
                "error": "Sekwencja jest zbyt krotka wzgledem zadanego opoznienia."
            }

        x = sequence[:-self.lag]
        y = sequence[self.lag:]

        # Wspolczynnik korelacji Pearsona
        correlation = np.corrcoef(x, y)[0, 1]

        # Statystyka t dla testu istotnosci korelacji
        m = len(x)

        # Zabezpieczenie na wypadek korelacji rownej +/-1
        if abs(correlation) >= 1.0:
            p_value = 0.0
        else:
            t_stat = correlation * np.sqrt((m - 2) / (1 - correlation**2))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=m - 2))

        is_pass = p_value >= self.alpha

        # 4. Werdykt na podstawie p-value i progu alpha
        lower_bound = self.alpha
        upper_bound = 1 - self.alpha

        is_pass = lower_bound < p_value < upper_bound

        return is_pass, p_value