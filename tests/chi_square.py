import numpy as np
from scipy import stats
from .base_test import BaseTest


class ChiSquareTest(BaseTest):
    def __init__(self, name="Test Chi-Square", significance_level=0.01, bins=100):
        super().__init__(name, significance_level)
        self.bins = bins

    def run(self, sequence: np.ndarray):
        """
        Test Chi-kwadrat sprawdzajacy rownomiernosc rozkladu ulamkow.
        Nie uzywamy tu bitow, lecz surowych danych wejsciowych.
        """
        n = len(sequence)

        # 1. Zliczamy ile liczb wpada do kazdego z 'k' koszykow (bins)
        # Przy idealnym rozkladzie w kazdym przedziale (np. 0.0-0.1)
        # powinno byc tyle samo liczb.
        observed_counts, _ = np.histogram(sequence, bins=self.bins, range=(0, 1))

        # 2. Obliczamy liczbe oczekiwana dla kazdego koszyka (E_i = n / k)
        expected_counts = np.full(self.bins, n / self.bins)

        # 3. Wyliczamy statystyke Chi-kwadrat i p-value
        # Wzor: suma((O_i - E_i)^2 / E_i)
        chi2_stat, p_value = stats.chisquare(f_obs=observed_counts, f_exp=expected_counts)

        # 4. Werdykt na podstawie p-value i progu alpha
        lower_bound = self.alpha
        upper_bound = 1 - self.alpha

        is_pass = lower_bound < p_value < upper_bound

        return is_pass, p_value