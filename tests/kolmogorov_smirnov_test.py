import numpy as np
from scipy import stats
from .base_test import BaseTest


class KsTest(BaseTest):
    def __init__(self, name="Kolmogorow-Smirnow Test", significance_level=0.01):
        super().__init__(name, significance_level)

    def _calculate_d_statistic(self, sequence: np.ndarray) -> float:
        """
        Samodzielne obliczenie statystyki D testu Kolmogorowa-Smirnowa
        dla porownania z rozkladem jednostajnym U(0,1).
        """
        x = np.sort(sequence)
        n = len(x)

        indices = np.arange(1, n + 1)

        d_plus = np.max(indices / n - x)
        d_minus = np.max(x - (indices - 1) / n)

        d_stat = max(d_plus, d_minus)
        return float(d_stat)

    def run(self, sequence: np.ndarray):
        """
        Test K-S sprawdza najwieksza roznice pomiedzy dystrybuanta empiryczna
        badanej sekwencji a dystrybuanta teoretyczna rozkladu jednostajnego U(0,1).
        """
        d_stat = self._calculate_d_statistic(sequence)

        # p-value wyznaczane na podstawie rozkladu statystyki KS
        _, p_value = stats.kstest(sequence, 'uniform')

        # 4. Werdykt na podstawie p-value i progu alpha
        lower_bound = self.alpha
        upper_bound = 1 - self.alpha

        is_pass = lower_bound < p_value < upper_bound

        return is_pass, p_value