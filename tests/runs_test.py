import numpy as np
from scipy import stats
from .base_test import BaseTest


class RunsTest(BaseTest):
    def __init__(self, name="Runs Test", significance_level=0.01):
        super().__init__(name, significance_level)

    def _binarize_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Zamienia liczby z [0,1) na ciag binarny:
        1 dla x >= 0.5
        0 dla x < 0.5
        """
        return (sequence >= 0.5).astype(np.uint8)

    def _count_runs(self, binary_sequence: np.ndarray) -> int:
        """
        Liczy liczbe serii w ciagu binarnym.
        """
        if len(binary_sequence) == 0:
            return 0

        return int(1 + np.sum(binary_sequence[1:] != binary_sequence[:-1]))

    def _calculate_statistics(self, binary_sequence: np.ndarray):
        """
        Oblicza liczbe serii, wartosc oczekiwana, wariancje i statystyke Z.
        """
        n = len(binary_sequence)
        n1 = int(np.sum(binary_sequence))
        n0 = n - n1

        runs = self._count_runs(binary_sequence)

        mean_r = (2 * n0 * n1) / n + 1

        var_r = (
            (2 * n0 * n1 * (2 * n0 * n1 - n))
            / (n**2 * (n - 1))
        )

        std_r = np.sqrt(var_r)

        z_stat = (runs - mean_r) / std_r

        return runs, mean_r, var_r, z_stat, n0, n1

    def run(self, sequence: np.ndarray):
        """
        Runs test sprawdza, czy liczba serii w zbinaryzowanej sekwencji
        jest zgodna z oczekiwana dla ciagu losowego.
        """
        binary_sequence = self._binarize_sequence(sequence)

        runs, mean_r, var_r, z_stat, n0, n1 = self._calculate_statistics(binary_sequence)

        # test dwustronny
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        # 4. Werdykt na podstawie p-value i progu alpha
        lower_bound = self.alpha
        upper_bound = 1 - self.alpha

        is_pass = lower_bound < p_value < upper_bound

        return is_pass, p_value