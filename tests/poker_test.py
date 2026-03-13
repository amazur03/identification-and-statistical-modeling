import numpy as np
from scipy import stats
from .base_test import BaseTest


class PokerTest(BaseTest):
    def __init__(self, name="Poker Test", significance_level=0.01, m=4):
        super().__init__(name, significance_level)
        self.m = m

    def run(self, sequence: np.ndarray):
        # 1. Pobieramy bity
        bits = self._sequence_to_bits(sequence)

        # 2. Dzielimy bity na grupy po m (np. 4 bity)
        # Ucinamy koncowke, ktora nie tworzy pelnego bloku
        n = len(bits)
        n_blocks = n // self.m
        reshaped_bits = bits[:n_blocks * self.m].reshape((n_blocks, self.m))

        # 3. Zamieniamy kazdy blok bitow na liczbe dziesietna (wzorzec)
        # Przyklad: [1, 0, 1, 0] -> 10
        powers = 2 ** np.arange(self.m - 1, -1, -1)
        patterns = np.dot(reshaped_bits, powers)

        # 4. Liczymy wystapienia kazdego wzorca (0 do 2^m - 1)
        observed_counts = np.bincount(patterns, minlength=2 ** self.m)

        # 5. Statystyka Chi-kwadrat
        # Oczekujemy, ze kazdy wzorzec wystapi n_blocks / 2^m razy
        expected_counts = np.full(2 ** self.m, n_blocks / 2 ** self.m)
        _, p_value = stats.chisquare(f_obs=observed_counts, f_exp=expected_counts)

        # 6. Werdykt dwustronny
        is_pass = self.alpha < p_value < (1 - self.alpha)

        return is_pass, p_value