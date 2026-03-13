import numpy as np
import math
from .base_test import BaseTest


class MonoBitTest(BaseTest):
    def __init__(self, name="Monobit Test", significance_level=0.01):
        super().__init__(name, significance_level)

    def run(self, sequence: np.ndarray):


        bits = self._sequence_to_bits(sequence, bits_per_number=32)


        n = len(bits)

        ones_count = int(np.sum(bits))
        zeros_count = n - ones_count

        transformed = np.where(bits == 0, -1, 1)

        s_n = float(np.sum(transformed))

        s_obs = abs(s_n) / math.sqrt(n)
        p_value = math.erfc(s_obs / math.sqrt(2))


        # 4. Werdykt na podstawie p-value i progu alpha
        lower_bound = self.alpha
        upper_bound = 1 - self.alpha

        is_pass = lower_bound < p_value < upper_bound

        return is_pass, p_value