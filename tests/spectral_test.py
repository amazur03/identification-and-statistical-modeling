import numpy as np
import math
from .base_test import BaseTest


class SpectralTest(BaseTest):
    def __init__(self, name="Spectral Test", significance_level=0.01):
        super().__init__(name, significance_level)

    def run(self, sequence: np.ndarray):
        """
        Test widmowy DFT w stylu NIST, wykonywany na strumieniu bitow.
        """
        bits = self._sequence_to_bits(sequence, bits_per_number=32)

        n = len(bits)

        if n < 2:
            return False, 0.0

        # Zamiana 0 -> -1, 1 -> +1
        x = np.where(bits == 0, -1, 1)

        # FFT
        spectrum = np.fft.fft(x)
        magnitudes = np.abs(spectrum[:n // 2])

        if len(magnitudes) == 0:
            return False, 0.0

        # Prog zgodny z idea testu NIST DFT
        threshold = math.sqrt(math.log(1 / 0.05) * n)

        # Liczba pikow ponizej progu
        count_below_threshold = np.sum(magnitudes < threshold)

        expected = 0.95 * (n / 2)
        variance = n * 0.95 * 0.05 / 4

        if variance == 0:
            return False, 0.0

        d = (count_below_threshold - expected) / math.sqrt(variance)

        p_value = math.erfc(abs(d) / math.sqrt(2))

        is_pass = p_value >= self.alpha

        return is_pass, p_value