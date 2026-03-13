import numpy as np
from generators.lcg_generator import LcgGenerator
from generators.mersenne_twister_generator import MersenneTwister
from generators.crypto_generator import CryptoGenerator
from generators.own_generator import OwnGenerator
from tests.chi_square import ChiSquareTest
from tests.kolmogorov_smirnov_test import KsTest
from tests.runs_test import RunsTest
from tests.monobit_test import MonoBitTest
from tests.serial2d_test import Serial2DTest
from tests.autocorrelation_test import AutocorrelationTest
from tests.poker_test import PokerTest



from utils.visualizations import plot_scatter, plot_histogram


def main():
    n_numbers = 100_000
    n_iterations = 50
    alpha = 0.01

    generators_to_test = [
        LcgGenerator(seed=16),
        MersenneTwister(),
        CryptoGenerator(),
        OwnGenerator(),
    ]

    tests_to_run = [
        ChiSquareTest(),
        KsTest(),
        RunsTest(),
        MonoBitTest(),
        Serial2DTest(),
        AutocorrelationTest(),
        PokerTest(),
    ]

    for generator in generators_to_test:
        # ZBIORCZY NAGŁÓWEK DLA GENERATORA
        print(f"\n\n\n" + "=" * 80)
        print(f"RAPORT GENERALNY: {generator.name.upper()}")
        print(f"Liczba prób: {n_iterations} | Rozmiar próbki: {n_numbers}")
        print("=" * 80)

        # Listy do podsumowania globalnego
        total_tests_run = len(tests_to_run) * n_iterations
        total_passes = 0
        test_summaries = []

        # Pętla po testach (zbieranie danych)
        for test in tests_to_run:
            p_values = []
            test_passes = 0

            for i in range(n_iterations):
                sequence = generator.generate(n_numbers)
                is_pass, p_value = test.run(sequence)
                p_values.append(p_value)
                if is_pass:
                    test_passes += 1

            total_passes += test_passes

            # Zapisujemy statystyki dla danego testu
            test_summaries.append({
                'name': test.name,
                'avg': np.mean(p_values),
                'med': np.median(p_values),
                'min': np.min(p_values),
                'max': np.max(p_values),
                'p_f': f"{test_passes}/{n_iterations - test_passes}",
                'rate': (test_passes / n_iterations) * 100
            })

        # --- SEKCJA 1: PODSUMOWANIE ZBIORCZE GENERATORA ---
        global_pass_rate = (total_passes / total_tests_run) * 100
        print(f"WYNIK OGÓLNY (Global Pass Rate): {global_pass_rate:>6.2f}%")
        print("-" * 80)

        # --- SEKCJA 2: POJEDYNCZE RAPORTY DLA TESTÓW ---
        print(f"{'Nazwa Testu':<22} | {'Średnia P':<10} | {'Mediana':<10} | {'Min/Max':<18} | {'Pass Rate'}")
        print("-" * 80)

        for s in test_summaries:
            min_max = f"{s['min']:.2f}/{s['max']:.2f}"
            print(f"{s['name'][:22]:<22} | {s['avg']:<10.4f} | {s['med']:<10.4f} | {min_max:<18} | {s['rate']:>8.1f}%")


if __name__ == "__main__":
    main()