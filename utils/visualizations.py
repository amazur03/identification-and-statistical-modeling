import matplotlib.pyplot as plt
import numpy as np


def plot_scatter(sequence: np.ndarray, title: str = "Wykres punktowy generatora", max_points: int = 100_00):
    """
    Rysuje wykres punktowy 2D na podstawie par sasiadujacych liczb (X_i, X_{i+1}).

    :param sequence: Ciag liczb z przedzialu [0, 1).
    :param title: Tytul wykresu.
    :param max_points: Ograniczenie liczby punktow, zeby nie obciazac zbytnio pamieci przy rysowaniu.
    """
    # Bierzemy tylko tyle punktow, ile trzeba do czytelnego wykresu
    seq_to_plot = sequence[:max_points]

    # Przesuniecie o 1, aby stworzyc pary (X_i, X_{i+1})
    x = seq_to_plot[:-1]
    y = seq_to_plot[1:]

    plt.figure(figsize=(8, 8))
    # s=1 to rozmiar punktu, alpha=0.5 to przezroczystosc (pomaga zauwazyc nakladajace sie punkty)
    plt.scatter(x, y, s=1, alpha=0.5, color='blue')
    plt.title(title)
    plt.xlabel("Wartosc X_i")
    plt.ylabel("Wartosc X_{i+1}")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def plot_histogram(sequence: np.ndarray, bins: int = 20, title: str = "Rozklad wartosci"):
    """
    Rysuje histogram sprawdzajacy rownomiernosc rozkladu.
    """
    plt.figure(figsize=(10, 5))
    plt.hist(sequence, bins=bins, edgecolor='black', alpha=0.7, color='green')
    plt.title(title)
    plt.xlabel("Przedzial [0, 1)")
    plt.ylabel("Liczba wystapien")
    plt.show()