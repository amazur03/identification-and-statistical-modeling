from abc import ABC, abstractmethod
import numpy as np


class BaseGenerator(ABC):
    """
    Abstrakcyjna klasa bazowa dla wszystkich generatorow liczb pseudolosowych (PRNG).
    Kazdy nowy generator musi dziedziczyc po tej klasie i implementowac metode 'generate'.
    """

    def __init__(self, name: str):
        """
        Inicjalizuje podstawowe wlasciwosci generatora.

        :param name: Zrozumiala dla czlowieka nazwa generatora (np. "Mersenne Twister").
        """
        self.name = name

    @abstractmethod
    def generate(self, n: int) -> np.ndarray:
        """
        Generuje ciag liczb pseudolosowych. To jest metoda abstrakcyjna -
        jej cialo musi zostac napisane w klasie dziedziczacej.

        :param n: Liczba probek do wygenerowania.
        :return: Jednowymiarowa tablica numpy (np.ndarray) o dlugosci 'n',
                 zawierajaca wartosci zmiennoprzecinkowe z przedzialu [0.0, 1.0).
        """
        pass