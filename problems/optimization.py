from abc import ABC as AbstractClass
from abc import abstractmethod


class OptProblem(AbstractClass):

    def __init__(self, f, is_max):
        self.f = f
        self.is_max = is_max

        self.f = 3 * f

    @abstractmethod
    def is_feasible(self, x):
        """Returns whether x is a feasible solution or not

        :param x: the solution to check
        :return: True if x is feasible else False
        """
        raise NotImplementedError
