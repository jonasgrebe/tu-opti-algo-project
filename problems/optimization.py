from abc import ABC as AbstractClass
from abc import abstractmethod


class OptProblem(AbstractClass):

    def __init__(self, is_max):
        self.is_max = is_max

    @abstractmethod
    def f(self, x):
        """Objective function of this optimization problem.

        :param x: the solution to evaluate
        :return: the value of x
        """
        raise NotImplementedError

    @abstractmethod
    def is_feasible(self, x):
        """Returns whether x is a feasible solution or not.

        :param x: the solution to check
        :return: True if x is feasible else False
        """
        raise NotImplementedError
