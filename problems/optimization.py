from abc import ABC as AbstractClass
from abc import abstractmethod


class OptProblem(AbstractClass):

    def __init__(self, is_max):
        self.is_max = is_max

    @abstractmethod
    def objective_function(self, sol):
        """Objective function of this optimization problem.

        :param sol: the solution to evaluate
        :return: the value of x
        """
        raise NotImplementedError

    @abstractmethod
    def heuristic(self, sol):
        """Heuristic function, can be used instead of the objective function.

        :param sol: the solution to evaluate
        :return: the value of x
        """
        raise NotImplementedError

    @abstractmethod
    def is_feasible(self, sol):
        """Returns whether x is a feasible solution or not.

        :param sol: the solution to check
        :return: True if x is feasible else False
        """
        raise NotImplementedError


class Solution:
    pass
