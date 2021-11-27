from abc import abstractmethod
from problems.optimization import OptProblem


class ConstructionProblem(OptProblem):

    def __init__(self, **kwargs):
        super(ConstructionProblem, self).__init__(**kwargs)

    @abstractmethod
    def get_empty_solution(self):
        """Returns a solution to start with.

        :return: "empty" partial solution
        """
        raise NotImplementedError

    @abstractmethod
    def get_construction_values(self, x):
        """Takes a solution x and returns the values of all construction options.

        :param x: partial solution
        :return: List of construction options
        """
        raise NotImplementedError

    @abstractmethod
    def construct(self, x, i):
        """Extend current partial solution using the construction option i.

        :param x: partial solution
        :param i: index of construction option
        :return: (partial) solution
        """
        raise NotImplementedError
