from abc import abstractmethod
from problems.optimization import OptProblem


class ConstructionProblem(OptProblem):

    def __init__(self, **kwargs):
        super(ConstructionProblem, self).__init__(**kwargs)

    @abstractmethod
    def get_elements(self):
        """Returns the elements."""
        raise NotImplementedError

    @abstractmethod
    def is_independent(self, sol, e):
        """Takes a set x and returns whether that set is independent.

        :param x: set
        :return: True iff x is independent
        """
        raise NotImplementedError
