from abc import abstractmethod
from problems.optimization import OptProblem


class IndependenceSystemProblem(OptProblem):

    def __init__(self, **kwargs):
        super(IndependenceSystemProblem, self).__init__(**kwargs)

    @abstractmethod
    def get_elements(self):
        """Returns the elements."""
        raise NotImplementedError

    @abstractmethod
    def c(self, e):
        """Takes an element e and returns its value.

        :param e: element
        :return: value of that element
        """
        raise NotImplementedError

    @abstractmethod
    def is_independent(self, x):
        """Takes a set x and returns whether that set is independent.

        :param x: set
        :return: True iff x is independent
        """
        raise NotImplementedError
