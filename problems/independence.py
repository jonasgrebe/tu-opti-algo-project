from abc import abstractmethod
from problems.optimization import OptProblem


class IndependenceProblem(OptProblem):
    def __init__(self, **kwargs):
        super(IndependenceProblem, self).__init__(**kwargs)

    @abstractmethod
    def get_elements(self, sol):
        """Returns the elements."""
        raise NotImplementedError

    @abstractmethod
    def is_independent(self, sol, element):  # TODO: refactor to 'set'
        """Takes a set e and returns whether that set is independent.

        :param sol: the current solution
        :param element: an element (of the ground set E)
        :return: True iff ele is independent
        """
        raise NotImplementedError

    @abstractmethod
    def costs(self, elements):
        """Returns the cost for a given element."""
        raise NotImplementedError
