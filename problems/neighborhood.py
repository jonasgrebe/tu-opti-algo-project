from abc import abstractmethod
from problems.optimization import OptProblem


class NeighborhoodProblem(OptProblem):

    def __init__(self, **kwargs):
        super(NeighborhoodProblem, self).__init__(**kwargs)

    @abstractmethod
    def get_neighborhood(self, x):
        """Returns the neighborhood of feasible instance x.

        :param x: feasible solution x
        :return: List of (feasible) solutions
        """
        raise NotImplementedError

    @abstractmethod
    def get_initial_solution(self):
        """Returns an arbitrary initial solution.

        :return: feasible initial solution
        """
        raise NotImplementedError
