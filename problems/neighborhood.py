from abc import abstractmethod
from problems.optimization import OptProblem


class NeighborhoodProblem(OptProblem):

    def __init__(self, **kwargs):
        super(NeighborhoodProblem, self).__init__(**kwargs)

    @abstractmethod
    def get_neighborhood(self, sol):
        """Returns the neighborhood of feasible instance x.

        :param sol: feasible solution x
        :return: List of (feasible) solutions
        """
        raise NotImplementedError

    @abstractmethod
    def get_next_neighbors(self, sol):
        """Yields the next set of neighbors not yielded yet.

        :param sol: feasible solution x
        :return: Next (feasible) solution
        """
        raise NotImplementedError

    @abstractmethod
    def get_arbitrary_solution(self):
        """Returns an arbitrary initial solution.

        :return: feasible initial solution
        """
        raise NotImplementedError

    def update_relaxation(self, step):
        pass

    def reset_relaxation(self):
        pass

    def is_relaxation_active(self):
        pass

    def is_relaxation_enabled(self):
        pass

    def toggle_relaxation(self, value):
        pass
