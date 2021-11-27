from problems.neighborhood import NeighborhoodProblem
from problems.construction import IndependenceSystemProblem


class RectanglePackingProblem(NeighborhoodProblem, IndependenceSystemProblem):
    def __init__(self, **kwargs):
        super(RectanglePackingProblem, self).__init__(**kwargs)

    def f(self, x):
        raise NotImplementedError

    def is_feasible(self, x):
        raise NotImplementedError

    def get_empty_solution(self):
        raise NotImplementedError

    def get_arbitrary_solution(self):
        raise NotImplementedError

    def get_neighborhood(self, x):
        raise NotImplementedError

    def get_construction_values(self, x):
        raise NotImplementedError

    def construct(self, x, i):
        raise NotImplementedError
