from problems.neighborhood import NeighborhoodProblem
from problems.construction import ConstructionProblem


class RectanglePackingProblem(NeighborhoodProblem, ConstructionProblem):
    def __init__(self, **kwargs):
        super(RectanglePackingProblem, self).__init__(**kwargs)

    def f(self, x):
        raise NotImplementedError

    def is_feasible(self, x):
        raise NotImplementedError

    def get_empty_solution(self):
        raise NotImplementedError

    def get_initial_solution(self):
        raise NotImplementedError

    def get_neighborhood(self, x):
        raise NotImplementedError

    def get_construction_values(self, x):
        raise NotImplementedError

    def construct(self, x, i):
        raise NotImplementedError
