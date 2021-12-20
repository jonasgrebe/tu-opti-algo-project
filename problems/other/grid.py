from problems.neighborhood import NeighborhoodProblem
import random


class TwoDGridProblem(NeighborhoodProblem):
    def __init__(self, f, **kwargs):
        super(TwoDGridProblem, self).__init__(**kwargs)
        self.f = f

    def objective_function(self, sol):
        pass

    def is_feasible(self, sol):
        sol, y = sol
        return isinstance(sol, int) and isinstance(y, int)

    def get_neighborhood(self, sol):
        sol, y = sol
        return [(sol + 1, y + 1),
                (sol - 1, y + 1),
                (sol + 1, y - 1),
                (sol - 1, y - 1)]

    def get_arbitrary_solution(self):
        x = random.randint(-10, 10)
        y = random.randint(-10, 10)
        return x, y
