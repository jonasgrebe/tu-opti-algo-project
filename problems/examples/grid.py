from problems.neighborhood import NeighborhoodProblem
import random


class TwoDGridProblem(NeighborhoodProblem):
    def __init__(self, f, **kwargs):
        super(TwoDGridProblem, self).__init__(**kwargs)
        self.f = f

    def f(self, x):
        pass

    def is_feasible(self, x):
        x, y = x
        return isinstance(x, int) and isinstance(y, int)

    def get_neighborhood(self, x):
        x, y = x
        return [(x + 1, y + 1),
                (x - 1, y + 1),
                (x + 1, y - 1),
                (x - 1, y - 1)]

    def get_arbitrary_solution(self):
        x = random.randint(-10, 10)
        y = random.randint(-10, 10)
        return x, y
