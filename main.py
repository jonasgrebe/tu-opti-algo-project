from algos import local_search
from problems.examples.grid import TwoDGridProblem


f = lambda x: x[0] ** 2 + x[1]**2
grid_problem = TwoDGridProblem(f=f, is_max=False)
print(local_search(grid_problem))
