from algos import local_search
from problems.examples.grid import TwoDGridProblem
from problems.examples.rectangle_packing import RectanglePackingProblem
from gui.rectangle_packing import RectanglePackingGUI


# f = lambda x: x[0] ** 2 + x[1]**2
# grid_problem = TwoDGridProblem(f=f, is_max=False)
# print(local_search(grid_problem))

problem = RectanglePackingProblem(6, 12, 1, 5, 1, 5)
gui = RectanglePackingGUI()
gui.problem = problem
init_sol = problem.get_arbitrary_solution()
print(local_search(problem, gui))

# while True:
#     gui.set_current_solution(init_sol)
