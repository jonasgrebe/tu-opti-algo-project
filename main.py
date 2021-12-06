from algos import local_search
from problems.examples.grid import TwoDGridProblem
from problems.examples.rectangle_packing import RectanglePackingProblem
from gui.rectangle_packing import RectanglePackingGUI


# f = lambda x: x[0] ** 2 + x[1]**2
# grid_problem = TwoDGridProblem(f=f, is_max=False)
# print(local_search(grid_problem))

problem = RectanglePackingProblem(box_length=8, num_rects=16, w_min=1, w_max=7, h_min=1, h_max=7)

gui = RectanglePackingGUI()
gui.problem = problem
init_sol = problem.get_arbitrary_solution()
gui.set_current_solution(init_sol)
print(local_search(problem, gui))

# while True:
#     gui.set_current_solution(init_sol)
