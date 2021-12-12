from gui.rectangle_packing import RectanglePackingGUI
from algos.local import local_search
from problems.examples.rectangle_packing import RectanglePackingProblem

gui = RectanglePackingGUI()

# For performance test
# problem = RectanglePackingProblem(8, 1000, 1, 8, 1, 8)
# init_sol = problem.get_arbitrary_solution()
# solution = local_search(init_sol, problem)
# print("objective value:", problem.objective_function(solution))
