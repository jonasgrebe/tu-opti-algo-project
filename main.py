from gui.rectangle_packing import RectanglePackingGUI
from algos.local import local_search
from problems.examples.rectangle_packing import RectanglePackingProblem
import time

gui = RectanglePackingGUI()

# For performance test
# problem = RectanglePackingProblem(8, 100, 1, 8, 1, 8)
# init_sol = problem.get_arbitrary_solution()
# t = time.time()
# solution = local_search(init_sol, problem)
# print("time: %.3f s" % (time.time() - t))
# gui = RectanglePackingGUI()
# gui.problem = problem
# gui.set_current_solution(solution)
