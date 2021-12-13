from algos.local import local_search
from problems.rectangle_packing.problem import RectanglePackingProblemGeometryBased


# For performance test
problem = RectanglePackingProblemGeometryBased(8, 1000, 1, 8, 1, 8)
init_sol = problem.get_arbitrary_solution()
solution = local_search(init_sol, problem)
print("objective value:", problem.objective_function(solution))