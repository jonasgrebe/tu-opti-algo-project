from algos.local import local_search
from problems.rectangle_packing.problem import RectanglePackingProblemGeometryBased


# For performance test
problem = RectanglePackingProblemGeometryBased(8, 1000, 1, 7, 1, 7)
solution = local_search(problem)
print("objective value:", problem.objective_function(solution))