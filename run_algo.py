from algos.local import local_search
from algos.greedy import greedy_search, greedy_search_fast
from problems.rectangle_packing.problem import RectanglePackingProblemGeometryBased
from problems.rectangle_packing.problem import RectanglePackingProblemGreedy, RectanglePackingProblemGreedyFast


# For performance test
problem = RectanglePackingProblemGeometryBased(8, 1000, 1, 7, 1, 7)
solution = local_search(problem)

print("objective value:", problem.objective_function(solution))
