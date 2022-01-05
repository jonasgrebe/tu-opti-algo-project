from algos.local import local_search
from algos.greedy import greedy_search
from problems.rectangle_packing.problem import RectanglePackingProblemGeometryBased
from problems.rectangle_packing.problem import RectanglePackingProblemGreedyLargestAreaStrategy


# For performance test
problem = RectanglePackingProblemGreedyLargestAreaStrategy(8, 100, 1, 7, 1, 7)
solution = greedy_search(problem)
print("objective value:", problem.objective_function(solution))
