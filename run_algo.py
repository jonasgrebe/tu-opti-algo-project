from algos.local import local_search
from algos.greedy import greedy_search, greedy_search_fast
from problems.rectangle_packing.problem import RPPGeometryBased
from problems.rectangle_packing.problem import RPPGreedy, RPPGreedyFast


# For performance test
problem = RPPGeometryBased(8, 1000, 1, 7, 1, 7)
solution = local_search(problem)

print("objective value:", problem.objective_function(solution))
