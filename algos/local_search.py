from problems.neighborhood import NeighborhoodProblem
import numpy as np


def local_search(problem: NeighborhoodProblem):
    # Step 1: Start with an arbitrary feasible solution
    current_solution = problem.get_arbitrary_solution()

    assert problem.is_feasible(current_solution)

    # Step 2: While there is a better solution nearby, go to that solution
    while True:
        neighborhood = problem.get_neighborhood(current_solution)

        neighborhood_values = [problem.f(neighbor_solution) for neighbor_solution in neighborhood]

        # Degree of freedom: use best neighbor
        if problem.is_max:
            best_neighbor_idx = np.argmax(neighborhood_values)
            is_better = neighborhood_values[best_neighbor_idx] > problem.f(current_solution)
        else:
            best_neighbor_idx = np.argmin(neighborhood_values)
            is_better = neighborhood_values[best_neighbor_idx] < problem.f(current_solution)

        if not is_better:
            break

        current_solution = neighborhood[best_neighbor_idx]
        print("current_solution:", current_solution)

    # Step 3: deliver final solution (local optimum)
    return current_solution
