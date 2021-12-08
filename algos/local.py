from problems.neighborhood import NeighborhoodProblem
from gui import BaseGUI

import numpy as np
import time


def local_search(init_solution, problem: NeighborhoodProblem, gui: BaseGUI = None):
    t = time.time()

    # Step 1: Start with an arbitrary feasible solution
    current_solution = init_solution

    # Step 2: While there is a better solution nearby, go to that solution
    while True:
        # Degree of freedom: may choose from one of these
        # next_solution = get_best_neighbor(problem, current_solution)

        # t = time.time()
        next_solution = get_next_better_neighbor(problem, current_solution)
        # print("getting next better neighbor took %.3f s" % (time.time() - t))

        if next_solution is None:
            break

        current_solution = next_solution

        if gui is not None:
            if gui.is_searching:
                # gui.set_and_animate_solution(current_solution)
                gui.set_current_solution(current_solution)
            else:
                break

    # tell gui that search is over
    if gui is not None:
        gui.stop_search()

    print("Local search took %.3f s" % (time.time() - t))

    # Step 3: deliver final solution (local optimum)
    return current_solution


def get_best_neighbor(problem, solution):
    value = problem.h(solution)

    t = time.time()
    neighborhood = problem.get_neighborhood(solution)
    print("getting all neighbors took %.3f s" % (time.time() - t))

    print("neighborhood size:", len(neighborhood))

    t = time.time()
    neighborhood_values = [problem.h(neighbor_solution) for neighbor_solution in neighborhood]
    print("evaluating all neighbors took %.3f s" % (time.time() - t))

    if problem.is_max:
        best_neighbor_idx = np.argmax(neighborhood_values)
        is_better = neighborhood_values[best_neighbor_idx] > value
    else:
        best_neighbor_idx = np.argmin(neighborhood_values)
        is_better = neighborhood_values[best_neighbor_idx] < value

    if not is_better:
        return None

    return neighborhood[best_neighbor_idx]


def get_next_better_neighbor(problem: NeighborhoodProblem, solution):
    value = problem.h(solution)

    for neighbors in problem.get_next_neighbors(solution):
        if not neighbors:
            continue

        neighbors_values = [problem.h(neighbor) for neighbor in neighbors]

        if problem.is_max:
            best_neighbor_idx = np.argmax(neighbors_values)
            is_better = neighbors_values[best_neighbor_idx] > value
        else:
            best_neighbor_idx = np.argmin(neighbors_values)
            is_better = neighbors_values[best_neighbor_idx] < value

        if is_better:
            return neighbors[best_neighbor_idx]

        # for neighbor in neighbors:
        #     if problem.is_max and problem.h(neighbor) > value:
        #         return neighbor
        #     elif not problem.is_max and problem.h(neighbor) < value:
        #         return neighbor
