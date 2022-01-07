from problems.neighborhood import NeighborhoodProblem
from gui import BaseGUI

import numpy as np
import time


MINIMUM_IMPROVEMENT = 0.01


def local_search(problem: NeighborhoodProblem, gui: BaseGUI = None):
    t = time.time()

    # Step 1: Start with an arbitrary feasible solution
    current_solution = gui.get_current_solution() if gui is not None else problem.get_arbitrary_solution()

    # Step 2: While there is a better solution nearby, go to that solution
    step = 0

    problem.reset_relaxation()
    while True:
        objective_value = problem.objective_function(current_solution)
        heuristic_value = problem.heuristic(current_solution)
        print("\rStep: %d - Objective value: %.2f - Heuristic value: %.2f"
              % (step, objective_value, heuristic_value), end="")

        # Degree of freedom: may choose from one of these
        # next_solution = get_best_neighbor(problem, current_solution)
        next_solution = get_next_better_neighbor(problem, current_solution)

        if next_solution is None:
            break

        current_solution = next_solution
        step += 1

        if gui is not None:
            if gui.is_searching:
                gui.set_and_animate_solution(current_solution)
                # gui.set_current_solution(current_solution)
            else:
                break

        # potentiall de-relax the problem
        problem.update_relaxation(step)

    # tell gui that search is over
    if gui is not None:
        gui.stop_search()

    print("\nLocal search took %.3f s" % (time.time() - t))

    # Step 3: deliver final solution (local optimum)
    return current_solution


def get_best_neighbor(problem, solution):
    value = problem.heuristic(solution)

    neighborhood = problem.get_neighborhood(solution)

    # print("neighborhood size:", len(neighborhood))

    neighborhood_values = [problem.heuristic(neighbor_solution) for neighbor_solution in neighborhood]

    if problem.is_max:
        best_neighbor_idx = np.argmax(neighborhood_values)
        is_significantly_better = neighborhood_values[best_neighbor_idx] > value + MINIMUM_IMPROVEMENT
    else:
        best_neighbor_idx = np.argmin(neighborhood_values)
        is_significantly_better = neighborhood_values[best_neighbor_idx] < value - MINIMUM_IMPROVEMENT

    if not (is_significantly_better or problem.is_relaxation_active()) :
        return None

    return neighborhood[best_neighbor_idx]


def get_next_better_neighbor(problem: NeighborhoodProblem, solution):
    value = problem.heuristic(solution)

    for neighbors in problem.get_next_neighbors(solution):
        if not neighbors:
            continue

        neighbors_values = [problem.heuristic(neighbor) for neighbor in neighbors]

        if problem.is_max:
            best_neighbor_idx = np.argmax(neighbors_values)
            is_significantly_better = neighbors_values[best_neighbor_idx] > value + MINIMUM_IMPROVEMENT
        else:
            best_neighbor_idx = np.argmin(neighbors_values)
            is_significantly_better = neighbors_values[best_neighbor_idx] < value - MINIMUM_IMPROVEMENT

        if is_significantly_better or problem.is_relaxation_active():
            return neighbors[best_neighbor_idx]

        # for neighbor in neighbors:
        #     if problem.is_max and problem.h(neighbor) > value:
        #         return neighbor
        #     elif not problem.is_max and problem.h(neighbor) < value:
        #         return neighbor
