from problems.construction import ConstructionProblem
from gui import BaseGUI

import numpy as np
import time

def greedy_search(problem: ConstructionProblem, gui: BaseGUI = None):  # TODO: rename to max_greedy_search
    t = time.time()

    # Step 2: get empty set
    partial_solution = problem.get_empty_solution()

    step = 0
    while not partial_solution.is_complete():

        # objective_value = problem.objective_function(partial_solution)
        #heuristic_value = problem.heuristic(partial_solution)
        #print("\rStep: %d - Objective value: %.2f - Heuristic value: %.2f"
        #      % (step, objective_value, heuristic_value), end="")

        partial_solution = get_best_next_element(problem, partial_solution)
        step += 1

        # Step 3: for each e in elements
        # if problem.is_independent(x.union({e})):
        #    x.add(e)

        if gui is not None:
            if gui.is_searching:
                gui.set_and_animate_solution(partial_solution)
                # gui.set_current_solution(partial_solution)
            else:
                break

    # tell gui that search is over
    if gui is not None:
        gui.stop_search()

    print("\nGreedy search took %.3f s" % (time.time() - t))

    return partial_solution


def get_best_next_element(problem, partial_solution):
    expansion = problem.get_expansion(partial_solution)
    expansion_values = [problem.heuristic(expanded_solution) for expanded_solution in expansion]
    best_expansion_idx = np.argmax(expansion_values) if problem.is_max else np.argmin(expansion_values)
    return expansion[best_expansion_idx]
