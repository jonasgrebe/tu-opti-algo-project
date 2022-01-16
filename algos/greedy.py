from problems.independence import IndependenceProblem
from gui import BaseGUI

import numpy as np
import time


def greedy_search(problem: IndependenceProblem, gui: BaseGUI = None):  # TODO: rename to max_greedy_search
    t = time.time()

    # Step 1: Get empty solution
    partial_solution = problem.get_empty_solution()

    # Step 2: Generate the candidate elements
    elements = problem.get_elements(partial_solution)
    elements = sorted(elements, key=problem.costs)
    # elements = np.array(elements, dtype=object)

    step = 0
    while not partial_solution.is_complete():
        # e, elements = elements[0], elements[1:]
        e = elements.pop(0)

        step += 1
        print(f"\rStep: {step} / Remaining Elements: {len(elements)} - Current Element: {e}", end="")

        # Step 3: Check union for independence
        if problem.is_independent(partial_solution, e):
            # if independent, then add element to partial solution
            partial_solution.add_element(e)

            if len(elements) > 0:
                elements = problem.filter_elements(partial_solution, elements, e)
        else:
            # already break here, if search has paused/stopped
            if gui is not None and not gui.is_searching:
                break

            # if not independent, continue with next element
            continue

        # Step 4: Animation
        if gui is not None:
            if gui.is_searching:
                gui.set_and_animate_solution(partial_solution)
                gui.update_search_info({'num_remaining_elements': len(elements)})
            else:
                break

    # tell gui that search is over
    if gui is not None:
        gui.stop_search()

    print("\nGreedy search took %.3f s" % (time.time() - t))

    return partial_solution


def greedy_search_fast(problem: IndependenceProblem, gui: BaseGUI = None):  # TODO: rename to max_greedy_search
    t = time.time()

    # Step 1: Get empty solution
    partial_solution = problem.get_empty_solution()

    step = 0

    while not partial_solution.is_complete():

        partial_solution = problem.get_expansion(partial_solution)

        step += 1
        print(f"\rStep: {step}", end="")

        if gui is not None:
            if gui.is_searching:
                gui.set_and_animate_solution(partial_solution)
                gui.update_search_info({'num_remaining_elements': len(partial_solution.get_remaining_elements())})
            else:
                break

    # tell gui that search is over
    if gui is not None:
        gui.stop_search()

    print("\nGreedy search took %.3f s" % (time.time() - t))

    return partial_solution
