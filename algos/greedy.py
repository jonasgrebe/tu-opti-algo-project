from problems.independence import IndependenceProblem
from gui import BaseGUI

import time


def greedy_search(problem: IndependenceProblem, gui: BaseGUI):
    """The BEST-IN-GREEDY algorithm."""
    # Step 1: Get elements sorted by costs
    element_iterator = problem.get_sorted_elements()

    # Step 2: Initialize the empty set (F in the slides)
    independent_set = problem.get_empty_independence_set()

    # Step 3: As long as the set remains independent, put elements into it
    for element in element_iterator:
        independent_set.add(element)
        if not problem.is_independent(independent_set):
            independent_set.remove(element)
        else:
            if gui.is_searching:
                gui.set_and_animate_independence_set(independent_set)
        if not gui.is_searching:
            break

    # Tell gui that search is over
    gui.stop_search()

    return independent_set
