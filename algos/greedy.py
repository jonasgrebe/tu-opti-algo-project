from problems.construction import IndependenceSystemProblem


def greedy_search(problem: IndependenceSystemProblem):
    # Step 1: get and sort elements
    elements = problem.get_elements()
    elements = sorted(elements, key=problem.c, reverse=problem.is_max)

    # Step 2: get empty set
    x = set()

    # Step 3: for each e in elements
    for e in elements:
        if problem.is_independent(x.union({e})):
            x.add(e)

    return x
