from algos.base import BaseAlgorithm


class LocalSearchAlgorithm(BaseAlgorithm):

    def __init__(self, **kwargs):
        super(LocalSearchAlgorithm, self).__init__(**kwargs)

        self.current_solution = problem.get_initial_instance()
