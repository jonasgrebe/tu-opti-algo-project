from algos.base import BaseAlgorithm


class GreedyAlgorithm(BaseAlgorithm):

    def __init__(self, **kwargs):
        super(GreedyAlgorithm, self).__init__(**kwargs)

        self.current_construction = problem.get_empty_instance()
