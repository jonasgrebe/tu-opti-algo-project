from algos.base import BaseAlgorithm


class GreeyAlgorihm(BaseAlgorithm):

    def __init__(self, **kwargs):
        super(GreeyAlgorihm, self).__init__(**kwargs)

        self.current_construction = problem.get_empty_instance()
