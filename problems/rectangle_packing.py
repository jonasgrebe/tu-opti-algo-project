from problems.neighborhood import NeighborhoodProblem
from problems.construction import ConstructionProblem


class RectanglePackingProblem(NeighborhoodProblem, ConstructionProblem):

    def __init__(self, **kwargs):
        super(RectanglePackingProblem, self).__init__(**kwargs)

    def is_feasible(self, x):
        """Returns whether x is a feasible solution or not

        :param x: the solution to check
        :return: True if x is feasible else False
        """
        raise NotImplementedError

    def get_neighborhood(self, x):
        """Returns the neighborhood of feasible instance x

        :param x: feasible solution x
        :return: List of (feasible) solutions
        """
        raise NotImplementedError

    def get_empty_instance(self):
        """Returns an instance to start with

        :return: "empty" partial solution
        """
        raise NotImplementedError

    def get_construction_options(self, x):
        """Takes an instance x and returns the values of all construction options

        :param x: partial solution
        :return: List of construction options
        """
        raise NotImplementedError

    def construct(self, x, i):
        """Extend current partial solution using the construction option i

        :param x: partial solution
        :param i: index of construction option
        :return: (partial) solution
        """
        raise NotImplementedError
