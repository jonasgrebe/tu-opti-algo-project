from typing import Iterator
from abc import abstractmethod
from problems.optimization import OptProblem


class IndependenceSet:
    def add(self, element):
        raise NotImplementedError

    def remove(self, element):
        raise NotImplementedError


class IndependenceProblem(OptProblem):
    def __init__(self, **kwargs):
        super(IndependenceProblem, self).__init__(**kwargs)

    @abstractmethod
    def get_sorted_elements(self) -> Iterator:
        """Returns the elements of the ground set E sorted by costs."""
        raise NotImplementedError

    @abstractmethod
    def get_empty_independence_set(self) -> IndependenceSet:
        """Returns an empty set."""
        raise NotImplementedError

    @abstractmethod
    def is_independent(self, independence_set: IndependenceSet) -> bool:
        """Takes a set e and returns whether that set is independent.

        :param independence_set: subset of the ground set E
        :return: True iff independence_set is independent
        """
        raise NotImplementedError

    @abstractmethod
    def is_basis(self, independence_set: IndependenceSet):
        raise NotImplementedError

    @abstractmethod
    def costs(self, elements):
        """Returns the cost for a given element."""
        raise NotImplementedError
