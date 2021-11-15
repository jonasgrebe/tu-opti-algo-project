from abc import ABC as AbstractClass
from abc import abstractmethod


class BaseAlgorithm(AbstractClass):

    def __init__(self, problem):
        
        self.problem = problem
