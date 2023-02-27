from abc import ABC, abstractmethod
import tensorflow as tf

class TensorflowRLModel(ABC):

    def __init__(self, name, description=None,):
        self.name = name
        self.description = description
