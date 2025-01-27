from abc import ABC, abstractmethod


class Regularization(ABC):
    """
    Base class for all regularizers. To implement a custom regularizer,\\
    you have to override `__init__` and `__call__` methods.

    Note, that the `__call__` method has to be jax-graddable
    """

    def __init__(self, tag: str):
        """
        Args:
            tag: regularizer's name to be displayed in logs
        """
        self._tag = tag

    @property
    def tag(self):
        """Regularizer's name to be displayed in logs"""
        return self._tag

    @abstractmethod
    def __call__(self):
        pass
