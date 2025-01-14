from abc import ABC, abstractmethod


class Metric(ABC):
    """
    Base class for all metrics. To implement a custom metric,\\
    you have to override `__init__` and `__call__` methods.
    """

    def __init__(self, tag: str):
        """
        Args:
            tag: metric's name to be displayed in logs
        """
        self._tag = tag

    @property
    def tag(self):
        """Metric's name to be displayed in logs"""
        return self._tag

    @abstractmethod
    def __call__(self, phi, theta) -> float:
        pass
