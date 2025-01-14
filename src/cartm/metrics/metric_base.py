from abc import ABC, abstractmethod
from functools import wraps


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
        self._hist = []

    @property
    def tag(self):
        """Metric's name to be displayed in logs"""
        return self._tag

    @property
    def history(self):
        """Metric's calculations history"""
        return self._hist

    def reset_history(self):
        """Resets history of metric's calculations"""
        self._hist = []

    @abstractmethod
    def _call_impl(self, phi, theta):
        pass

    def __call__(self, phi, theta) -> float:
        value = self._call_impl(phi, theta)
        self._hist.append(value)
        return value
