from abc import ABC, abstractmethod

from jax import Array


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
    def tag(self) -> str:
        """Metric's name to be displayed in logs"""
        return self._tag

    @property
    def history(self) -> list:
        """Metric's calculations history"""
        return self._hist

    def reset_history(self):
        """Resets history of metric's calculations"""
        self._hist = []

    @abstractmethod
    def _call_impl(self, phi_it: Array, phi_wt: Array, theta: Array) -> float:
        pass

    def __call__(self, phi_it: Array, phi_wt: Array, theta: Array) -> float:
        """
        Args:
            phi_it: matrix of shape (I, T), representing distribution p(w_i|t)
            phi_wt: matrix of shape (W, T), representing distribution p(w|t)
            theta: matrix of shape (I, T), representing distribution p(t|C_i)
        """
        value = self._call_impl(phi_it=phi_it, phi_wt=phi_wt, theta=theta)
        self._hist.append(value)
        return value
