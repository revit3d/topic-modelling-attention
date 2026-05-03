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
            tag: metric's name to be displayed in logs.
        """
        self._tag = tag
        self._hist = []

    @property
    def tag(self) -> str:
        """Metric's name to be displayed in logs."""
        return self._tag

    @property
    def history(self) -> list:
        """Metric's calculations history."""
        return self._hist

    def reset_history(self):
        """Resets history of metric's calculations."""
        self._hist = []

    @abstractmethod
    def partial_update(
        self,
        *,
        batch: Array,
        phi: Array,
        theta: Array,
    ):
        """
        Args:
            batch: matrix of shape (I,), containing tokens.
            phi: matrix of shape (W, T), representing distribution p(w|t).
            theta: matrix of shape (I, T), representing distribution p(t|C_i).
        """
        pass

    @abstractmethod
    def _flush(self) -> float:
        pass

    def flush(self) -> float:
        value = self._flush()
        self._hist.append(value)
        return value
