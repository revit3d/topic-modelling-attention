from abc import ABC, abstractmethod

from jax import Array


class Regularization(ABC):
    """
    Base class for all regularizers. To implement a custom regularizer,\\
    you have to override `__init__` and `__call__` methods.

    Note, that the `__call__` method has to be jax-graddable.
    """

    def __init__(self, tag: str, tau: float):
        """
        Args:
            tag: regularizer's name to be displayed in logs.
            tau: value that controls regularization's strength.
        """
        self._tag = tag
        self._tau = tau

    @property
    def tag(self):
        """Regularizer's name to be displayed in logs."""
        return self._tag

    @property
    def multiplier(self):
        """Value that controls regularization's strength."""
        return self._tau

    @abstractmethod
    def _call_impl(self, phi_wt: Array) -> float:
        pass

    def __call__(self, phi_wt: Array) -> float:
        """
        Args:
            phi: matrix of shape (W, T), representing distribution p(w|t).
        """
        value = self._call_impl(phi_wt)
        return self.multiplier * value
