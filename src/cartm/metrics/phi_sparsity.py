import jax.numpy as jnp
from jax import Array

from .metric_base import Metric


class SparsityMetric(Metric):
    def __init__(self, tag: str = None, eps: float = 1e-12):
        """
        Args:
            eps: value counts as non-zero if value > eps
        """
        if tag is None:
            tag = self.__class__.__name__
        super().__init__(tag=tag)

        self.eps = eps

    def _call_impl(self, phi_it: Array, phi_wt: Array, theta: Array) -> float:
        num_zeros = jnp.sum(jnp.abs(phi_wt) < self.eps)
        num_elems = jnp.prod(phi_wt.shape)
        return num_zeros / num_elems
