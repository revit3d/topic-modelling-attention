from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from .metric_base import Metric


class SparsityMetric(Metric):
    def __init__(self, tag: str = None, eps: float = 1e-12):
        """
        Args:
            tag: metric's name to be displayed in logs.
            eps: value counts as non-zero if value >= eps.
        """
        if tag is None:
            tag = self.__class__.__name__
        super().__init__(tag=tag)

        self.eps = eps

    @partial(jax.jit, static_argnums=0)
    def _call_impl(self, phi_it: Array, phi_wt: Array, theta: Array) -> float:
        num_zeros = jnp.sum(jnp.abs(phi_wt) < self.eps)
        num_elems = phi_wt.shape[0] * phi_wt.shape[1]
        return num_zeros / num_elems
