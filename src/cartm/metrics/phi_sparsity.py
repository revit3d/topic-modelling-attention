import jax.numpy as jnp
from jax import Array

from cartm.metrics.metric_base import Metric


class SparsityMetric(Metric):
    def __init__(self, tag: str | None = None, eps: float = 1e-12):
        """
        Args:
            tag: metric's name to be displayed in logs.
            eps: value counts as non-zero if value >= eps.
        """
        if tag is None:
            tag = self.__class__.__name__
        super().__init__(tag=tag)

        self.eps = eps

        self._num_zeros = 0
        self._num_elems = 0

    def partial_update(
        self,
        *,
        batch: Array,
        phi: Array,
        theta: Array,
    ):
        self._num_zeros = jnp.sum(jnp.abs(phi) < self.eps).item()
        self._num_elems = phi.shape[0] * phi.shape[1]

    def _flush(self) -> float:
        return self._num_zeros / self._num_elems
