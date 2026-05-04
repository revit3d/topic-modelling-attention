import jax.numpy as jnp
from jax import Array

from cartm.core import EPSILON
from cartm.metrics.metric_base import Metric


class PerplexityMetric(Metric):
    def __init__(self, tag: str | None = None):
        """
        Args:
            tag: metric's name to be displayed in logs.
        """
        if tag is None:
            tag = self.__class__.__name__
        super().__init__(tag=tag)

        self._num_words = 0
        self._likelihood = 0.0

    def partial_update(
        self,
        *,
        batch: Array,
        phi: Array,
        theta: Array,
        valid_mask: Array,
    ):
        self._num_words += valid_mask.sum().item()

        # p(w_i|C_i) = p(w_i|t)p(t|C_i) = \sum_t (phi_it * theta_it)
        p_wi = jnp.sum(theta * phi[batch], axis=1)

        # L = \sum_d \sum_w n_dw \log p(w|d) = \sum_i 1 * \log p(w_i|C_i)
        self._likelihood += jnp.sum(jnp.log(p_wi + EPSILON) * valid_mask)

    def _flush(self) -> float:
        # perplexity = exp{-L / I}
        perplexity = jnp.exp(-self._likelihood / self._num_words).item()
        self._num_words = 0
        self._likelihood = 0.0
        return perplexity
