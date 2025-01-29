import jax.numpy as jnp
from jax import Array

from .metric_base import Metric


class PerplexityMetric(Metric):
    def __init__(self, tag: str = None, eps: float = 1e-12):
        """
        Args:
            data: tokenized corpus
            tag: metric's name to be displayed in logs
        """
        if tag is None:
            tag = self.__class__.__name__
        super().__init__(tag=tag)

        self._eps = eps

    def _call_impl(self, phi_it: Array, phi_wt: Array, theta: Array) -> float:
        num_words = len(phi_it)

        # p(w_i|C_i) = p(w_i|t)p(t|C_i) = \sum_t (phi_it * theta_it)
        p_wi = jnp.sum(theta * phi_it, axis=1)

        # L = \sum_d \sum_w n_dw \log p(w|d) = \sum_i 1 * \log p(w_i|C_i)
        likelihood = jnp.sum(jnp.log(p_wi + self._eps))

        # perplexity = exp{-L / I}
        perplexity = jnp.exp(-likelihood / num_words)
        return perplexity
