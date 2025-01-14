import jax.numpy as jnp
from scipy.sparse import csr_matrix

from .metric_base import Metric


class PerplexityMetric(Metric):
    def __init__(self, data: csr_matrix | jnp.ndarray, words_total: int = None, tag: str = None, eps: float = 1e-12):
        """
        Args:
            data: bag of words fitted on the corpus
        """
        if tag is None:
            tag = self.__class__.__name__
        super().__init__(tag=tag)

        self.bow_true = data
        if words_total is None:
            words_total = self.bow_true.sum()
        self.norm_cf = words_total
        if isinstance(self.bow_true, csr_matrix):
            self.bow_true = self.bow_true.toarray()
        self._eps = eps

    def _call_impl(self, phi, theta):
        likelihood = jnp.sum(self.bow_true * jnp.log(theta @ phi.T + self._eps))
        perplexity = jnp.exp(-likelihood / self.norm_cf)
        return perplexity
