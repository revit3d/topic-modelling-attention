from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from cartm.metrics.metric_base import Metric


class AttentivePerplexityMetric(Metric):
    """
    Perplexity for Attentive ARTM with phi[w, t] = p(t|w).

    It reconstructs p(w|t) from p(t|w) using Bayes rule and empirical p(w)
    estimated on the evaluation sample (or passed explicitly via kwargs).
    """

    def __init__(self, vocab_size: int, tag: str = None, eps: float = 1e-12):
        if tag is None:
            tag = self.__class__.__name__
        super().__init__(tag=tag)

        self.vocab_size = vocab_size
        self._eps = eps

    @partial(jax.jit, static_argnums=0)
    def _call_impl(
        self,
        phi_it: Array,
        phi_wt: Array,
        theta: Array,
        **kwargs,
    ) -> float:
        """
        Args:
            phi_it: (I, T), not used directly here.
            phi_wt: (W, T), where phi_wt[w, t] = p(t|w)
            theta:  (I, T), where theta[i, t] = p(t|C_i)

        Kwargs:
            batch: (I,) token ids
            word_probs: optional (W,) precomputed p(w)

        Returns:
            Perplexity.
        """
        batch = kwargs.get("batch", None)
        word_probs = kwargs.get("word_probs", None)

        if batch is None and word_probs is None:
            raise ValueError(
                "AttentivePerplexityMetric requires either `batch` or `word_probs`."
            )

        num_words = theta.shape[0]

        if word_probs is None:
            n_w = jnp.bincount(batch, length=self.vocab_size)
            total = jnp.maximum(jnp.sum(n_w), 1)
            p_w = n_w / total  # (W,)
        else:
            p_w = word_probs

        # p(t) = sum_w p(t|w) p(w)
        p_t = jnp.sum(phi_wt * p_w[:, None], axis=0)  # (T,)

        # reconstruct p(w|t) = p(t|w) p(w) / p(t)
        p_w_given_t = phi_wt * p_w[:, None] / (p_t[None, :] + self._eps)  # (W, T)

        if batch is None:
            raise ValueError(
                "AttentivePerplexityMetric needs `batch` to evaluate token probabilities."
            )

        token_pw_t = p_w_given_t[batch]  # (I, T)

        # p(w_i | C_i) = sum_t p(w_i|t) p(t|C_i)
        p_wi = jnp.sum(token_pw_t * theta, axis=1)

        log_likelihood = jnp.sum(jnp.log(p_wi + self._eps))
        perplexity = jnp.exp(-log_likelihood / num_words)
        return perplexity
