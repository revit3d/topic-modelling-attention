from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from .metric_base import Metric


class CoherenceMetric(Metric):
    def __init__(
            self,
            data: Array,
            top_k: int,
            tag: str = None,
            eps: float = 1e-12,
    ):
        """
        Args:
            data: bag of words with shape (D, W), fitted on the corpus.
            top_k: number of top words to calculate pmi.
            tag: metric's name to be displayed in logs.
            eps: parameter used for numerical stability.
        """
        if tag is None:
            tag = self.__class__.__name__
        super().__init__(tag=tag)

        self.word_doc_indicator = (data > 0).astype(int).T  # (W, D)
        self.word_occurence = self.word_doc_indicator.sum(axis=1)  # (W, )
        self.top_k = top_k
        self._eps = eps

    @partial(jax.jit, static_argnums=0)
    def _call_impl(self, phi_it: Array, phi_wt: Array, theta: Array):
        top_words_per_topic = jnp.argpartition(
            phi_wt,
            kth=-self.top_k,
            axis=0,
        )[-self.top_k:]  # (W_k, T)
        n_docs = self.word_doc_indicator.shape[1]

        top_words_per_topic = top_words_per_topic.T  # (T, W_k)
        top_word_indicator = self.word_doc_indicator[top_words_per_topic]  # (T, W_k, D)
        top_word_indicator_T = top_word_indicator.transpose(0, 2, 1)  # (T, D, W_k)

        co_occurrences = top_word_indicator @ top_word_indicator_T  # (T, W_k, W_k)
        co_occurrences /= n_docs  # normalize probabilities
        occurrences = self.word_occurence[top_words_per_topic]   # (T, W_k)
        occurrences /= n_docs  # normalize probabilities
        pmi = jnp.log(
            co_occurrences / occurrences[..., None] / occurrences[:, None, :] + self._eps
        )  # (T, W_k, W_k)

        unique_pmis = jnp.triu(pmi, k=1)  # (T, W_k, W_k)
        n_pairs = self.top_k * (self.top_k - 1) // 2
        coherence_per_topic = unique_pmis.sum(axis=(1, 2)) / n_pairs  # (T, )
        return jnp.mean(coherence_per_topic)
