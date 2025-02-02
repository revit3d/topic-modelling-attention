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
            data: bag of words with shape (D, W), fitted on the corpus
            top_k: number of top words to calculate pmi
            tag: metric's name to be displayed in logs
        """
        if tag is None:
            tag = self.__class__.__name__
        super().__init__(tag=tag)

        data = data.T  # (W, D)
        self.word_doc_indicator = data.astype(bool)
        self.word_co_occurence = self.word_doc_indicator @ self.word_doc_indicator.T  # (W, W)
        self.word_occurence = self.word_doc_indicator.sum(axis=1)  # (W, )
        self.top_k = top_k
        self._eps = eps

    def _call_impl(self, phi_it: Array, phi_wt: Array, theta: Array):
        top_words_per_topic = jnp.argpartition(
            phi_wt,
            kth=-self.top_k,
            axis=0,
        )[-self.top_k:]  # (W_k, T)
        n_topics = phi_wt.shape[1]
        coherence = []

        for t in range(n_topics):
            top_words = sorted(
                top_words_per_topic[t],
                key=lambda w: -phi_wt[w, t],
                reverse=True,
            )  # (W_k, )
            co_occurrences = self.word_co_occurence[top_words, top_words]  # (W_k, W_k)
            occurrences = self.word_occurence[top_words]  # (W_k, )
            pmi = jnp.log(
                co_occurrences / occurrences[:, None] / occurrences[None, :] + self._eps
            )  # (W_k, W_k)
            coherence.append(jnp.triu(pmi, k=1).mean())
        return jnp.mean(coherence)
