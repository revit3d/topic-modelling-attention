import numpy as np
import scipy.sparse as sp
import jax
import jax.numpy as jnp
from jax import Array

from cartm.metrics.metric_base import Metric


class NPMICoherenceMetric(Metric):
    def __init__(
        self,
        bow: sp.csr_matrix,
        top_k: int,
        tag: str | None = None,
    ):
        """
        Args:
            bow: bag of words fitted on the corpus in sparse format.
            top_k: number of top words to calculate pmi.
            tag: metric's name to be displayed in logs.
        """
        if tag is None:
            tag = self.__class__.__name__
        super().__init__(tag=tag)

        self.bow = bow.sign().astype(np.uint8).tocsc(copy=False)
        self.df = np.asarray(self.bow.getnnz(axis=0)).ravel().astype(np.float32)
        self.n_docs = self.bow.shape[0]
        self.top_k = top_k

    def _call_impl(self, phi_it: Array, phi_wt: Array, theta: Array, **kwargs) -> float:
        top_words = jnp.argpartition(phi_wt, -self.top_k, axis=0)[-self.top_k:]  # (k, T)
        top_words = np.asarray(jax.device_get(top_words.T))  # (T, k)

        selected, inv = np.unique(top_words, return_inverse=True)
        inv = inv.reshape(top_words.shape)

        X_sub = self.bow[:, selected]  # (D, m)
        cooc = (X_sub.T @ X_sub).toarray().astype(np.float32)  # (m, m)
        df = self.df[selected]

        p_i = df / self.n_docs
        p_ij = cooc / self.n_docs

        denom = p_i[:, None] * p_i[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            pmi = np.log(p_ij / denom)
            npmi = pmi / (-np.log(p_ij))

        # define zero co-occurrence as -1
        npmi = np.where(cooc > 0, npmi, -1.0)

        triu = np.triu_indices(self.top_k, k=1)
        topic_scores = []
        for topic_idx in inv:
            topic_npmi = npmi[np.ix_(topic_idx, topic_idx)]
            topic_scores.append(topic_npmi[triu].mean())

        return float(np.mean(topic_scores))
