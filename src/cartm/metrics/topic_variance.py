from typing import Literal

import jax.numpy as jnp
from jax import Array

from .metric_base import Metric


class TopicVarianceMetric(Metric):
    def __init__(
            self,
            top_k: int,
            distance_metric: Literal['jaccard', 'cosine', 'hellinger'] = 'jaccard',
            tag: str = None,
            eps: float = 1e-12,
    ):
        """
        Args:
            data: bag of words fitted on the corpus
            top_k: number of top words to calculate distance metric
            distance_metric: metric for calculating topics' similarity
            tag: metric's name to be displayed in logs
        """
        if tag is None:
            tag = self.__class__.__name__
        super().__init__(tag=tag)

        self.distance_metric = distance_metric
        self.top_k = top_k
        if distance_metric == 'jaccard':
            self._dist_func = self._jaccard_distance
        elif distance_metric == 'cosine':
            self._dist_func = self._cosine_distance
        elif distance_metric == 'hellinger':
            self._dist_func = self._hellinger_distance
        else:
            raise NotImplementedError()
        self._eps = eps

    def _call_impl(self, phi_it: Array, phi_wt: Array, theta: Array = None) -> float:
        top_words_per_topic = jnp.argpartition(phi_wt, -self.top_k, axis=0)[-self.top_k:]  # (W_k, T)
        dist_matrix = self._compute_distance_matrix(top_words_per_topic)  # (T, T)
        dist_matrix += jnp.diag(jnp.full(len(dist_matrix), jnp.inf))  # add inf to diagonal
        min_dist_per_topic = dist_matrix.min(axis=0)
        return jnp.mean(min_dist_per_topic)

    def _jaccard_distance(self, v1: Array, v2: Array) -> float:
        s1, s2 = set(v1), set(v2)
        intersection = s1.intersection(s2)
        union = s1.union(s2)
        assert union != 0
        return 1 - intersection / union

    def _cosine_distance(self, v1: Array, v2: Array) -> float:
        scalar_prod = jnp.sum(v1 * v2)
        norm = (jnp.sum(v1**2)**0.5) * (jnp.sum(v2**2)**0.5)
        assert norm != 0
        return 1 - scalar_prod / norm

    def _hellinger_distance(self, v1: Array, v2: Array) -> float:
        dist2 = jnp.sum((v1**0.5 - v2**0.5)**2) / 2
        return dist2**0.5

    def _compute_distance_matrix(self, top_words_per_topic: Array) -> Array:
        n_topics = len(top_words_per_topic)
        distance_matrix = jnp.zeros((n_topics, n_topics))
        for t1 in range(n_topics):
            for t2 in range(t1 + 1, n_topics):
                dist = self._dist_func(top_words_per_topic[t1], top_words_per_topic[t2])
                distance_matrix = distance_matrix.at[t1, t2].set(dist)
                distance_matrix = distance_matrix.at[t2, t1].set(dist)
        return distance_matrix
