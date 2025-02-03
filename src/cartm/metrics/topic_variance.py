from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array

from .metric_base import Metric


class TopicVarianceMetric(Metric):
    def __init__(
            self,
            distance_metric: Literal['jaccard', 'cosine', 'hellinger'] = 'jaccard',
            top_k: int = None,
            tag: str = None,
            eps: float = 1e-12,
    ):
        """
        Args:
            distance_metric: metric for calculating topics' similarity.
            top_k: number of top words to calculate distance metric.
                Ignored if distance_metric != 'jaccard'.
            tag: metric's name to be displayed in logs.
            eps: parameter used for numerical stability.
        """
        if tag is None:
            tag = self.__class__.__name__
        super().__init__(tag=tag)

        self.distance_metric = distance_metric

        if distance_metric == 'jaccard' and top_k is None:
            raise ValueError("top_k parameter must be set with distance_metric = 'jaccard'")
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
        # topic vectors can have different nature depending on the chosen metric
        if self.distance_metric == 'jaccard':
            top_words_per_topic = jnp.argpartition(phi_wt, -self.top_k, axis=0)  # (W, T)
            top_words_per_topic = top_words_per_topic[-self.top_k:].T  # (T, W_k)
            W, T = phi_wt.shape
            topic_vectors = jnp.zeros((T, W), dtype=bool)  # (T, W)
            topic_vectors = topic_vectors.at[jnp.arange(T)[:, None], top_words_per_topic].set(True)
            print(topic_vectors)
        else:
            topic_vectors = phi_wt.T  # (T, W)

        dist_matrix = self._compute_distance_matrix(topic_vectors)  # (T, T)
        dist_matrix += jnp.diag(jnp.full(len(dist_matrix), jnp.inf))  # add inf to diagonal
        print(dist_matrix)
        min_dist_per_topic = dist_matrix.min(axis=0)
        return jnp.mean(min_dist_per_topic)

    def _jaccard_distance(self, v1: Array, v2: Array) -> float:
        intersection = jnp.logical_and(v1, v2).sum()
        union = jnp.logical_or(v1, v2).sum()
        return 1 - intersection / union

    def _cosine_distance(self, v1: Array, v2: Array) -> float:
        scalar_prod = jnp.sum(v1 * v2)
        norm = (jnp.sum(v1**2)**0.5) * (jnp.sum(v2**2)**0.5)
        return 1 - scalar_prod / (norm + self._eps)

    def _hellinger_distance(self, v1: Array, v2: Array) -> float:
        dist2 = jnp.sum((v1**0.5 - v2**0.5)**2) / 2
        return dist2**0.5

    def _compute_distance_matrix(self, topic_vectors: Array) -> Array:
        def compute_distances_to_topic(t1, all_topics):
            return jax.vmap(lambda t2: self._dist_func(t1, t2))(all_topics)

        distance_matrix = jax.vmap(
            lambda t: compute_distances_to_topic(t, topic_vectors)
        )(topic_vectors)  # (T, T)

        return distance_matrix
