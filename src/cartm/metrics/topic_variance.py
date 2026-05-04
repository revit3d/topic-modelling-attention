from typing import Literal, Callable

import jax
import jax.numpy as jnp
from jax import Array

from cartm.core import EPSILON
from cartm.metrics.metric_base import Metric


class TopicVarianceMetric(Metric):
    def __init__(
        self,
        distance_metric: Literal["jaccard", "cosine", "hellinger"] = "jaccard",
        top_k: int | None = None,
        tag: str | None = None,
    ):
        """
        Args:
            distance_metric: metric for calculating topics' similarity.
            top_k: number of top words to calculate distance metric.
                Ignored if distance_metric != 'jaccard'.
            tag: metric's name to be displayed in logs.
        """
        if tag is None:
            tag = self.__class__.__name__
        super().__init__(tag=tag)

        self.distance_metric = distance_metric

        if distance_metric == "jaccard" and top_k is None:
            raise ValueError(
                "top_k parameter must be set with distance_metric = 'jaccard'"
            )
        self.top_k = top_k

        if distance_metric == "jaccard":
            self._dist_func = self._jaccard_distance
        elif distance_metric == "cosine":
            self._dist_func = self._cosine_distance
        elif distance_metric == "hellinger":
            self._dist_func = self._hellinger_distance
        else:
            raise NotImplementedError()

        self._last_phi = None

    def partial_update(
        self,
        *,
        batch: Array,
        phi: Array,
        theta: Array,
        valid_mask: Array,
    ):
        self._last_phi = phi

    def _flush(self) -> float:
        if self._last_phi is None:
            return 0.0

        # topic vectors can have different nature depending on the chosen metric
        if self.distance_metric == "jaccard":
            top_words_per_topic = jnp.argpartition(
                self._last_phi, -self.top_k, axis=0
            )  # (W, T)
            top_words_per_topic = top_words_per_topic[-self.top_k:].T  # (T, W_k)
            W, T = self._last_phi.shape
            topic_vectors = jnp.zeros((T, W), dtype=bool)  # (T, W)
            topic_vectors = topic_vectors.at[
                jnp.arange(T)[:, None], top_words_per_topic
            ].set(True)
        else:
            topic_vectors = self._last_phi.T  # (T, W)

        dist_matrix = self._compute_distance_matrix(
            topic_vectors, self._dist_func
        )  # (T, T)
        dist_matrix += jnp.diag(
            jnp.full(len(dist_matrix), jnp.inf)
        )  # add inf to diagonal
        min_dist_per_topic = dist_matrix.min(axis=0)

        self._last_phi = None
        return jnp.mean(min_dist_per_topic).item()

    @staticmethod
    @jax.jit
    def _jaccard_distance(v1: Array, v2: Array) -> Array:
        intersection = jnp.logical_and(v1, v2).sum()
        union = jnp.logical_or(v1, v2).sum()
        return 1 - intersection / union

    @staticmethod
    @jax.jit
    def _cosine_distance(v1: Array, v2: Array) -> Array:
        scalar_prod = jnp.sum(v1 * v2)
        norm = (jnp.sum(v1**2) ** 0.5) * (jnp.sum(v2**2) ** 0.5)
        return 1 - scalar_prod / (norm + EPSILON)

    @staticmethod
    @jax.jit
    def _hellinger_distance(v1: Array, v2: Array) -> Array:
        dist2 = jnp.sum((v1**0.5 - v2**0.5) ** 2) / 2
        return dist2**0.5

    @staticmethod
    @jax.jit(static_argnames=("dist_func"))
    def _compute_distance_matrix(topic_vectors: Array, dist_func: Callable) -> Array:
        def compute_distances_to_topic(t1, all_topics):
            return jax.vmap(lambda t2: dist_func(t1, t2))(all_topics)

        distance_matrix = jax.vmap(
            lambda t: compute_distances_to_topic(t, topic_vectors)
        )(
            topic_vectors
        )  # (T, T)

        return distance_matrix
