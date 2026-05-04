import pytest
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose

import cartm.metrics as mtc


def calc_jaccard_primitive(
    phi_wt: jax.Array,
    t1: int,
    t2: int,
    top_k: int,
):
    topk1 = jnp.argsort(phi_wt[:, t1], descending=True)[:top_k]
    topk2 = jnp.argsort(phi_wt[:, t2], descending=True)[:top_k]

    s1, s2 = set(topk1.tolist()), set(topk2.tolist())
    intersection = s1.intersection(s2)
    union = s1.union(s2)
    return 1 - len(intersection) / len(union)


def calc_cosine_primitive(
    phi_wt: jax.Array,
    t1: int,
    t2: int,
    vocab_size: int,
):
    numerator = 0.0
    for w in range(vocab_size):
        numerator += phi_wt[w, t1] * phi_wt[w, t2]

    denominator_t1 = 0.0
    for w in range(vocab_size):
        denominator_t1 += phi_wt[w, t1] ** 2
    denominator_t1 = denominator_t1**0.5

    denominator_t2 = 0.0
    for w in range(vocab_size):
        denominator_t2 += phi_wt[w, t2] ** 2
    denominator_t2 = denominator_t2**0.5

    denominator = denominator_t1 * denominator_t2
    return 1 - numerator / denominator


def calc_hellinger_primitive(
    phi_wt: jax.Array,
    t1: int,
    t2: int,
    vocab_size: int,
):
    dist = 0.0
    for w in range(vocab_size):
        dist += (phi_wt[w, t1] ** 0.5 - phi_wt[w, t2] ** 0.5) ** 2
    return (dist / 2) ** 0.5


def calc_topic_variance_primitive(
    phi_wt: jax.Array,
    dist_metric: str,
    vocab_size: int,
    n_topics: int,
    top_k: int,
):
    dist_matrix = np.zeros((n_topics, n_topics), dtype=np.float64)
    dist_funcs = {
        "jaccard": partial(calc_jaccard_primitive, top_k=top_k),
        "cosine": partial(calc_cosine_primitive, vocab_size=vocab_size),
        "hellinger": partial(calc_hellinger_primitive, vocab_size=vocab_size),
    }

    for t1 in range(n_topics):
        for t2 in range(n_topics):
            dist_func = dist_funcs[dist_metric]
            dist_matrix[t1][t2] = dist_func(phi_wt, t1, t2)

    closest_topic = np.full((n_topics,), fill_value=-1, dtype=np.int32)
    for t1 in range(n_topics):
        min_dist = np.inf
        for t2 in range(n_topics):
            if t1 != t2 and dist_matrix[t1][t2] < min_dist:
                min_dist = dist_matrix[t1][t2]
                closest_topic[t1] = t2
    assert ~np.any(closest_topic == -1)

    metric = 0.0
    for t1 in range(n_topics):
        t2 = closest_topic[t1]
        metric += dist_matrix[t1][t2]
    return metric / n_topics


@pytest.mark.parametrize("distance_metric", ["jaccard", "cosine", "hellinger"])
def test_topic_variance(distance_metric, data, phi, theta, config):
    top_k = 8
    topic_variance_primitive = calc_topic_variance_primitive(
        phi_wt=phi,
        dist_metric=distance_metric,
        vocab_size=config.vocab_size,
        n_topics=config.n_topics,
        top_k=top_k,
    )
    topic_variance_metric = mtc.TopicVarianceMetric(
        top_k=top_k if distance_metric == "jaccard" else None,
        distance_metric=distance_metric,
    )
    topic_variance_metric.partial_update(
        batch=data,
        phi=phi,
        theta=theta,
        valid_mask=jnp.ones_like(data, dtype=jnp.bool_),
    )
    topic_variance = topic_variance_metric.flush()
    assert_allclose(topic_variance, topic_variance_primitive, rtol=1e-5, atol=1e-6)


def test_topic_variance_rejects_jaccard_without_top_k():
    with pytest.raises(ValueError, match="top_k"):
        mtc.TopicVarianceMetric(distance_metric="jaccard", top_k=None)


def test_topic_variance_rejects_unknown_distance_metric():
    with pytest.raises(NotImplementedError):
        mtc.TopicVarianceMetric(distance_metric="unknown")  # type: ignore[arg-type]
