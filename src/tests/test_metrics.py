import pytest
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from numpy.testing import assert_almost_equal

import cartm.metrics as mtc
from cartm.core import EPSILON
from cartm.preprocessing import build_bow


def calc_perplexity_primitive(
    phi_it: jax.Array,
    theta: jax.Array,
    n_words: int,
    n_topics: int,
):
    p_i = np.zeros((n_words,))
    for i in range(n_words):
        for t in range(n_topics):
            p_i[i] += phi_it[i][t] * theta[i][t]
        p_i[i] = np.log(p_i[i] + EPSILON)

    likelihood = 0.0
    for i in range(n_words):
        likelihood += p_i[i]
    return np.exp(-likelihood / n_words)


def calc_sparsity_primitive(
    phi_wt: jax.Array,
    vocab_size: int,
    n_topics: int,
):
    zero_cnt, cnt = 0, 0
    for w in range(vocab_size):
        for t in range(n_topics):
            cnt += 1
            if np.isclose(phi_wt[w][t], 0):
                zero_cnt += 1
    return zero_cnt / cnt


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


def calc_coherence_primitive(
    bow: sp.csr_matrix,
    phi_wt: jax.Array,
    vocab_size: int,
    n_documents: int,
    n_topics: int,
    top_k: int,
):
    word_counts = np.zeros((vocab_size,), dtype=np.int32)
    for d in range(n_documents):
        for w in range(vocab_size):
            word_counts[w] += int(bow[d, w] != 0)

    pair_counts = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    for d in range(n_documents):
        for w1 in range(vocab_size):
            for w2 in range(w1 + 1, vocab_size):
                pair_counts[w1][w2] += (bow[d, w1] != 0) * (bow[d, w2] != 0)

    coherence = []
    for t in range(n_topics):
        topk_indices = jnp.argsort(phi_wt[:, t], descending=True)[:top_k]
        topk_indices = topk_indices.sort()

        topic_coherence = 0.0
        n_pairs = 0
        for i, w1 in enumerate(topk_indices):
            for w2 in topk_indices[i + 1:]:
                assert word_counts[w1] != 0 and word_counts[w2] != 0
                p_w1_w2 = pair_counts[w1][w2] / n_documents
                p_w1 = word_counts[w1] / n_documents
                p_w2 = word_counts[w2] / n_documents
                if pair_counts[w1][w2] == 0:
                    npmi = -1.0
                else:
                    pmi = np.log(p_w1_w2 / (p_w1 * p_w2) + EPSILON)
                    npmi = pmi / -(np.log(p_w1_w2) + EPSILON)
                topic_coherence += npmi
                n_pairs += 1

        assert n_pairs == top_k * (top_k - 1) // 2
        topic_coherence = topic_coherence / n_pairs
        coherence.append(topic_coherence.item())

    return np.mean(coherence)


@pytest.fixture
def theta(config):
    key = jax.random.key(config.seed)
    theta = jax.random.uniform(key=key, shape=(config.n_words, config.n_topics))
    theta = theta / theta.sum(axis=1, keepdims=True)
    return theta


def test_perplexity(data, phi, theta, config):
    phi_it = phi[data]
    perplexity_primitive = calc_perplexity_primitive(
        phi_it=phi_it,
        theta=theta,
        n_topics=config.n_topics,
        n_words=config.n_words,
    )
    perplexity_metric = mtc.PerplexityMetric()(
        phi_it=phi_it,
        phi_wt=None,
        theta=theta,
    )
    assert_almost_equal(perplexity_metric, perplexity_primitive, decimal=5)


@pytest.mark.parametrize("zero_threshold", [0.1, 0.3, 0.6, 0.8])
def test_sparsity(zero_threshold, phi, config):
    thresh_mask = phi < zero_threshold
    phi_wt_thresh = phi.at[thresh_mask].set(0.0)
    sparsity_primitive = calc_sparsity_primitive(
        phi_wt=phi_wt_thresh,
        vocab_size=config.vocab_size,
        n_topics=config.n_topics,
    )
    sparsity_metric = mtc.SparsityMetric()(
        phi_it=None,
        phi_wt=phi_wt_thresh,
        theta=None,
    )
    assert_almost_equal(sparsity_metric, sparsity_primitive)


@pytest.mark.parametrize("distance_metric", ["jaccard", "cosine", "hellinger"])
def test_topic_variance(distance_metric, phi, config):
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
    )(phi_it=None, phi_wt=phi, theta=None)
    assert_almost_equal(topic_variance_metric, topic_variance_primitive)


def test_coherence(data, doc_bounds, phi, config):
    top_k = 8
    bow = build_bow(
        tokenized_data=data,
        document_bounds=doc_bounds,
        vocab_size=config.vocab_size,
    )
    coherence_primitive = calc_coherence_primitive(
        bow=bow,
        phi_wt=phi,
        vocab_size=config.vocab_size,
        n_documents=config.n_documents,
        n_topics=config.n_topics,
        top_k=top_k,
    )
    coherence_metric = mtc.NPMICoherenceMetric(bow=bow, top_k=top_k)(
        phi_it=None,
        phi_wt=phi,
        theta=None,
    )
    assert_almost_equal(coherence_metric, coherence_primitive, decimal=5)
