import pytest

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_almost_equal

import cartm.metrics as mtc


vocab_size = 20
n_words = 100
n_documents = 10
n_topics = 12
top_k = 8
eps = 1e-12
seed = 42


def calc_perplexity_primitive(phi_it: jax.Array, theta: jax.Array):
    p_i = np.zeros((n_words, ))
    for i in range(n_words):
        for t in range(n_topics):
            p_i[i] += phi_it[i][t] * theta[i][t]
        p_i[i] = np.log(p_i[i] + eps)

    likelihood = 0.0
    for i in range(n_words):
        likelihood += p_i[i]
    return np.exp(-likelihood / n_words)


def calc_sparsity_primitive(phi_wt: jax.Array):
    zero_cnt, cnt = 0, 0
    for w in range(vocab_size):
        for t in range(n_topics):
            cnt += 1
            if np.isclose(phi_wt[w][t], 0):
                zero_cnt += 1
    return zero_cnt / cnt


def calc_jaccard_primitive(phi_wt: jax.Array, t1: int, t2: int):
    topk1 = jnp.argsort(phi_wt[:, t1], descending=True)[:top_k]
    topk2 = jnp.argsort(phi_wt[:, t2], descending=True)[:top_k]

    s1, s2 = set(topk1.tolist()), set(topk2.tolist())
    intersection = s1.intersection(s2)
    union = s1.union(s2)
    return 1 - len(intersection) / len(union)


def calc_cosine_primitive(phi_wt: jax.Array, t1: int, t2: int):
    numerator = 0.0
    for w in range(vocab_size):
        numerator += phi_wt[w, t1] * phi_wt[w, t2]

    denominator_t1 = 0.0
    for w in range(vocab_size):
        denominator_t1 += phi_wt[w, t1]**2
    denominator_t1 = denominator_t1**0.5

    denominator_t2 = 0.0
    for w in range(vocab_size):
        denominator_t2 += phi_wt[w, t2]**2
    denominator_t2 = denominator_t2**0.5

    denominator = denominator_t1 * denominator_t2
    return 1 - numerator / denominator


def calc_hellinger_primitive(phi_wt: jax.Array, t1: int, t2: int):
    dist = 0.0
    for w in range(vocab_size):
        dist += (phi_wt[w, t1]**0.5 - phi_wt[w, t2]**0.5)**2
    return (dist / 2)**0.5


def calc_topic_variance_primitive(phi_wt: jax.Array, dist_metric: str):
    dist_matrix = np.zeros((n_topics, n_topics), dtype=float)
    dist_funcs = {
        'jaccard': calc_jaccard_primitive,
        'cosine': calc_cosine_primitive,
        'hellinger': calc_hellinger_primitive,
    }

    for t1 in range(n_topics):
        for t2 in range(n_topics):
            dist_func = dist_funcs[dist_metric]
            dist_matrix[t1][t2] = dist_func(phi_wt, t1, t2)

    closest_topic = np.full((n_topics, ), fill_value=-1, dtype=int)
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


def calc_coherence_primitive(bow: jax.Array, phi_wt: jax.Array):
    word_counts = np.zeros((vocab_size, ), dtype=int)
    for d in range(n_documents):
        for w in range(vocab_size):
            word_counts[w] += int(bow[d][w] != 0)

    pair_counts = np.zeros((vocab_size, vocab_size), dtype=int)
    for d in range(n_documents):
        bow_d = bow[d]
        for w1 in range(vocab_size):
            for w2 in range(w1 + 1, vocab_size):
                pair_counts[w1][w2] += (bow_d[w1] != 0) * (bow_d[w2] != 0)

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
                pmi = np.log(p_w1_w2 / (p_w1 * p_w2) + eps)
                topic_coherence += pmi
                n_pairs += 1

        assert n_pairs == top_k * (top_k - 1) // 2
        topic_coherence = topic_coherence / n_pairs
        coherence.append(topic_coherence.item())

    return np.mean(coherence)


@pytest.fixture
def data():
    key = jax.random.key(seed)
    return jax.random.randint(
        key=key,
        shape=(n_words, ),
        minval=0,
        maxval=vocab_size,
    )


@pytest.fixture
def doc_bounds():
    key = jax.random.key(seed)
    return jnp.concatenate([
        jnp.array([0]),
        jax.random.randint(
            key=key,
            shape=(n_documents - 1, ),
            minval=1,
            maxval=n_words - 1
        ),
        jnp.array([n_words]),
    ]).sort()


@pytest.fixture
def bag_of_words(data, doc_bounds):
    bow = []
    for d in range(n_documents):
        doc_start_idx = doc_bounds[d]
        doc_end_idx = doc_bounds[d + 1]
        doc_tokens = data[doc_start_idx:doc_end_idx]
        bow.append(jnp.add.at(jnp.zeros(vocab_size), doc_tokens, 1, inplace=False))
    bow = jnp.array(bow)

    assert len(bow) == n_documents
    assert bow.sum() == len(data)
    return bow


@pytest.fixture
def phi_wt():
    key = jax.random.key(seed)
    phi = jax.random.uniform(key=key, shape=(vocab_size, n_topics))
    phi = phi / phi.sum(axis=0, keepdims=True)
    return phi


@pytest.fixture
def phi_it(data, phi_wt):
    return jnp.take_along_axis(
        phi_wt,
        indices=data[:, None],
        axis=0,
    )


@pytest.fixture
def theta():
    key = jax.random.key(seed)
    theta = jax.random.uniform(key=key, shape=(n_words, n_topics))
    theta = theta / theta.sum(axis=1, keepdims=True)
    return theta


def test_perplexity(phi_it, theta):
    perplexity_primitive = calc_perplexity_primitive(
        phi_it=phi_it,
        theta=theta,
    )
    perplexity_metric = mtc.PerplexityMetric()(
        phi_it=phi_it,
        phi_wt=None,
        theta=theta,
    )
    assert_almost_equal(perplexity_metric, perplexity_primitive, decimal=5)


@pytest.mark.parametrize('zero_threshold', [0.1, 0.3, 0.6, 0.8])
def test_sparsity(zero_threshold, phi_wt):
    thresh_mask = (phi_wt < zero_threshold)
    phi_wt_thresh = phi_wt.at[thresh_mask].set(0.0)
    sparsity_primitive = calc_sparsity_primitive(phi_wt=phi_wt_thresh)
    sparsity_metric = mtc.SparsityMetric()(
        phi_it=None,
        phi_wt=phi_wt_thresh,
        theta=None,
    )
    assert_almost_equal(sparsity_metric, sparsity_primitive)


@pytest.mark.parametrize('distance_metric', ['jaccard', 'cosine', 'hellinger'])
def test_topic_variance(distance_metric, phi_wt):
    topic_variance_primitive = calc_topic_variance_primitive(
        phi_wt=phi_wt,
        dist_metric=distance_metric,
    )
    topic_variance_metric = mtc.TopicVarianceMetric(
        top_k=top_k if distance_metric == 'jaccard' else None,
        distance_metric=distance_metric,
    )(phi_it=None, phi_wt=phi_wt, theta=None)
    assert_almost_equal(topic_variance_metric, topic_variance_primitive)


def test_coherence(bag_of_words, phi_wt):
    coherence_primitive = calc_coherence_primitive(bow=bag_of_words, phi_wt=phi_wt)
    coherence_metric = mtc.CoherenceMetric(data=bag_of_words, top_k=top_k)(
        phi_it=None,
        phi_wt=phi_wt,
        theta=None,
    )
    assert_almost_equal(coherence_metric, coherence_primitive, decimal=5)
