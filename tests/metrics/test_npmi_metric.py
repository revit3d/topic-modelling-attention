import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from numpy.testing import assert_allclose

from cartm.core import EPSILON
import cartm.metrics as mtc
from cartm.preprocessing import CorpusDataLoader, build_bow_from_loader


def _docs_from_tokenized(data: jax.Array, document_bounds: jax.Array) -> list[str]:
    tokens = [int(x) for x in np.asarray(data)]
    bounds = [bool(x) for x in np.asarray(document_bounds)]

    docs: list[str] = []
    start = 0

    for i in range(1, len(tokens)):
        if bounds[i]:
            docs.append(" ".join(f"w{token}" for token in tokens[start:i]))
            start = i

    docs.append(" ".join(f"w{token}" for token in tokens[start:]))
    return docs


def _loader_from_tokenized(
    data: jax.Array,
    document_bounds: jax.Array,
    vocab_size: int,
) -> CorpusDataLoader:
    return CorpusDataLoader(
        _docs_from_tokenized(data, document_bounds),
        vocabulary={f"w{i}": i for i in range(vocab_size)},
        tokenizer=str.split,
        stopwords=(),
        lower=False,
    )


def calc_npmi_coherence_primitive(
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
                    npmi = np.float64(-1.0)
                else:
                    pmi = np.log(p_w1_w2 / (p_w1 * p_w2) + EPSILON)
                    npmi = pmi / -(np.log(p_w1_w2) + EPSILON)
                topic_coherence += npmi.item()
                n_pairs += 1

        assert n_pairs == top_k * (top_k - 1) // 2
        topic_coherence = topic_coherence / n_pairs
        coherence.append(topic_coherence)

    return np.mean(coherence)


def test_npmi_coherence(data, doc_bounds, phi, theta, config):
    top_k = 8

    loader = _loader_from_tokenized(
        data=data,
        document_bounds=doc_bounds,
        vocab_size=config.vocab_size,
    )
    bow = build_bow_from_loader(loader)

    assert bow.shape[0] == config.n_documents
    assert bow.shape[1] == config.vocab_size + 1

    coherence_primitive = calc_npmi_coherence_primitive(
        bow=bow,
        phi_wt=phi,
        vocab_size=config.vocab_size,
        n_documents=config.n_documents,
        n_topics=config.n_topics,
        top_k=top_k,
    )

    coherence_metric = mtc.NPMICoherenceMetric(bow=bow, top_k=top_k)
    coherence_metric.partial_update(
        batch=data,
        phi=phi,
        theta=theta,
        valid_mask=jnp.ones_like(data, dtype=jnp.bool_),
    )

    coherence = coherence_metric.flush()

    assert_allclose(coherence, coherence_primitive, rtol=1e-5, atol=1e-6)


def test_npmi_coherence_zero_cooccurrence_is_minus_one():
    bow = sp.csr_matrix(
        np.array(
            [
                [1, 0],
                [0, 1],
            ],
            dtype=np.uint32,
        )
    )
    phi = jnp.array(
        [
            [0.9],
            [0.8],
        ]
    )

    metric = mtc.NPMICoherenceMetric(bow=bow, top_k=2)
    metric.partial_update(
        batch=jnp.array([0, 1]),
        phi=phi,
        theta=jnp.ones((2, 1)),
        valid_mask=jnp.array([True, True]),
    )

    actual = metric.flush()

    assert_allclose(actual, -1.0, rtol=1e-6, atol=1e-7)
