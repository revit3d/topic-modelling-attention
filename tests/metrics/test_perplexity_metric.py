import pytest

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose

from cartm.core import EPSILON
import cartm.metrics as mtc


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


def test_perplexity(data, phi, theta, config):
    perplexity_primitive = calc_perplexity_primitive(
        phi_it=phi[data],
        theta=theta,
        n_topics=config.n_topics,
        n_words=config.n_words,
    )
    perplexity_metric = mtc.PerplexityMetric()
    valid_mask = jnp.ones_like(data, dtype=jnp.bool_)
    perplexity_metric.partial_update(
        batch=data[:50],
        phi=phi,
        theta=theta[:50],
        valid_mask=valid_mask[:50],
    )
    perplexity_metric.partial_update(
        batch=data[50:],
        phi=phi,
        theta=theta[50:],
        valid_mask=valid_mask[50:],
    )
    perplexity = perplexity_metric.flush()
    assert_allclose(perplexity, perplexity_primitive, rtol=1e-5, atol=1e-6)


def test_perplexity_ignores_padding_tokens():
    phi = jnp.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
        ]
    )
    theta = jnp.array(
        [
            [1.0, 0.0],
            [0.25, 0.75],
            [0.0, 1.0],
        ]
    )
    batch = jnp.array([0, 1, 0])
    valid_mask = jnp.array([True, True, False])

    p0 = 0.9
    p1 = 0.2 * 0.25 + 0.8 * 0.75
    expected = np.exp(-(np.log(p0) + np.log(p1)) / 2)

    metric = mtc.PerplexityMetric()
    metric.partial_update(
        batch=batch,
        phi=phi,
        theta=theta,
        valid_mask=valid_mask,
    )

    actual = metric.flush()

    assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_perplexity_flush_without_words_raises():
    metric = mtc.PerplexityMetric()

    with pytest.raises(ValueError, match="No words"):
        metric.flush()
