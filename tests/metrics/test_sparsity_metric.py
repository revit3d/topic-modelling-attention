import pytest

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose

import cartm.metrics as mtc


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


@pytest.mark.parametrize("zero_threshold", [0.1, 0.3, 0.6, 0.8])
def test_sparsity(zero_threshold, data, phi, theta, config):
    thresh_mask = phi < zero_threshold
    phi_wt_thresh = phi.at[thresh_mask].set(0.0)
    sparsity_primitive = calc_sparsity_primitive(
        phi_wt=phi_wt_thresh,
        vocab_size=config.vocab_size,
        n_topics=config.n_topics,
    )
    sparsity_metric = mtc.SparsityMetric()
    sparsity_metric.partial_update(
        batch=data,
        phi=phi_wt_thresh,
        theta=theta,
        valid_mask=jnp.ones_like(data, dtype=jnp.bool_),
    )
    sparsity = sparsity_metric.flush()
    assert_allclose(sparsity, sparsity_primitive, rtol=1e-5, atol=1e-6)
