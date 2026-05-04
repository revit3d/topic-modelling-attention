import jax
import numpy as np
from numpy.testing import assert_allclose

import cartm.regularization as reg


def calc_sparsity_grad_primitive(phi: jax.Array, alpha: jax.Array, tau: float):
    return tau * alpha / phi


def test_sparsity_reg(phi, config):
    tau = 0.3
    key = jax.random.key(config.seed)
    prior = jax.random.uniform(
        key=key,
        shape=(config.vocab_size, config.n_topics),
        minval=0.01,
        maxval=1.0,
    )
    prior = prior / prior.sum(axis=0, keepdims=True)

    regularization = reg.SparsityRegularization(alpha=prior, tau=tau)
    regularization = jax.grad(regularization)
    sparsity_regularization = regularization(phi)

    sparsity_primitive = calc_sparsity_grad_primitive(phi=phi, alpha=prior, tau=tau)

    assert_allclose(sparsity_regularization, sparsity_primitive, rtol=1e-5, atol=1e-6)
