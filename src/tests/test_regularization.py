import jax
import numpy as np
from numpy.testing import assert_allclose

import cartm.regularization as reg


def calc_sparsity_grad_primitive(phi: jax.Array, alpha: jax.Array, tau: float):
    return tau * alpha / phi


def calc_decorrelation_grad_primitive(
    phi: jax.Array,
    tau: float,
    vocab_size: int,
    n_topics: int,
):
    grad = np.zeros((vocab_size, n_topics))
    for w in range(vocab_size):
        for t in range(n_topics):
            for s in range(n_topics):
                if t == s:
                    continue
                grad[w][t] += phi[w][s]
    return -tau * grad


def test_sparsity_reg(phi, config):
    tau = 0.3
    key = jax.random.key(config.seed)
    prior = jax.random.normal(key=key, shape=(config.vocab_size, config.n_topics))
    prior /= prior.sum(axis=0)

    regularization = reg.SparsityRegularization(alpha=prior, tau=tau)
    regularization = jax.grad(regularization)
    sparsity_regularization = regularization(phi)

    sparsity_primitive = calc_sparsity_grad_primitive(phi=phi, alpha=prior, tau=tau)

    assert_allclose(sparsity_regularization, sparsity_primitive, rtol=1e-5, atol=1e-6)


def test_decorrelation_reg(phi, config):
    tau = 0.3
    regularization = reg.DecorrelationRegularization(tau=tau)
    regularization = jax.grad(regularization)
    decorrelation_regularization = regularization(phi)

    decorrelation_primitive = calc_decorrelation_grad_primitive(
        phi=phi,
        tau=tau,
        vocab_size=config.vocab_size,
        n_topics=config.n_topics,
    )

    assert_allclose(decorrelation_regularization, decorrelation_primitive, rtol=1e-5, atol=1e-6)
