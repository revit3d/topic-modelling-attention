import pytest

import jax
import numpy as np
from numpy.testing import assert_allclose

import cartm.regularization as reg


n_words = 100
n_topics = 10
vocab_size = 20
tau = 0.3
seed = 42


def calc_sparsity_grad_primitive(phi: jax.Array, alpha: jax.Array, tau: float):
    return tau * alpha / phi


def calc_decorrelation_grad_primitive(phi: jax.Array, tau: float):
    grad = np.zeros((vocab_size, n_topics))
    for w in range(vocab_size):
        for t in range(n_topics):
            for s in range(n_topics):
                if t == s:
                    continue
                grad[w][t] += phi[w][s]
    return tau * grad


@pytest.fixture
def phi():
    key = jax.random.key(seed)
    phi = jax.random.uniform(key=key, shape=(vocab_size, n_topics))
    phi /= phi.sum(axis=0, keepdims=True)
    return phi


def test_sparsity_reg(phi):
    key = jax.random.key(seed)
    prior = jax.random.normal(key=key, shape=(vocab_size, n_topics))
    prior /= prior.sum(axis=0)

    regularization = reg.SparsityRegularization(alpha=prior, tau=tau)
    regularization = jax.grad(regularization)
    sparsity_regularization = regularization(phi)

    sparsity_primitive = calc_sparsity_grad_primitive(phi=phi, alpha=prior, tau=tau)

    assert_allclose(sparsity_regularization, sparsity_primitive)


def test_decorrelation_reg(phi):
    regularization = reg.DecorrelationRegularization(tau=tau)
    regularization = jax.grad(regularization)
    decorrelation_regularization = regularization(phi)

    decorrelation_primitive = calc_decorrelation_grad_primitive(phi=phi, tau=tau)

    assert_allclose(decorrelation_regularization, decorrelation_primitive, rtol=5e-7)
