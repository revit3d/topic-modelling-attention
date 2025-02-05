import pytest

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose

from cartm.model import ContextTopicModel
from cartm.prepocessing import BatchLoader
import cartm.metrics as mtc
import cartm.regularization as reg


vocab_size = 40
n_words = 100
n_documents = 20
n_topics = 12
ctx_len = 5
gamma = 0.6
eps = 1e-12
seed = 42


def calc_norm_matrix_primitive(x: jax.Array):
    assert len(x.shape) == 2

    x_norm = []
    for x_j in x.T:
        x_j_norm = calc_norm_vector_primitive(x=x_j)
        x_norm.append(x_j_norm)
    return jnp.stack(x_norm).T


def calc_norm_vector_primitive(x: jax.Array):
    assert len(x.shape) == 1

    x_plus = []
    for x_i in x:
        x_plus.append(max(x_i, 0))
    x_plus = jnp.array(x_plus)

    x_norm = x_plus / (jnp.sum(x_plus) + eps)
    return x_norm


def calc_phi_hatch_primitive(phi: jax.Array, n_t: jax.Array):
    phi_hatch_naive = np.zeros_like(phi, dtype=float)
    for w in range(vocab_size):
        for t in range(n_topics):
            phi_hatch_naive[w][t] = phi[w][t] * n_t[t]
        phi_hatch_naive[w] = calc_norm_vector_primitive(phi_hatch_naive[w])
    return jnp.array(phi_hatch_naive)


def calc_theta_primitive(data: jax.Array, phi_hatch: jax.Array, doc_bounds: jax.Array):
    theta = []
    doc_bounds_prefix = set((doc_bounds[1:] - 1).tolist())
    doc_bounds_suffix = set(doc_bounds[:-1].tolist())

    for w in range(n_words):
        prefix_context_vec, prefix_context_weights = [], []
        for i in range(1, ctx_len + 1):
            if w - i >= 0 and w - i not in doc_bounds_prefix:
                prefix_context_vec.append(phi_hatch[data[w - i]])
                prefix_context_weights.append(gamma * (1 - gamma)**i)
            else:
                break

        suffix_context_vec, suffix_context_weights = [], []
        for i in range(1, ctx_len + 1):
            if w + i < n_words and w + i not in doc_bounds_suffix:
                suffix_context_vec.append(phi_hatch[data[w + i]])
                suffix_context_weights.append(gamma * (1 - gamma)**i)
            else:
                break

        context_weights = jnp.array(prefix_context_weights[::-1] + suffix_context_weights)
        context_weights = calc_norm_vector_primitive(context_weights)

        context_vec = jnp.array(prefix_context_vec[::-1] + suffix_context_vec)
        context_vec = context_vec * context_weights[:, None]
        context_vec = jnp.sum(context_vec, axis=0)

        if context_vec.shape == (0, ):
            # no words in context
            context_vec = jnp.zeros(shape=(n_topics, ))

        assert context_vec.shape == (n_topics, )

        theta.append(context_vec)
    return jnp.array(theta)


def calc_p_ti_primitive(data: jax.Array, phi: jax.Array, theta: jax.Array):
    p_ti_naive = np.zeros((n_words, n_topics))
    for w in range(n_words):
        for t in range(n_topics):
            word = data[w]
            p_ti_naive[w][t] = phi[word][t] * theta[w][t]
        p_ti_naive[w] = calc_norm_vector_primitive(p_ti_naive[w])
    return jnp.array(p_ti_naive)


def calc_n_t_primitive(p_ti: jax.Array):
    n_t = np.zeros((n_topics, ))
    for w in range(n_words):
        for t in range(n_topics):
            n_t[t] += p_ti[w][t]
    return jnp.array(n_t)


def calc_phi_primitive(data: jax.Array, p_ti: jax.Array):
    # not testing partial derivatives here, trusting in jax.grad
    phi = np.zeros((vocab_size, n_topics))
    for w in range(n_words):
        for t in range(n_topics):
            word = data[w]
            phi[word][t] += p_ti[w][t]
    phi = calc_norm_matrix_primitive(phi)
    return phi


@pytest.fixture
def model():
    return ContextTopicModel(
        vocab_size=vocab_size,
        ctx_len=ctx_len,
        n_topics=n_topics,
        gamma=gamma,
        eps=eps,
    )


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
def phi():
    key = jax.random.key(seed)
    phi = jax.random.uniform(key=key, shape=(vocab_size, n_topics))
    phi = calc_norm_matrix_primitive(phi)
    return phi


@pytest.fixture
def n_t():
    return jnp.full(shape=(n_topics, ), fill_value=n_words / n_topics)


def test_norm(model):
    key = jax.random.key(seed)
    random_vector = jax.random.normal(key=key, shape=(n_words, ))
    random_matrix = jax.random.normal(key=key, shape=(n_words, n_words))

    vec_norm_primitive = calc_norm_vector_primitive(random_vector)
    vec_norm_model = model._norm(random_vector)
    assert_allclose(vec_norm_model, vec_norm_primitive)
    assert not jnp.any(jnp.isnan(vec_norm_model))

    mat_norm_primitive = calc_norm_matrix_primitive(random_matrix)
    mat_norm_model = model._norm(random_matrix)
    assert_allclose(mat_norm_model, mat_norm_primitive, rtol=5e-7)
    assert not jnp.any(jnp.isnan(mat_norm_model))


def test_phi_hatch(model, phi, n_t):
    phi_hatch_primitive = calc_phi_hatch_primitive(phi=phi, n_t=n_t)
    phi_hatch_model = model._calc_phi_hatch(phi=phi, n_t=n_t)
    assert jnp.allclose(phi_hatch_primitive, phi_hatch_model)


def test_theta(model, phi, n_t, data, doc_bounds):
    phi_hatch = model._calc_phi_hatch(phi=phi, n_t=n_t)

    theta_primitive = calc_theta_primitive(
        data=data,
        phi_hatch=phi_hatch,
        doc_bounds=doc_bounds,
    )
    theta_model = model._calc_theta(
        batch=data,
        phi_hatch=phi_hatch,
        ctx_bounds=doc_bounds,
    )
    assert np.isclose(jnp.abs(theta_model - theta_primitive).sum(), 0.0, atol=1e-5)
    assert_allclose(theta_model, theta_primitive, rtol=5e-7)


def test_p_ti(model, phi, n_t, data, doc_bounds):
    phi_hatch = model._calc_phi_hatch(phi=phi, n_t=n_t)
    theta = model._calc_theta(
        batch=data,
        phi_hatch=phi_hatch,
        ctx_bounds=doc_bounds,
    )

    p_ti_primitive = calc_p_ti_primitive(
        data=data,
        phi=phi,
        theta=theta,
    )
    p_ti_model, _ = model._calc_p_ti(
        batch=data,
        phi=phi,
        theta=theta,
    )
    assert_allclose(p_ti_model, p_ti_primitive)


def test_n_t(model, phi, n_t, data, doc_bounds):
    phi_hatch = model._calc_phi_hatch(phi=phi, n_t=n_t)
    theta = model._calc_theta(
        batch=data,
        phi_hatch=phi_hatch,
        ctx_bounds=doc_bounds,
    )
    p_ti, _ = model._calc_p_ti(
        batch=data,
        phi=phi,
        theta=theta,
    )

    n_t_primitive = calc_n_t_primitive(p_ti=p_ti)
    n_t_model = model._calc_n_t(p_ti=p_ti)
    assert_allclose(n_t_model, n_t_primitive, rtol=5e-7)


def test_phi(model, phi, n_t, data, doc_bounds):
    phi_hatch = model._calc_phi_hatch(phi=phi, n_t=n_t)
    theta = model._calc_theta(
        batch=data,
        phi_hatch=phi_hatch,
        ctx_bounds=doc_bounds,
    )
    p_ti, _ = model._calc_p_ti(
        batch=data,
        phi=phi,
        theta=theta,
    )
    grad_reg = jax.grad(lambda _: 0.0)

    phi_primitive = calc_phi_primitive(data=data, p_ti=p_ti)
    phi_model = model._calc_phi(batch=data, phi=phi, p_ti=p_ti, grad_reg=grad_reg)
    assert_allclose(phi_model, phi_primitive, rtol=5e-7)


def test_step(model, phi, n_t, data, doc_bounds):
    phi_hatch = model._calc_phi_hatch(phi=phi, n_t=n_t)
    theta = model._calc_theta(
        batch=data,
        phi_hatch=phi_hatch,
        ctx_bounds=doc_bounds,
    )
    p_ti, phi_it = model._calc_p_ti(
        batch=data,
        phi=phi,
        theta=theta,
    )
    n_t_new = model._calc_n_t(p_ti=p_ti)
    grad_reg = jax.grad(lambda _: 0.0)
    phi_new = model._calc_phi(
        batch=data,
        phi=phi,
        p_ti=p_ti,
        grad_reg=grad_reg,
    )

    phi_it_model, phi_new_model, theta_model, n_t_model = model._step(
        batch=data,
        ctx_bounds=doc_bounds,
        phi=phi,
        n_t=n_t,
        grad_reg=grad_reg,
    )
    assert_allclose(phi_it_model, phi_it, rtol=5e-7)
    assert_allclose(phi_new_model, phi_new, rtol=5e-7)
    assert_allclose(theta_model, theta, rtol=5e-7)
    assert_allclose(n_t_model, n_t_new, rtol=5e-7)


def test_batched_step(model, phi, n_t, data, doc_bounds):
    batches = BatchLoader(data=data, doc_bounds=doc_bounds, batch_size=10)
    grad_reg = jax.grad(lambda _: 0.0)
    _ = model._batched_step_wrapper(
        batches=batches,
        phi=phi,
        n_t=n_t,
        grad_reg=grad_reg,
        lr=0.01,
    )


def test_add_remove_metric(model):
    metric = mtc.PerplexityMetric()
    model.add_metric(metric=metric)

    assert len(model._metrics) == 1

    model.remove_metric(tag=metric.tag)

    assert len(model._metrics) == 0


def test_calc_metric(model):
    metric = mtc.PerplexityMetric()
    model.add_metric(metric=metric)
    phi_it = jnp.zeros((n_words, n_topics))
    theta = jnp.zeros((n_words, n_topics))
    model._calc_metrics(phi_it=phi_it, phi_wt=None, theta=theta, verbose=False)

    assert len(metric.history) == 1


def test_add_remove_regularization(model):
    regularization = reg.DecorrelationRegularization(tau=0.3)
    model.add_regularization(regularization=regularization)

    assert len(model._regularizations) == 1

    model.remove_regularization(regularization.tag)

    assert len(model._regularizations) == 0


def test_compose_regularizations(model, phi):
    regularization1 = reg.DecorrelationRegularization(tau=0.1, tag='reg1')
    regularization2 = reg.DecorrelationRegularization(tau=0.6, tag='reg2')
    model.add_regularization(regularization=regularization1)
    model.add_regularization(regularization=regularization2)

    reg1_grad = jax.grad(regularization1)
    reg2_grad = jax.grad(regularization2)
    reg_grad = model._compose_regularizations()

    assert_allclose(reg_grad(phi), reg1_grad(phi) + reg2_grad(phi), rtol=5e-7)
