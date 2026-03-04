import pytest

import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from .algo_primitives import (
    calc_phi_hatch_primitive,
    calc_theta_primitive,
    calc_p_it_primitive,
    calc_n_t_primitive,
    calc_phi_wt_primitive,
)
from .math_primitives import (
    calc_norm_vector_primitive,
    calc_norm_matrix_primitive,
    EPSILON,
)
from cartm.model import ContextTopicModel
from cartm.preprocessing import BatchLoader
import cartm.metrics as mtc
import cartm.regularization as reg


@pytest.fixture
def model(config):
    return ContextTopicModel(
        vocab_size=config.vocab_size,
        ctx_len=config.ctx_len,
        n_topics=config.n_topics,
        gamma=config.gamma,
        eps=EPSILON,
    )


@pytest.fixture
def phi(config):
    key = jax.random.key(config.seed)
    phi = jax.random.uniform(key=key, shape=(config.vocab_size, config.n_topics))
    phi = calc_norm_matrix_primitive(phi)
    return phi


@pytest.fixture
def n_t(config):
    return jnp.full(
        shape=(config.n_topics,), fill_value=config.n_words / config.n_topics
    )


def test_norm(model, config):
    key = jax.random.key(config.seed)
    random_vector = jax.random.normal(key=key, shape=(config.n_words,))
    random_matrix = jax.random.normal(key=key, shape=(config.n_words, config.n_words))

    vec_norm_primitive = calc_norm_vector_primitive(random_vector)
    vec_norm_model = model._norm(random_vector)
    assert_allclose(vec_norm_model, vec_norm_primitive, rtol=1e-5, atol=1e-6)
    assert not jnp.any(jnp.isnan(vec_norm_model))

    mat_norm_primitive = calc_norm_matrix_primitive(random_matrix)
    mat_norm_model = model._norm(random_matrix)
    assert_allclose(mat_norm_model, mat_norm_primitive, rtol=1e-5, atol=1e-6)
    assert not jnp.any(jnp.isnan(mat_norm_model))


def test_phi_hatch(model, phi, n_t, config):
    phi_hatch_primitive = calc_phi_hatch_primitive(
        phi=phi, n_t=n_t, vocab_size=config.vocab_size, n_topics=config.n_topics
    )
    phi_hatch_model = model._calc_phi_hatch(phi=phi, n_t=n_t)
    assert_allclose(phi_hatch_primitive, phi_hatch_model, rtol=1e-5, atol=1e-6)


def test_theta(model, phi, n_t, data, doc_bounds, config):
    phi_hatch = model._calc_phi_hatch(phi=phi, n_t=n_t)

    theta_primitive = calc_theta_primitive(
        data=data,
        phi_hatch=phi_hatch,
        doc_bounds=doc_bounds,
        ctx_len=config.ctx_len,
        gamma=config.gamma,
    )
    theta_model = model._calc_theta(
        batch=data,
        phi_hatch=phi_hatch,
        ctx_bounds=doc_bounds,
    )
    assert_allclose(jnp.abs(theta_model - theta_primitive).sum(), 0.0, atol=1e-5)
    assert_allclose(theta_model, theta_primitive, rtol=1e-5, atol=1e-6)


def test_p_ti(model, phi, n_t, data, doc_bounds, config):
    phi_hatch = model._calc_phi_hatch(phi=phi, n_t=n_t)
    theta = model._calc_theta(
        batch=data,
        phi_hatch=phi_hatch,
        ctx_bounds=doc_bounds,
    )

    p_ti_primitive = calc_p_it_primitive(
        data=data,
        phi=phi,
        theta=theta,
        n_topics=config.n_topics,
        n_words=config.n_words,
    )
    p_ti_model, _ = model._calc_p_ti(
        batch=data,
        phi=phi,
        theta=theta,
    )
    assert_allclose(p_ti_model, p_ti_primitive, rtol=1e-5, atol=1e-6)


def test_n_t(model, phi, n_t, data, doc_bounds, config):
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

    n_t_primitive = calc_n_t_primitive(
        p_ti=p_ti, n_topics=config.n_topics, n_words=config.n_words
    )
    n_t_model = model._calc_n_t(p_ti=p_ti)
    assert_allclose(n_t_model, n_t_primitive, rtol=1e-5, atol=1e-6)


def test_phi(model, phi, n_t, data, doc_bounds, config):
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

    phi_primitive = calc_phi_wt_primitive(
        data=data,
        p_ti=p_ti,
        vocab_size=config.vocab_size,
        n_topics=config.n_topics,
        n_words=config.n_words,
    )
    phi_model = model._calc_phi(batch=data, phi=phi, p_ti=p_ti, grad_reg=grad_reg)
    assert_allclose(phi_model, phi_primitive, rtol=1e-5, atol=1e-6)


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
    assert_allclose(phi_it_model, phi_it, rtol=1e-5, atol=1e-6)
    assert_allclose(phi_new_model, phi_new, rtol=1e-5, atol=1e-6)
    assert_allclose(theta_model, theta, rtol=1e-5, atol=1e-6)
    assert_allclose(n_t_model, n_t_new, rtol=1e-5, atol=1e-6)


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


def test_calc_metric(model, config):
    metric = mtc.PerplexityMetric()
    model.add_metric(metric=metric)
    size = (config.n_words, config.n_topics)
    phi_it = jnp.zeros(size)
    theta = jnp.zeros(size)
    model._calc_metrics(phi_it=phi_it, phi_wt=None, theta=theta, verbose=False)

    assert len(metric.history) == 1


def test_add_remove_regularization(model):
    regularization = reg.DecorrelationRegularization(tau=0.3)
    model.add_regularization(regularization=regularization)

    assert len(model._regularizations) == 1

    model.remove_regularization(regularization.tag)

    assert len(model._regularizations) == 0


def test_compose_regularizations(model, phi):
    regularization1 = reg.DecorrelationRegularization(tau=0.1, tag="reg1")
    regularization2 = reg.DecorrelationRegularization(tau=0.6, tag="reg2")
    model.add_regularization(regularization=regularization1)
    model.add_regularization(regularization=regularization2)

    reg1_grad = jax.grad(regularization1)
    reg2_grad = jax.grad(regularization2)
    reg_grad = model._compose_regularizations()

    assert_allclose(
        reg_grad(phi), reg1_grad(phi) + reg2_grad(phi), rtol=1e-5, atol=1e-6
    )
