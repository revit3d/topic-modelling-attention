import pytest

import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from cartm import ContextTopicModel
from cartm.core import get_context_weights_1d
from cartm.preprocessing import BatchLoader
import cartm.metrics as mtc
import cartm.regularization as reg
from tests.algo_primitives import (
    calc_phi_hatch_primitive,
    calc_theta_primitive,
    calc_p_it_primitive,
    calc_n_t_primitive,
    calc_phi_wt_primitive,
)


@pytest.fixture
def model(config):
    return ContextTopicModel(
        vocab_size=config.vocab_size,
        ctx_len=config.ctx_len,
        n_topics=config.n_topics,
        gamma=config.gamma,
    )


def test_step(phi, n_t, data, doc_bounds, config):
    phi_it = phi[data]
    phi_hatch = calc_phi_hatch_primitive(
        phi=phi, n_t=n_t, vocab_size=config.vocab_size, n_topics=config.n_topics
    )
    theta = calc_theta_primitive(
        data=data,
        phi_hatch=phi_hatch,
        doc_bounds=doc_bounds,
        ctx_len=config.ctx_len,
        gamma=config.gamma,
    )
    p_ti = calc_p_it_primitive(
        data=data,
        phi=phi,
        theta=theta,
        n_topics=config.n_topics,
        n_words=config.n_words,
    )
    n_t_new = calc_n_t_primitive(
        p_ti=p_ti, n_topics=config.n_topics, n_words=config.n_words
    )
    phi_new = calc_phi_wt_primitive(
        data=data,
        p_ti=p_ti,
        vocab_size=config.vocab_size,
        n_topics=config.n_topics,
        n_words=config.n_words,
    )

    grad_reg = jax.grad(lambda _: 0.0)
    ctx_weights = get_context_weights_1d(
        ctx_len=config.ctx_len, gamma=config.gamma, self_aware=False
    )
    phi_it_model, phi_new_model, theta_model, n_t_model = ContextTopicModel._step(
        batch=data,
        ctx_bounds=doc_bounds,
        phi=phi,
        n_t=n_t,
        ctx_weights=ctx_weights,
        grad_reg=grad_reg,
        num_attn_passes=1,
    )
    assert_allclose(phi_it_model, phi_it, rtol=1e-5, atol=1e-6)
    assert_allclose(phi_new_model, phi_new, rtol=1e-5, atol=1e-6)
    assert_allclose(theta_model, theta, rtol=1e-5, atol=1e-6)
    assert_allclose(n_t_model, n_t_new, rtol=1e-5, atol=1e-6)


def test_batched_step(model, phi, n_t, data, doc_bounds, config):
    batches = BatchLoader(data=data, doc_bounds=doc_bounds, batch_size=10)
    grad_reg = jax.grad(lambda _: 0.0)
    ctx_weights = get_context_weights_1d(
        ctx_len=config.ctx_len, gamma=config.gamma, self_aware=False
    )
    _ = model._batched_step_wrapper(
        batches=batches,
        phi=phi,
        n_t=n_t,
        ctx_weights=ctx_weights,
        grad_reg=grad_reg,
        num_attn_passes=1,
        lr=0.01,
    )


def test_add_remove_metric(model):
    metric = mtc.PerplexityMetric()
    model.add_metric(metric=metric)

    assert len(model._metrics) == 1

    model.remove_metric(tag=metric.tag)

    assert len(model._metrics) == 0


def test_calc_metric(model, data, config):
    metric = mtc.PerplexityMetric()
    model.add_metric(metric=metric)
    size = (config.n_words, config.n_topics)
    phi_it = jnp.zeros(size)
    theta = jnp.zeros(size)
    model._calc_metrics(
        batch=data,
        phi_it=phi_it,
        phi_wt=None,
        theta=theta,
        verbose=False,
    )

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
