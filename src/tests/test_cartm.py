import pytest

import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from cartm import ContextTopicModel
from cartm.core import get_context_weights_1d
from cartm.preprocessing import BatchedCorpusLoader
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
    model = ContextTopicModel(
        vocab_size=config.vocab_size,
        ctx_len=config.ctx_len,
        n_topics=config.n_topics,
        gamma=config.gamma,
    )
    model._init_state(seed=config.seed)
    return model


def test_step(phi, n_t, data, doc_bounds, config):
    phi_it = phi[data]
    phi_hatch_primitive = calc_phi_hatch_primitive(
        phi=phi, n_t=n_t, vocab_size=config.vocab_size, n_topics=config.n_topics
    )
    theta_primitive = calc_theta_primitive(
        data=data,
        phi_hatch=phi_hatch_primitive,
        doc_bounds=doc_bounds,
        ctx_len=config.ctx_len,
        gamma=config.gamma,
    )
    p_ti_primitive = calc_p_it_primitive(
        data=data,
        phi=phi,
        theta=theta_primitive,
        n_topics=config.n_topics,
        n_words=config.n_words,
    )
    n_t_new = calc_n_t_primitive(
        p_ti=p_ti_primitive, n_topics=config.n_topics, n_words=config.n_words
    )
    phi_primitive = calc_phi_wt_primitive(
        data=data,
        p_ti=p_ti_primitive,
        vocab_size=config.vocab_size,
        n_topics=config.n_topics,
        n_words=config.n_words,
    )

    grad_reg = jax.grad(lambda _: 0.0)
    ctx_weights = get_context_weights_1d(
        ctx_len=config.ctx_len, gamma=config.gamma, self_aware=False
    )
    theta_model, n_t_model, n_wt_model = ContextTopicModel._step(
        batch=data,
        ctx_bounds=doc_bounds,
        phi=phi,
        n_t=n_t,
        ctx_weights=ctx_weights,
        num_attn_passes=1,
    )
    grad_phi = grad_reg(phi)
    phi_model = ContextTopicModel._update_phi(
        phi=phi,
        grad_phi=grad_phi,
        n_wt=n_wt_model,
    )
    assert_allclose(theta_model, theta_primitive, rtol=1e-5, atol=1e-6)
    assert_allclose(n_t_model, n_t_new, rtol=1e-5, atol=1e-6)
    assert_allclose(phi_model, phi_primitive, rtol=1e-5, atol=1e-6)


def test_batched_step(model, phi, n_t, data, doc_bounds, config):
    batches = BatchedCorpusLoader(data=data, doc_bounds=doc_bounds, batch_size=10)
    grad_reg = jax.grad(lambda _: 0.0)
    ctx_weights = get_context_weights_1d(
        ctx_len=config.ctx_len, gamma=config.gamma, self_aware=False
    )

    _ = model._batched_step_wrapper(
        batches=batches,
        ctx_weights=ctx_weights,
        grad_reg=grad_reg,
        num_attn_passes=1,
        lr=0.01,
        num_batches_before_update=-1,
    )
    _ = model._batched_step_wrapper(
        batches=batches,
        ctx_weights=ctx_weights,
        grad_reg=grad_reg,
        num_attn_passes=1,
        lr=0.01,
        num_batches_before_update=1,
    )


def test_add_remove_metric(model):
    metric = mtc.PerplexityMetric()
    model.add_metric(metric=metric)

    assert len(model._metrics) == 1

    model.remove_metric(tag=metric.tag)

    assert len(model._metrics) == 0


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
