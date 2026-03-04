import pytest
import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from .algo_primitives import (
    calc_theta_primitive,
    calc_p_ti_primitive,
    calc_N_tw_primitive,
    calc_n_t_primitive,
    calc_phi_tw_primitive,
)
from .math_primitives import (
    calc_norm_matrix_primitive,
    EPSILON,
)
from cartm.improved_model import AttentiveTopicModel


@pytest.fixture
def model(config):
    return AttentiveTopicModel(
        vocab_size=config.vocab_size,
        ctx_len=config.ctx_len,
        n_topics=config.n_topics,
        gamma=config.gamma,
        eps=EPSILON,
    )


@pytest.fixture
def phi(config):
    key = jax.random.key(config.seed)
    phi = jax.random.uniform(key=key, shape=(config.n_topics, config.vocab_size))
    phi = calc_norm_matrix_primitive(phi)
    return phi


@pytest.fixture
def n_t(config):
    return jnp.full(
        shape=(config.n_topics,), fill_value=config.n_words / config.n_topics
    )


def test_p_ti(model, phi, n_t, data, doc_bounds, config):
    phi_ti = model._calc_phi_hatch(phi=phi, batch=data)
    theta = model._calc_theta(phi_ti=phi_ti, ctx_bounds=doc_bounds)

    p_ti_primitive = calc_p_ti_primitive(
        phi=phi_ti,
        theta=theta,
        n_t=n_t,
        n_topics=config.n_topics,
        n_words=config.n_words,
    )
    p_ti_model, _ = model._calc_p_ti(phi_ti=phi_ti, theta=theta, n_t=n_t)

    assert_allclose(p_ti_model, p_ti_primitive, rtol=1e-5, atol=1e-6)


def test_N_tw(model, phi, n_t, data, doc_bounds, config):
    phi_ti = model._calc_phi_hatch(phi=phi, batch=data)
    theta = model._calc_theta(phi_ti=phi_ti, ctx_bounds=doc_bounds)
    p_ti, _ = model._calc_p_ti(phi_ti=phi_ti, theta=theta, n_t=n_t)

    N_tw_primitive = calc_N_tw_primitive(
        data=data,
        p_ti=p_ti,
        theta=theta,
        doc_bounds=doc_bounds,
        vocab_size=config.vocab_size,
        n_topics=config.n_topics,
        n_words=config.n_words,
        ctx_len=config.ctx_len,
        gamma=config.gamma,
    )
    N_tw_model = model._calc_N_tw(
        phi=phi,
        p_ti=p_ti,
        theta=theta,
        batch=data,
        ctx_bounds=doc_bounds,
    )

    assert_allclose(N_tw_model, N_tw_primitive, rtol=1e-5, atol=1e-6)


def test_phi(model, phi, n_t, data, doc_bounds, config):
    phi_ti = model._calc_phi_hatch(phi=phi, batch=data)
    theta = model._calc_theta(phi_ti=phi_ti, ctx_bounds=doc_bounds)
    p_ti, _ = model._calc_p_ti(phi_ti=phi_ti, theta=theta, n_t=n_t)
    N_tw = model._calc_N_tw(
        phi=phi,
        p_ti=p_ti,
        theta=theta,
        batch=data,
        ctx_bounds=doc_bounds,
    )

    grad_reg = jax.grad(lambda _: 0.0)

    phi_primitive = calc_phi_tw_primitive(
        data=data,
        p_ti=p_ti,
        N_tw=N_tw,
        phi_old=phi,
        vocab_size=config.vocab_size,
        n_topics=config.n_topics,
        n_words=config.n_words,
    )
    phi_model = model._calc_phi(
        batch=data, phi=phi, p_ti=p_ti, N_tw=N_tw, grad_reg=grad_reg
    )

    assert_allclose(phi_model, phi_primitive, rtol=1e-5, atol=1e-6)


def test_step(model, phi, n_t, data, doc_bounds, config):
    grad_reg = jax.grad(lambda _: 0.0)

    phi_it_model, phi_new_model, theta_model, n_t_model = model._step(
        batch=data, ctx_bounds=doc_bounds, phi=phi, n_t=n_t, grad_reg=grad_reg
    )

    phi_hatch = phi[:, data]
    theta = calc_theta_primitive(
        data=data,
        phi_hatch=phi.T,
        doc_bounds=doc_bounds,
        ctx_len=config.ctx_len,
        gamma=config.gamma,
    ).T
    p_ti = calc_p_ti_primitive(
        phi=phi_hatch,
        theta=theta,
        n_t=n_t,
        n_topics=config.n_topics,
        n_words=config.n_words,
    )

    n_t_expected = calc_n_t_primitive(
        p_ti=p_ti.T, n_topics=config.n_topics, n_words=config.n_words
    )

    N_tw = calc_N_tw_primitive(
        data=data,
        p_ti=p_ti,
        theta=theta,
        doc_bounds=doc_bounds,
        vocab_size=config.vocab_size,
        n_topics=config.n_topics,
        n_words=config.n_words,
        ctx_len=config.ctx_len,
        gamma=config.gamma,
    )
    phi_new = calc_phi_tw_primitive(
        data=data,
        p_ti=p_ti,
        N_tw=N_tw,
        phi_old=phi,
        vocab_size=config.vocab_size,
        n_topics=config.n_topics,
        n_words=config.n_words,
    )

    assert_allclose(theta_model, theta, rtol=1e-5, atol=1e-6)
    assert_allclose(n_t_model, n_t_expected, rtol=1e-5, atol=1e-6)
    assert_allclose(phi_new_model, phi_new, rtol=1e-5, atol=1e-6)
