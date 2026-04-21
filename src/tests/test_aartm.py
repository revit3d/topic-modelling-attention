import pytest
import jax
from numpy.testing import assert_allclose

from cartm import AttentiveTopicModel
from cartm.core import get_context_weights_1d
from tests.algo_primitives import (
    calc_theta_primitive,
    calc_p_ti_primitive,
    calc_N_tw_primitive,
    calc_n_t_primitive,
    calc_phi_tw_primitive,
)
from tests.math_primitives import calc_norm_matrix_primitive


@pytest.fixture
def phi(config):
    key = jax.random.key(config.seed)
    phi = jax.random.uniform(key=key, shape=(config.n_topics, config.vocab_size))
    phi = calc_norm_matrix_primitive(phi)
    return phi.T


def test_step(phi, n_t, data, doc_bounds, config):
    phi_hatch = phi[data]
    theta_primitive = calc_theta_primitive(
        data=data,
        phi_hatch=phi,
        doc_bounds=doc_bounds,
        ctx_len=config.ctx_len,
        gamma=config.gamma,
    )
    p_ti_primitive = calc_p_ti_primitive(
        phi=phi_hatch,
        theta=theta_primitive,
        n_t=n_t,
        n_topics=config.n_topics,
        n_words=config.n_words,
    )

    n_t_primitive = calc_n_t_primitive(
        p_ti=p_ti_primitive, n_topics=config.n_topics, n_words=config.n_words
    )

    N_tw_primitive = calc_N_tw_primitive(
        data=data,
        p_ti=p_ti_primitive,
        theta=theta_primitive,
        doc_bounds=doc_bounds,
        vocab_size=config.vocab_size,
        n_topics=config.n_topics,
        n_words=config.n_words,
        ctx_len=config.ctx_len,
        gamma=config.gamma,
    )
    phi_primitive = calc_phi_tw_primitive(
        data=data,
        p_ti=p_ti_primitive,
        N_tw=N_tw_primitive,
        phi_old=phi,
        vocab_size=config.vocab_size,
        n_topics=config.n_topics,
        n_words=config.n_words,
    )

    grad_reg = jax.grad(lambda _: 0.0)
    ctx_weights = get_context_weights_1d(
        ctx_len=config.ctx_len, gamma=config.gamma, self_aware=False
    )
    phi_it_model, phi_model, theta_model, n_t_model, n_wt_model, N_wt_model = AttentiveTopicModel._step(
        batch=data,
        ctx_bounds=doc_bounds,
        phi=phi,
        n_t=n_t,
        ctx_weights=ctx_weights,
        grad_reg=grad_reg,
        num_attn_passes=1,
    )

    assert_allclose(theta_model, theta_primitive, rtol=1e-5, atol=1e-6)
    assert_allclose(n_t_model, n_t_primitive, rtol=1e-5, atol=1e-6)
    assert_allclose(N_wt_model, N_tw_primitive.T, rtol=1e-5, atol=1e-6)
    assert_allclose(phi_model, phi_primitive, rtol=1e-5, atol=1e-6)
