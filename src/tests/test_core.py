import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from cartm.core import (
    norm,
    calc_attn,
    calc_attn_transposed,
    get_context_weights_1d,
)
from tests.algo_primitives import (
    calc_attn_primitive,
    calc_attn_transposed_primitive,
)
from tests.math_primitives import (
    calc_norm_vector_primitive,
    calc_norm_matrix_primitive,
)


def test_norm(config):
    key = jax.random.key(config.seed)
    random_vector = jax.random.normal(key=key, shape=(config.n_words,))
    random_matrix = jax.random.normal(key=key, shape=(config.n_words, config.n_words))

    vec_norm_primitive = calc_norm_vector_primitive(random_vector)
    vec_norm_fast = norm(random_vector)
    assert_allclose(vec_norm_fast, vec_norm_primitive, rtol=1e-5, atol=1e-6)
    assert not jnp.any(jnp.isnan(vec_norm_fast))

    mat_norm_primitive = calc_norm_matrix_primitive(random_matrix)
    mat_norm_fast = norm(random_matrix)
    assert_allclose(mat_norm_fast, mat_norm_primitive, rtol=1e-5, atol=1e-6)
    assert not jnp.any(jnp.isnan(mat_norm_fast))


def test_attn(phi, data, doc_bounds, config):
    pad_len = 50
    ctx_weights = get_context_weights_1d(
        ctx_len=config.ctx_len, gamma=config.gamma, self_aware=False
    )
    matrix = phi[data]
    token_mask = jnp.concatenate(
        [
            jnp.ones_like(doc_bounds),
            jnp.zeros(pad_len),
        ],
        dtype=jnp.bool_,
    )

    attn_primitive = calc_attn_primitive(
        matrix=matrix,
        ctx_bounds=doc_bounds,
        ctx_len=config.ctx_len,
        gamma=config.gamma,
    )
    attn_fast = calc_attn(
        matrix=matrix,
        ctx_bounds=doc_bounds,
        ctx_weights=ctx_weights,
        token_mask=token_mask[:len(doc_bounds)],
    )
    attn_fast_padded = calc_attn(
        matrix=jnp.concatenate([matrix, jnp.zeros((pad_len, config.n_topics))]),
        ctx_bounds=jnp.concatenate([doc_bounds, jnp.zeros(pad_len)]),
        ctx_weights=ctx_weights,
        token_mask=token_mask,
    )[:len(doc_bounds)]
    assert_allclose(attn_fast, attn_primitive, rtol=1e-5, atol=1e-6)
    assert_allclose(attn_fast_padded, attn_primitive, rtol=1e-5, atol=1e-6)


def test_attn_transposed(phi, data, doc_bounds, config):
    pad_len = 50
    ctx_weights = get_context_weights_1d(
        ctx_len=config.ctx_len, gamma=config.gamma, self_aware=False
    )
    matrix = phi[data]
    token_mask = jnp.concatenate(
        [
            jnp.ones_like(doc_bounds),
            jnp.zeros(pad_len),
        ],
        dtype=jnp.bool_,
    )

    attn_primitive = calc_attn_transposed_primitive(
        matrix=matrix,
        ctx_bounds=doc_bounds,
        ctx_len=config.ctx_len,
        gamma=config.gamma,
    )
    attn_fast = calc_attn_transposed(
        matrix=matrix,
        ctx_bounds=doc_bounds,
        ctx_weights=ctx_weights,
        token_mask=token_mask[:len(doc_bounds)],
    )
    attn_fast_padded = calc_attn_transposed(
        matrix=jnp.concatenate([matrix, jnp.zeros((pad_len, config.n_topics))]),
        ctx_bounds=jnp.concatenate([doc_bounds, jnp.zeros(pad_len)]),
        ctx_weights=ctx_weights,
        token_mask=token_mask,
    )[:len(doc_bounds)]
    assert_allclose(attn_fast, attn_primitive, rtol=1e-5, atol=1e-6)
    assert_allclose(attn_fast_padded, attn_primitive, rtol=1e-5, atol=1e-6)


def test_attn_linear_invariant(doc_bounds, config):
    ctx_weights = get_context_weights_1d(
        ctx_len=config.ctx_len, gamma=config.gamma, self_aware=False
    )
    key = jax.random.key(seed=config.seed)
    x = jax.random.uniform(key, (config.n_words, config.vocab_size))
    y = jax.random.uniform(key, (config.n_words, config.vocab_size))

    attn_forward = calc_attn(
        matrix=x,
        ctx_bounds=doc_bounds,
        ctx_weights=ctx_weights,
        token_mask=jnp.ones_like(doc_bounds, dtype=jnp.bool_),
    )
    attn_backward = calc_attn_transposed(
        matrix=y,
        ctx_bounds=doc_bounds,
        ctx_weights=ctx_weights,
        token_mask=jnp.ones_like(doc_bounds, dtype=jnp.bool_),
    )
    left = attn_forward.T @ y
    right = x.T @ attn_backward
    assert_allclose(left, right, rtol=1e-5, atol=1e-6)
