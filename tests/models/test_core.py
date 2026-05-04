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
    key_x, key_y = jax.random.split(key)
    x = jax.random.uniform(key_x, (config.n_words, config.vocab_size))
    y = jax.random.uniform(key_y, (config.n_words, config.vocab_size))

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


def test_norm_clips_negatives_and_handles_zero_columns_axis0():
    x = jnp.array(
        [
            [-1.0, 0.0, 2.0],
            [3.0, 0.0, -4.0],
        ]
    )

    expected = jnp.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )

    actual = norm(x, axis=0)

    assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    assert not jnp.any(jnp.isnan(actual))


def test_norm_clips_negatives_and_handles_zero_rows_axis1():
    x = jnp.array(
        [
            [-1.0, -2.0, -3.0],
            [2.0, 2.0, 0.0],
        ]
    )

    expected = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
        ]
    )

    actual = norm(x, axis=1)

    assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    assert not jnp.any(jnp.isnan(actual))


def test_context_weights_without_self_awareness():
    weights = get_context_weights_1d(ctx_len=2, gamma=0.5, self_aware=False)

    expected = jnp.array([0.125, 0.25, 0.0, 0.25, 0.125])

    assert_allclose(weights, expected, rtol=1e-6, atol=1e-7)


def test_context_weights_with_self_awareness():
    weights = get_context_weights_1d(ctx_len=2, gamma=0.5, self_aware=True)

    expected = jnp.array([0.125, 0.25, 0.5, 0.25, 0.125])

    assert_allclose(weights, expected, rtol=1e-6, atol=1e-7)


def test_calc_attn_respects_document_boundaries():
    matrix = jnp.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 3.0],
        ]
    )
    ctx_bounds = jnp.array([False, True, False])
    token_mask = jnp.array([True, True, True])
    ctx_weights = get_context_weights_1d(ctx_len=1, gamma=0.5, self_aware=False)

    actual = calc_attn(
        matrix=matrix,
        ctx_bounds=ctx_bounds,
        ctx_weights=ctx_weights,
        token_mask=token_mask,
    )

    expected = jnp.array(
        [
            [0.0, 0.0],  # token 0 has no valid context in its document
            [2.0, 3.0],  # token 1 can see token 2 only
            [0.0, 1.0],  # token 2 can see token 1 only
        ]
    )

    assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_calc_attn_with_self_awareness():
    matrix = jnp.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    ctx_bounds = jnp.array([False, False, False])
    token_mask = jnp.array([True, True, True])
    ctx_weights = get_context_weights_1d(ctx_len=1, gamma=0.5, self_aware=True)

    actual = calc_attn(
        matrix=matrix,
        ctx_bounds=ctx_bounds,
        ctx_weights=ctx_weights,
        token_mask=token_mask,
    )

    expected = jnp.array(
        [
            [2.0 / 3.0, 1.0 / 3.0],
            [0.5, 0.75],
            [2.0 / 3.0, 1.0],
        ]
    )

    assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_calc_attn_and_transposed_are_adjoint_with_padding_and_self_awareness():
    key = jax.random.key(42)
    key_x, key_y = jax.random.split(key)

    n_words = 8
    n_topics = 4
    pad_len = 3

    x = jax.random.uniform(key_x, shape=(n_words + pad_len, n_topics))
    y = jax.random.uniform(key_y, shape=(n_words + pad_len, n_topics))

    token_mask = jnp.array([True] * n_words + [False] * pad_len)
    ctx_bounds = jnp.array(
        [False, False, True, False, False, True, False, False, False, False, False]
    )
    ctx_weights = get_context_weights_1d(ctx_len=2, gamma=0.4, self_aware=True)

    forward = calc_attn(
        matrix=x,
        ctx_bounds=ctx_bounds,
        ctx_weights=ctx_weights,
        token_mask=token_mask,
    )
    backward = calc_attn_transposed(
        matrix=y,
        ctx_bounds=ctx_bounds,
        ctx_weights=ctx_weights,
        token_mask=token_mask,
    )

    left = forward.T @ y
    right = x.T @ backward

    assert_allclose(left, right, rtol=1e-5, atol=1e-6)
