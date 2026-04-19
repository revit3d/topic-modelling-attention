import jax
import jax.numpy as jnp


EPSILON = 1e-12


@jax.jit(static_argnames="axis")
def norm(x: jax.Array, axis: int = 0) -> jax.Array:
    # take x+ = max(x, 0) element-wise (perform projection on positive simplex)
    x = jnp.maximum(x, 0.0)
    res = jnp.sum(x, axis=axis, keepdims=True)
    safe_norm = jnp.where(res > EPSILON, res, 1.0)
    x = jnp.where(res > EPSILON, x / safe_norm, 0.0)
    return x


def get_context_weights_1d(ctx_len: int, gamma: float, self_aware: bool) -> jax.Array:
    # w_i = gamma * (1 - gamma)**i
    suffix_context_weights = (
        jnp.cumprod(jnp.full(ctx_len, (1.0 - gamma))) * gamma
    )  # (C, )
    prefix_context_weights = suffix_context_weights[::-1]  # (C, )
    self_context_weight = jnp.array([gamma * self_aware], dtype=jnp.float32)

    ctx_weights = jnp.concatenate(
        [
            prefix_context_weights,
            self_context_weight,
            suffix_context_weights,
        ]
    )
    return jnp.array(ctx_weights)  # (2C + 1, )


def _shift_1d(x: jax.Array, offset: int) -> jax.Array:
    n = x.shape[0]

    if offset == 0:
        return x

    if offset > 0:
        pad = jnp.zeros((offset,), dtype=x.dtype)
        return jnp.concatenate([x[offset:], pad], axis=0)

    k = -offset
    pad = jnp.zeros((k,), dtype=x.dtype)
    return jnp.concatenate([pad, x[: n - k]], axis=0)


def _shift_2d(x: jax.Array, offset: int) -> jax.Array:
    n, h = x.shape

    if offset == 0:
        return x

    if offset > 0:
        pad = jnp.zeros((offset, h), dtype=x.dtype)
        return jnp.concatenate([x[offset:], pad], axis=0)

    k = -offset
    pad = jnp.zeros((k, h), dtype=x.dtype)
    return jnp.concatenate([pad, x[: n - k]], axis=0)


def _valid_mask(length: int, offset: int) -> jax.Array:
    if offset == 0:
        return jnp.ones((length,), dtype=bool)

    if offset > 0:
        return jnp.concatenate(
            [
                jnp.ones((length - offset,), dtype=bool),
                jnp.zeros((offset,), dtype=bool),
            ],
            axis=0,
        )

    k = -offset
    return jnp.concatenate(
        [
            jnp.zeros((k,), dtype=bool),
            jnp.ones((length - k,), dtype=bool),
        ],
        axis=0,
    )


@jax.jit
def calc_attn(
    matrix: jax.Array,
    ctx_bounds: jax.Array,
    ctx_weights: jax.Array,
) -> jax.Array:
    batch_size, _ = matrix.shape
    ctx_len = (ctx_weights.shape[-1] - 1) // 2
    doc_ids = jnp.cumsum(ctx_bounds)

    offsets = range(-ctx_len, ctx_len + 1)
    denom = jnp.zeros((batch_size,), dtype=matrix.dtype)
    for k, d in enumerate(offsets):
        valid = _valid_mask(batch_size, d)
        same_doc = _shift_1d(doc_ids, d) == doc_ids
        mask = valid & same_doc
        denom = denom + ctx_weights[k] * mask.astype(matrix.dtype)

    inv_denom = jnp.where(denom > EPSILON, 1.0 / denom, 0.0)
    out = jnp.zeros_like(matrix)
    for k, d in enumerate(offsets):
        valid = _valid_mask(batch_size, d)
        same_doc = _shift_1d(doc_ids, d) == doc_ids
        mask = valid & same_doc

        coeff = ctx_weights[k] * mask.astype(matrix.dtype) * inv_denom  # (I, )
        shifted = _shift_2d(matrix, d)  # shifted[i] = matrix[i + d]

        out = out + coeff[:, None] * shifted

    return out


@jax.jit
def calc_attn_transposed(
    matrix: jax.Array,
    ctx_bounds: jax.Array,
    ctx_weights: jax.Array,
) -> jax.Array:
    batch_size, _ = matrix.shape
    ctx_len = (ctx_weights.shape[-1] - 1) // 2
    doc_ids = jnp.cumsum(ctx_bounds)

    offsets = range(-ctx_len, ctx_len + 1)
    denom = jnp.zeros((batch_size,), dtype=matrix.dtype)
    for k, d in enumerate(offsets):
        valid = _valid_mask(batch_size, d)
        same_doc = _shift_1d(doc_ids, d) == doc_ids
        mask = valid & same_doc
        denom = denom + ctx_weights[k] * mask.astype(matrix.dtype)

    inv_denom = jnp.where(denom > EPSILON, 1.0 / denom, 0.0)
    out = jnp.zeros_like(matrix)
    for k, d in enumerate(offsets):
        valid = _valid_mask(batch_size, d)
        same_doc = _shift_1d(doc_ids, d) == doc_ids
        mask = valid & same_doc

        coeff = ctx_weights[k] * mask.astype(matrix.dtype) * inv_denom  # (I, )
        contrib = coeff[:, None] * matrix
        out = out + _shift_2d(contrib, -d)

    return out
