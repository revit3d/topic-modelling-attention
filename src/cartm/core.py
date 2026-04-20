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


import jax
import jax.numpy as jnp

EPSILON = 1e-8


@jax.jit
def calc_attn(
    matrix: jax.Array,
    ctx_bounds: jax.Array,
    ctx_weights: jax.Array,
) -> jax.Array:
    batch_size, _ = matrix.shape
    ctx_len = (ctx_weights.shape[-1] - 1) // 2
    doc_ids = jnp.cumsum(ctx_bounds)
    offsets = jnp.arange(-ctx_len, ctx_len + 1)
    base_idx = jnp.arange(batch_size)

    idx = base_idx[None, :] + offsets[:, None]
    valid = (idx >= 0) & (idx < batch_size)
    idx_clipped = jnp.clip(idx, 0, batch_size - 1)

    shifted = matrix[idx_clipped]
    gathered_doc = doc_ids[idx_clipped]
    mask = valid & (gathered_doc == doc_ids[None, :])

    mask_f = mask.astype(matrix.dtype)
    coeff = ctx_weights[:, None] * mask_f

    denom = jnp.sum(coeff, axis=0)
    inv_denom = jnp.where(denom > EPSILON, 1.0 / denom, 0.0)

    coeff = coeff * inv_denom[None, :]
    out = jnp.sum(coeff[..., None] * shifted, axis=0)

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
    offsets = jnp.arange(-ctx_len, ctx_len + 1)
    base_idx = jnp.arange(batch_size)

    nbr_idx = base_idx[None, :] + offsets[:, None]
    nbr_valid = (nbr_idx >= 0) & (nbr_idx < batch_size)
    nbr_idx_safe = jnp.clip(nbr_idx, 0, batch_size - 1)

    shifted_doc_ids = doc_ids[nbr_idx_safe]
    same_doc = shifted_doc_ids == doc_ids[None, :]
    mask = nbr_valid & same_doc

    weights = ctx_weights[:, None]
    denom = jnp.sum(weights * mask.astype(matrix.dtype), axis=0)
    inv_denom = jnp.where(denom > EPSILON, 1.0 / denom, 0.0)

    coeff = weights * mask.astype(matrix.dtype) * inv_denom[None, :]  # (2C + 1, I)

    src_idx = base_idx[None, :] - offsets[:, None]
    src_valid = (src_idx >= 0) & (src_idx < batch_size)
    src_idx_safe = jnp.clip(src_idx, 0, batch_size - 1)

    gathered_matrix = matrix[src_idx_safe]  # (2C + 1, I, T)
    gathered_coeff = coeff[jnp.arange(coeff.shape[0])[:, None], src_idx_safe]
    gathered_coeff = gathered_coeff * src_valid.astype(matrix.dtype)

    out = jnp.sum(gathered_coeff[..., None] * gathered_matrix, axis=0)
    return out
