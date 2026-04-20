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

    def compute_for_offset(d, w):
        idx = base_idx + d

        valid = (idx >= 0) & (idx < batch_size)
        idx_clipped = jnp.clip(idx, 0, batch_size - 1)

        gathered_matrix = matrix[idx_clipped]
        gathered_doc = doc_ids[idx_clipped]

        mask = valid & (gathered_doc == doc_ids)

        return mask, gathered_matrix, w

    mask, shifted, w = jax.vmap(compute_for_offset)(offsets, ctx_weights)

    mask_f = mask.astype(matrix.dtype)

    denom = jnp.sum(w[:, None] * mask_f, axis=0)
    inv_denom = jnp.where(denom > EPSILON, 1.0 / denom, 0.0)

    coeff = w[:, None] * mask_f * inv_denom

    out = jnp.einsum('kn,knh->nh', coeff, shifted)

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

    def compute_for_offset(d, w):
        idx = base_idx + d

        valid = (idx >= 0) & (idx < batch_size)
        idx_clipped = jnp.clip(idx, 0, batch_size - 1)

        same_doc = doc_ids[idx_clipped] == doc_ids

        mask = valid & same_doc
        return mask, w, idx_clipped

    mask, w, idx = jax.vmap(compute_for_offset)(offsets, ctx_weights)
    mask_f = mask.astype(matrix.dtype)

    denom = jnp.sum(w[:, None] * mask_f, axis=0)
    inv_denom = jnp.where(denom > EPSILON, 1.0 / denom, 0.0)

    coeff = w[:, None] * mask_f * inv_denom  # (2C + 1, I)
    gathered = matrix[idx]  # (2C + 1, I, T)
    out = jnp.sum(coeff[:, :, None] * gathered, axis=0)

    return out
