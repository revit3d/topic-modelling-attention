import jax
import jax.numpy as jnp


EPSILON = 1e-12


@jax.jit(static_argnames="axis")
def norm(x: jax.Array, axis: int = 0) -> jax.Array:
    # take x+ = max(x, 0) element-wise (perform projection on positive simplex)
    x = jnp.maximum(x, 0.0)
    norm = jnp.sum(x, axis=axis, keepdims=True)
    safe_norm = jnp.where(norm > EPSILON, norm, 1.0)
    x = jnp.where(norm > EPSILON, x / safe_norm, 0.0)
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
def doc_ids_from_bounds(matrix: jax.Array, ctx_bounds: jax.Array) -> jax.Array:
    batch_size = matrix.shape[0]
    doc_starts = jnp.zeros(batch_size, dtype=jnp.int32)
    doc_starts = doc_starts.at[ctx_bounds[:-1]].set(1)
    return jnp.cumsum(doc_starts)


@jax.jit
def calc_attn(
    matrix: jax.Array,
    ctx_bounds: jax.Array,
    ctx_weights: jax.Array,
) -> jax.Array:
    ctx_len = (ctx_weights.shape[-1] - 1) // 2
    batch_size = matrix.shape[0]
    doc_ids = doc_ids_from_bounds(matrix, ctx_bounds)
    offsets = jnp.arange(-ctx_len, ctx_len + 1, dtype=jnp.int32)

    def compute_attn_i(i):
        src_idx = i + offsets
        valid_pos = (src_idx >= 0) & (src_idx < batch_size)
        clipped_idx = jnp.clip(src_idx, 0, batch_size - 1)
        same_doc = doc_ids[clipped_idx] == doc_ids[i]
        mask = valid_pos & same_doc

        weights = ctx_weights * mask
        norm = jnp.sum(weights, axis=-1, keepdims=True)
        safe_norm = jnp.where(norm > EPSILON, norm, 1.0)
        weights = jnp.where(norm > EPSILON, weights / safe_norm, 0.0)

        window = matrix[clipped_idx]  # (2C + 1, H)
        return jnp.dot(weights, window)  # (H, )

    return jax.vmap(compute_attn_i)(jnp.arange(batch_size))


@jax.jit
def calc_attn_transposed(
    matrix: jax.Array,
    ctx_bounds: jax.Array,
    ctx_weights: jax.Array,
) -> jax.Array:
    batch_size, _ = matrix.shape
    ctx_len = (ctx_weights.shape[-1] - 1) // 2
    doc_ids = doc_ids_from_bounds(matrix, ctx_bounds)
    offsets = jnp.arange(-ctx_len, ctx_len + 1, dtype=jnp.int32)

    def compute_window(i):
        src_idx = i + offsets
        valid_pos = (src_idx >= 0) & (src_idx < batch_size)
        clipped_idx = jnp.clip(src_idx, 0, batch_size - 1)
        same_doc = doc_ids[clipped_idx] == doc_ids[i]
        mask = valid_pos & same_doc

        weights = ctx_weights * mask
        norm = jnp.sum(weights, axis=-1, keepdims=True)
        safe_norm = jnp.where(norm > EPSILON, norm, 1.0)
        weights = jnp.where(norm > EPSILON, weights / safe_norm, 0.0)

        return clipped_idx, weights, mask

    all_idx, all_weights, all_masks = jax.vmap(compute_window)(jnp.arange(batch_size))
    contrib = all_weights[..., None] * matrix[:, None, :]  # (I, 2C + 1, H)
    contrib = contrib * all_masks[..., None]  # mask invalid clipped positions

    out = jnp.zeros_like(matrix).at[all_idx].add(contrib)
    return out
