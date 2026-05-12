from __future__ import annotations

import jax
import jax.numpy as jnp

from cartm.aartm import AttentiveTopicModel
from cartm.core import EPSILON, norm, calc_attn


class AttentiveTopicModelNoNWT(AttentiveTopicModel):
    """
    Ablation of AttentiveTopicModel without the context-induced N_wt update term.
    """

    @staticmethod
    @jax.jit
    def _update_phi(
        grad_phi: jax.Array,
        n_wt: jax.Array,
        N_wt: jax.Array,
    ) -> jax.Array:
        n_w = jnp.sum(n_wt, axis=1, keepdims=True)
        safe_n_w = jnp.where(n_w > EPSILON, n_w, 1.0)
        coeff = n_wt / safe_n_w

        phi_new = n_wt + coeff * grad_phi
        phi_new = norm(phi_new, axis=1)
        return phi_new

    @staticmethod
    @jax.jit(static_argnames=("num_attn_passes",))
    def _step(
        batch: jax.Array,
        ctx_bounds: jax.Array,
        token_mask: jax.Array,
        phi: jax.Array,
        n_t: jax.Array,
        ctx_weights: jax.Array,
        num_attn_passes: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        p_it = norm(phi[batch], axis=1) * token_mask[:, None]

        for _ in range(num_attn_passes):
            theta = calc_attn(
                matrix=p_it,
                ctx_bounds=ctx_bounds,
                ctx_weights=ctx_weights,
                token_mask=token_mask,
            )
            p_it = norm(p_it * theta / (n_t + EPSILON), axis=1) * token_mask[:, None]

        n_t_new = jnp.sum(p_it, axis=0)
        n_wt = jax.ops.segment_sum(p_it, batch, phi.shape[0])
        N_wt = jnp.zeros_like(phi)

        return theta, n_t_new, n_wt, N_wt
