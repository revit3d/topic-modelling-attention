from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from cartm.aartm import AttentiveTopicModel
from cartm.core import EPSILON, norm, calc_attn


class AttentiveTopicModelNoNWT(AttentiveTopicModel):
    """
    Ablation of AttentiveTopicModel without the context-induced N_wt update term.
    This is the most important ablation for the paper.
    """

    @staticmethod
    @jax.jit(static_argnames=("grad_reg",))
    def _update_phi(
        phi: jax.Array,
        n_wt: jax.Array,
        N_wt: jax.Array,
        grad_reg: Callable,
    ) -> jax.Array:
        n_w = jnp.sum(n_wt, axis=1, keepdims=True)
        safe_n_w = jnp.where(n_w > EPSILON, n_w, 1.0)
        coeff = n_wt / safe_n_w

        phi_new = n_wt - coeff * phi * grad_reg(phi)
        phi_new = norm(phi_new, axis=1)
        return phi_new

    @staticmethod
    @jax.jit(static_argnames=("grad_reg", "num_attn_passes"))
    def _step(
        batch: jax.Array,
        ctx_bounds: jax.Array,
        phi: jax.Array,
        n_t: jax.Array,
        ctx_weights: jax.Array,
        grad_reg: Callable,
        num_attn_passes: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        phi_it = phi[batch]
        p_it = norm(phi_it, axis=1)  # (I, T)

        for _ in range(num_attn_passes):
            theta = calc_attn(
                matrix=p_it,
                ctx_bounds=ctx_bounds,
                ctx_weights=ctx_weights,
            )
            p_it = norm(p_it * theta / (n_t + EPSILON), axis=1)

        n_t_new = jnp.sum(p_it, axis=0)
        n_wt = jax.ops.segment_sum(p_it, batch, phi.shape[0])
        N_wt = jnp.zeros_like(phi)

        phi_new = AttentiveTopicModelNoNWT._update_phi(
            phi=phi,
            n_wt=n_wt,
            N_wt=N_wt,
            grad_reg=grad_reg,
        )

        return phi_it, phi_new, theta, n_t_new, n_wt, N_wt
