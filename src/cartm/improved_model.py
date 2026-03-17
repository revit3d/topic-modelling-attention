from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

from cartm.model import ContextTopicModel


class AttentiveTopicModel(ContextTopicModel):
    @partial(jax.jit, static_argnums=0)
    def _calc_phi_hatch(self, *, phi: jax.Array, batch: jax.Array) -> jax.Array:
        return phi[:, batch]  # (T, I)

    @partial(jax.jit, static_argnums=0)
    def _calc_theta(self, *, phi_ti: jax.Array, ctx_bounds: jax.Array) -> jax.Array:
        theta_it = self._calc_attn(matrix=phi_ti.T, ctx_bounds=ctx_bounds)  # (I, T)
        return theta_it.T

    @partial(jax.jit, static_argnums=0)
    def _calc_p_ti(
        self,
        *,
        phi_ti: jax.Array,
        theta: jax.Array,
        n_t: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        p_ti = self._norm(phi_ti * theta / (n_t[:, None] + self._eps), axis=0)
        return p_ti, phi_ti

    @partial(jax.jit, static_argnums=0)
    def _calc_N_tw(
        self,
        *,
        phi: jax.Array,
        p_ti: jax.Array,
        theta: jax.Array,
        batch: jax.Array,
        ctx_bounds: jax.Array,
    ) -> jax.Array:
        vocab_size = phi.shape[1]
        ohe = jnp.eye(vocab_size)[batch]  # (I, W)
        q_iw = self._calc_attn(matrix=ohe, ctx_bounds=ctx_bounds)  # (W, I)
        N_tw = (p_ti / (theta + self._eps)) @ q_iw
        return N_tw

    @partial(jax.jit, static_argnums=0, static_argnames="grad_reg")
    def _calc_phi(
        self,
        *,
        batch: jax.Array,
        phi: jax.Array,
        p_ti: jax.Array,
        N_tw: jax.Array,
        grad_reg: Callable,
    ) -> jax.Array:
        n_wt = jnp.add.at(jnp.zeros_like(phi.T), batch, p_ti.T, inplace=False)
        phi_new = n_wt.T + phi * N_tw
        phi_new -= phi * grad_reg(phi)
        phi_new = self._norm(phi_new, axis=0)
        return phi_new

    @partial(jax.jit, static_argnums=0, static_argnames="grad_reg")
    def _step(
        self,
        *,
        batch: jax.Array,
        ctx_bounds: jax.Array,
        phi: jax.Array,
        n_t: jax.Array,
        grad_reg: Callable,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        phi = phi.T

        # calculate phi' (words -> topics) matrix (phi with old p_{ti})
        phi_ti = self._calc_phi_hatch(phi=phi, batch=batch)  # (T, I)

        # calculate theta_it = p(t|C_i) matrix
        theta = self._calc_theta(phi_ti=phi_ti, ctx_bounds=ctx_bounds)  # (T, I)

        # update p_{ti} - topic probability distribution for i-th context
        # phi_it = p(C_i|t)
        p_ti, _ = self._calc_p_ti(phi_ti=phi_ti, theta=theta, n_t=n_t)  # (T, I)

        # update n_{t} - topic probability distribution
        n_t_new = self._calc_n_t(p_ti=p_ti.T)  # (T, )

        # update N_tw matrix
        N_tw = self._calc_N_tw(
            phi=phi,
            p_ti=p_ti,
            theta=theta,
            batch=batch,
            ctx_bounds=ctx_bounds,
        )  # (T, W)
        # N_tw = jnp.zeros_like(phi)

        # update phi_wt = p(w|t) matrix
        phi_new = self._calc_phi(
            batch=batch,
            phi=phi,
            p_ti=p_ti,
            N_tw=N_tw,
            grad_reg=grad_reg,
        )  # (T, W)

        return phi_ti.T, phi_new.T, theta.T, n_t_new
