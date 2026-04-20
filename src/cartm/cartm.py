from typing import Callable, Iterable

import jax
import jax.numpy as jnp

from cartm.model_base import ModelBase
from cartm.core import EPSILON, norm, calc_attn


class ContextTopicModel(ModelBase):
    """
    Topic model which uses local context of words.

    Phi matrix represents p(w|t) probability distribution.
    """

    @staticmethod
    @jax.jit(static_argnames=("grad_reg"))
    def _update_phi(
        phi: jax.Array,
        p_it: jax.Array,
        batch: jax.Array,
        grad_reg: Callable,
    ) -> jax.Array:
        """Update phi_wt = p(w|t) matrix"""
        phi_new = jax.ops.segment_sum(p_it, batch, phi.shape[0])
        phi_new -= phi * grad_reg(phi)  # (W, T)
        phi_new = norm(phi_new, axis=0)  # (W, T)
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
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        # phi_it = p(C_i|t)
        phi_it = phi[batch]

        # calculate phi' (words -> topics) matrix (phi with old p_{ti})
        p_it = norm(phi_it * n_t, axis=1)  # (I, T)

        for _ in range(num_attn_passes):
            # calculate theta_it = p(t|C_i) matrix
            theta = calc_attn(
                matrix=p_it,
                ctx_bounds=ctx_bounds,
                ctx_weights=ctx_weights,
            )  # (I, T)

            # update p_{ti} - topic probability distribution for i-th context
            p_it = norm(p_it * theta / (n_t + EPSILON), axis=1)  # (I, T)

        # update n_{t} - topic probability distribution
        n_t_new = jnp.sum(p_it, axis=0)  # (T, )

        phi_new = ContextTopicModel._update_phi(
            phi=phi, p_it=p_it, batch=batch, grad_reg=grad_reg
        )

        return phi_it, phi_new, theta, n_t_new, p_it

    def _batched_step_wrapper(
        self,
        *,
        batches: Iterable[tuple[jax.Array, jax.Array]],
        phi: jax.Array,
        n_t: jax.Array,
        ctx_weights: jax.Array,
        grad_reg: Callable,
        num_attn_passes: int,
        lr: float,
        num_batches_before_update: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        phi_it = []
        theta = []
        batch_all = []
        p_it = []
        phi_new = phi.copy()
        n_t_new = n_t.copy()
        n_t_total = jnp.zeros_like(n_t)
        batch_counter = 0

        for i, (batch, ctx_bounds_batch) in enumerate(batches):
            phi_it_step, phi_step, theta_step, n_t_step, p_it_step = self._step(
                batch=batch,
                ctx_bounds=ctx_bounds_batch,
                phi=phi,
                n_t=n_t,
                ctx_weights=ctx_weights,
                grad_reg=grad_reg,
                num_attn_passes=num_attn_passes,
            )
            n_t_total += n_t_step
            phi_it.append(phi_it_step)
            theta.append(theta_step)
            batch_all.append(batch)
            p_it.append(p_it_step)

            batch_counter += 1
            if (
                num_batches_before_update > 0
                and (
                    batch_counter == num_batches_before_update
                    or i == len(batches) - 1
                )
            ):
                batch_total = jnp.concatenate(batch_all[-batch_counter:])
                p_it_total = jnp.concatenate(p_it[-batch_counter:])
                phi_step = self._update_phi(
                    phi=phi, p_it=p_it_total, batch=batch_total, grad_reg=grad_reg
                )
                phi_new = phi_new * (1 - lr) + phi_step * lr
                n_t_new = n_t_new * (1 - lr) + n_t_total * lr
                n_t_total = jnp.zeros_like(n_t)
                batch_counter = 0

        phi_it = jnp.concatenate(phi_it)
        theta = jnp.concatenate(theta)
        batch_all = jnp.concatenate(batch_all)

        if num_batches_before_update <= 0:
            p_it = jnp.concatenate(p_it)
            phi_new = self._update_phi(
                phi=phi, p_it=p_it, batch=batch_all, grad_reg=grad_reg
            )
            n_t_new = n_t_total

        return phi_it, phi_new, theta, n_t_new, batch_all

    def _init_state(self, *, seed: int):
        key = jax.random.key(seed)

        self.phi = jax.random.uniform(
            key=key,
            shape=(self.vocab_size, self.n_topics),
        )
        self.phi = norm(self.phi, axis=0)
        self.n_t = jnp.ones(self.n_topics, dtype=jnp.float32)
