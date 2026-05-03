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
    @jax.jit
    def _update_phi(
        phi: jax.Array,
        grad_phi: jax.Array,
        n_wt: jax.Array,
    ) -> jax.Array:
        """Update phi_wt = p(w|t) matrix"""
        phi_new = n_wt + phi * grad_phi  # (W, T)
        phi_new = norm(phi_new, axis=0)  # (W, T)
        return phi_new

    @staticmethod
    @jax.jit(static_argnames=("num_attn_passes"))
    def _step(
        batch: jax.Array,
        ctx_bounds: jax.Array,
        phi: jax.Array,
        n_t: jax.Array,
        ctx_weights: jax.Array,
        num_attn_passes: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        p_it = norm(phi[batch] * n_t, axis=1)  # (I, T)

        for _ in range(num_attn_passes):
            # calculate theta_it = p(t|C_i) matrix
            theta = calc_attn(
                matrix=p_it,
                ctx_bounds=ctx_bounds,
                ctx_weights=ctx_weights,
            )  # (I, T)

            p_it = norm(p_it * theta / (n_t + EPSILON), axis=1)  # (I, T)

        n_t_new = jnp.sum(p_it, axis=0)  # (T,)
        n_wt = jax.ops.segment_sum(p_it, batch, phi.shape[0])  # (W, T)

        return theta, n_t_new, n_wt

    def _batched_step_wrapper(
        self,
        *,
        batches: Iterable[tuple[jax.Array, jax.Array]],
        ctx_weights: jax.Array,
        grad_reg: Callable,
        num_attn_passes: int,
        lr: float,
        num_batches_before_update: int,
    ) -> tuple[jax.Array, jax.Array]:
        if num_batches_before_update <= 0:
            lr = 1.0

        phi_new = self.phi
        n_t_new = self.n_t
        n_t_total = jnp.zeros_like(self.n_t)
        n_wt_total = jnp.zeros_like(self.phi)
        batch_counter = 0

        def reset_totals():
            nonlocal n_t_total, n_wt_total, batch_counter

            n_t_total = jnp.zeros_like(self.n_t)
            n_wt_total = jnp.zeros_like(self.phi)
            batch_counter = 0

        def flush():
            nonlocal phi_new, n_t_new

            grad_phi = grad_reg(phi_new)
            phi_step = self._update_phi(
                phi=phi_new, grad_phi=grad_phi, n_wt=n_wt_total
            )
            phi_new = phi_new * (1.0 - lr) + phi_step * lr
            n_t_new = n_t_new * (1.0 - lr) + n_t_total * lr

        for batch, ctx_bounds_batch in batches:
            theta, n_t_step, n_wt_step = self._step(
                batch=batch,
                ctx_bounds=ctx_bounds_batch,
                phi=phi_new,
                n_t=n_t_new,
                ctx_weights=ctx_weights,
                num_attn_passes=num_attn_passes,
            )
            n_t_total += n_t_step
            n_wt_total += n_wt_step

            self._calc_metrics_batch(batch=batch, phi=phi_new, theta=theta)

            batch_counter += 1
            if batch_counter == num_batches_before_update:
                flush()
                reset_totals()

        if batch_counter > 0:
            flush()
        return phi_new, n_t_new

    def _init_state(self, *, seed: int):
        key = jax.random.key(seed)

        self.phi = jax.random.uniform(
            key=key,
            shape=(self.vocab_size, self.n_topics),
        )
        self.phi = norm(self.phi, axis=0)
        self.n_t = jnp.ones(self.n_topics, dtype=jnp.float32)

    def _calc_metrics_batch(
        self,
        *,
        batch: jax.Array,
        phi: jax.Array,
        theta: jax.Array,
    ):
        if len(self._metrics) == 0:
            return

        for metric in self._metrics.values():
            metric.partial_update(
                batch=batch,
                phi=phi,
                theta=theta,
            )
