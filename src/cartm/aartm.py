from typing import Callable, Iterable

import jax
import jax.numpy as jnp

from cartm.model_base import ModelBase
from cartm.core import EPSILON, norm, calc_attn, calc_attn_transposed


class AttentiveTopicModel(ModelBase):
    """
    Topic model which uses local context of words.

    Phi matrix represents p(t|w) probability distribution.
    """

    @staticmethod
    @jax.jit
    def _update_phi(
        grad_phi: jax.Array,
        n_wt: jax.Array,
        N_wt: jax.Array,
    ) -> jax.Array:
        """Update phi_wt = p(t|w) matrix"""
        n_w = jnp.sum(n_wt, axis=1, keepdims=True)
        safe_n_w = jnp.where(n_w > EPSILON, n_w, 1.0)
        coeff = n_wt / safe_n_w

        phi_new = n_wt + coeff * N_wt + coeff * grad_phi
        phi_new = norm(phi_new, axis=1)
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
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        p_it = norm(phi[batch], axis=1)  # (I, T)

        for _ in range(num_attn_passes):
            theta = calc_attn(
                matrix=p_it,
                ctx_bounds=ctx_bounds,
                ctx_weights=ctx_weights,
            )  # (I, T)

            p_it = norm(p_it * theta / (n_t + EPSILON), axis=1)  # (I, T)

        n_t_new = jnp.sum(p_it, axis=0)  # (T,)
        n_wt = jax.ops.segment_sum(p_it, batch, phi.shape[0])  # (W, T)

        ratio = p_it / (theta + EPSILON)  # (I, T)
        attn_t_ratio = calc_attn_transposed(
            matrix=ratio,
            ctx_bounds=ctx_bounds,
            ctx_weights=ctx_weights,
        )  # (I, T)
        N_wt = jax.ops.segment_sum(attn_t_ratio, batch, phi.shape[0])  # (W, T)

        return theta, n_t_new, n_wt, N_wt

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
        N_wt_total = jnp.zeros_like(self.phi)
        batch_counter = 0

        def reset_totals():
            nonlocal n_t_total, n_wt_total, N_wt_total, batch_counter

            n_t_total = jnp.zeros_like(self.n_t)
            n_wt_total = jnp.zeros_like(self.phi)
            N_wt_total = jnp.zeros_like(self.phi)
            batch_counter = 0

        def flush():
            nonlocal phi_new, n_t_new

            grad_phi = grad_reg(phi_new)
            phi_step = self._update_phi(
                grad_phi=grad_phi, n_wt=n_wt_total, N_wt=N_wt_total
            )
            phi_new = phi_new * (1.0 - lr) + phi_step * lr
            n_t_new = n_t_new * (1.0 - lr) + n_t_total * lr

        for batch, ctx_bounds_batch in batches:
            theta, n_t_step, n_wt_step, N_wt_step = self._step(
                batch=batch,
                ctx_bounds=ctx_bounds_batch,
                phi=phi_new,
                n_t=n_t_new,
                ctx_weights=ctx_weights,
                num_attn_passes=num_attn_passes,
            )
            n_t_total += n_t_step
            n_wt_total += n_wt_step
            N_wt_total += N_wt_step

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
        )  # (W, T)
        self.phi = norm(self.phi, axis=1)

        self.n_t = jnp.ones(self.n_topics, dtype=jnp.float32)

    def _compose_regularizations(self):
        regs = self._regularizations.values()
        reg_grad = jax.grad(
            lambda x: sum(
                [1.0,]
                + [reg(self.renormalize_phi(p_w=self.p_w, phi=x)) for reg in regs]
            )
        )
        return jax.jit(reg_grad)

    @staticmethod
    @jax.jit
    def renormalize_phi(p_w: jax.Array, phi: jax.Array):
        """
        phi = p(t|w) -> phi = p(w|t)
        """
        p_t = jnp.sum(phi * p_w[:, None], axis=0)  # (T, )
        phi_wt = phi * p_w[:, None] / (p_t[None, :] + EPSILON)  # (W, T)
        return phi_wt

    def _calc_metrics_batch(
        self,
        *,
        batch: jax.Array,
        phi: jax.Array,
        theta: jax.Array,
    ):
        if len(self._metrics) == 0:
            return

        phi_wt = self.renormalize_phi(p_w=self.p_w, phi=phi)
        for metric in self._metrics.values():
            metric.partial_update(
                batch=batch,
                phi=phi_wt,
                theta=theta,
            )
