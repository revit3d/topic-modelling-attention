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
    @jax.jit(static_argnames=("grad_reg"))
    def _update_phi(
        phi: jax.Array,
        n_wt: jax.Array,
        N_wt: jax.Array,
        grad_reg: Callable,
    ) -> jax.Array:
        """Update phi_wt = p(t|w) matrix"""
        n_w = jnp.sum(n_wt, axis=1, keepdims=True)
        safe_n_w = jnp.where(n_w > EPSILON, n_w, 1.0)
        coeff = n_wt / safe_n_w

        phi_new = n_wt + coeff * N_wt - coeff * grad_reg(phi)
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

        phi_new = AttentiveTopicModel._update_phi(
            phi=phi, n_wt=n_wt, N_wt=N_wt, grad_reg=grad_reg
        )

        return phi_it, phi_new, theta, n_t_new, n_wt, N_wt

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
        phi_new = phi.copy()
        n_t_new = n_t.copy()
        n_t_total = jnp.zeros_like(n_t)
        n_wt_total = jnp.zeros_like(phi)
        N_wt_total = jnp.zeros_like(phi)
        batch_counter = 0

        for i, (batch, ctx_bounds_batch) in enumerate(batches):
            phi_it_step, phi_step, theta_step, n_t_step, n_wt_step, N_wt_step = self._step(
                batch=batch,
                ctx_bounds=ctx_bounds_batch,
                phi=phi,
                n_t=n_t,
                ctx_weights=ctx_weights,
                grad_reg=grad_reg,
                num_attn_passes=num_attn_passes,
            )
            n_t_total += n_t_step
            n_wt_total += n_wt_step
            N_wt_total += N_wt_step
            phi_it.append(phi_it_step)
            theta.append(theta_step)
            batch_all.append(batch)

            batch_counter += 1
            if (
                num_batches_before_update > 0
                and (
                    batch_counter == num_batches_before_update
                    or i == len(batches) - 1
                )
            ):
                phi_step = self._update_phi(
                    phi=phi, n_wt=n_wt_total, N_wt=N_wt_total, grad_reg=grad_reg
                )
                phi_new = phi_new * (1 - lr) + phi_step * lr
                n_t_new = n_t_new * (1 - lr) + n_t_total * lr
                n_t_total = jnp.zeros_like(n_t)
                n_wt_total = jnp.zeros_like(phi)
                N_wt_total = jnp.zeros_like(phi)
                batch_counter = 0

        phi_it = jnp.concatenate(phi_it)
        theta = jnp.concatenate(theta)
        batch_all = jnp.concatenate(batch_all)

        if num_batches_before_update <= 0:
            phi_new = self._update_phi(
                phi=phi, n_wt=n_wt_total, N_wt=N_wt_total, grad_reg=grad_reg
            )
            n_t_new = n_t_total

        return phi_it, phi_new, theta, n_t_new, batch_all

    def _init_state(self, *, seed: int):
        key = jax.random.key(seed)

        self.phi = jax.random.uniform(
            key=key,
            shape=(self.vocab_size, self.n_topics),
        )  # (W, T)
        self.phi = norm(self.phi, axis=1)

        self.n_t = jnp.ones(self.n_topics, dtype=jnp.float32)

    def renormalize_phi(self, *, batch: jax.Array, phi: jax.Array):
        """
        phi = p(t|w) -> phi = p(w|t)
        """
        n_w = jnp.bincount(batch, length=self.vocab_size)
        total = jnp.maximum(jnp.sum(n_w), 1)
        p_w = n_w / total  # (W,)
        p_t = jnp.sum(phi * p_w[:, None], axis=0)  # (T, )
        phi_wt = phi * p_w[:, None] / (p_t[None, :] + EPSILON)  # (W, T)
        phi_it = phi_wt[batch]  # (I, T)
        return phi_wt, phi_it

    def _calc_metrics(
        self,
        *,
        batch: jax.Array,
        phi_it: jax.Array,
        phi_wt: jax.Array,
        theta: jax.Array,
        verbose: int,
    ):
        if len(self._metrics) == 0:
            return

        if verbose > 1:
            print("  Metrics:")

        phi_wt_renorm, phi_it_renorm = self.renormalize_phi(batch=batch, phi=phi_wt)
        for tag, metric in self._metrics.items():
            value = metric(
                phi_it=phi_it_renorm,
                phi_wt=phi_wt_renorm,
                theta=theta,
            )
            if verbose > 1:
                print(f"    {tag}: {value:.04f}")
