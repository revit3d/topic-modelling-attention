from typing import Callable

import jax
import jax.numpy as jnp

from cartm.model_base import ModelBase
from cartm.core import EPSILON, norm, calc_attn, calc_attn_transposed


class AttentiveTopicModel(ModelBase):
    """
    Topic model which uses local context of words.

    Phi matrix represents p(t|w) probability distribution.
    """

    def __init__(
        self,
        vocab_size: int,
        ctx_len: int,
        *,
        n_topics: int = 10,
        gamma: float = 0.6,
        self_aware_context: bool = False,
        regularizers: list = None,
        metrics: list = None,
    ):
        """
        Args:
            vocab_size: corpus vocabulary size, W.
            ctx_len: one-sided context size, C.
            n_topics: number of topics, T.
            gamma: parameter used for calculating weights of the word embeddings in the context.
            self_aware_context: whether to use the word itself in its context.
            regularizers: list of regularizations (see `add_regularization` method).
            metrics: list of metrics calculated on each step.

        Note:
            - Total context of a word on `i`-th index is ctx_len words to the left,\\
            `ctx_len` words to the right, and the word itself (if `self_aware_context` = True).
        """
        super().__init__(
            vocab_size=vocab_size,
            ctx_len=ctx_len,
            n_topics=n_topics,
            gamma=gamma,
            self_aware_context=self_aware_context,
            regularizers=regularizers,
            metrics=metrics,
        )

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
        n_w = jnp.sum(n_wt, axis=1, keepdims=True)      # (W, 1)
        safe_n_w = jnp.where(n_w > EPSILON, n_w, 1.0)

        ratio = p_it / (theta + EPSILON)  # (I, T)
        attn_t_ratio = calc_attn_transposed(
            matrix=ratio,
            ctx_bounds=ctx_bounds,
            ctx_weights=ctx_weights,
        )  # (I, T)
        N_wt = jax.ops.segment_sum(attn_t_ratio, batch, phi.shape[0])  # (W, T)

        coeff = n_wt / safe_n_w
        phi_new = n_wt + coeff * N_wt
        phi_new -= coeff * phi * grad_reg(phi)
        phi_new = norm(phi_new, axis=1)

        return phi_it, phi_new, theta, n_t_new

    def _init_state(self, *, seed: int, data_size: int):
        key = jax.random.key(seed)

        self.phi = jax.random.uniform(
            key=key,
            shape=(self.vocab_size, self.n_topics),
        )  # (W, T)
        self.phi = norm(self.phi, axis=1)

        self.n_t = jnp.ones((self.n_topics,), dtype=jnp.float32)

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
