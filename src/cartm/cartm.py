from typing import Callable

import jax
import jax.numpy as jnp

from cartm.model_base import ModelBase
from cartm.core import EPSILON, norm, calc_attn


class ContextTopicModel(ModelBase):
    """
    Topic model which uses local context of words.

    Phi matrix represents p(w|t) probability distribution.
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
            eps: parameter set for balance between numerical stability and precision.
            learnable_context: allow gradient optimization over the context weights.

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
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
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

        # update phi_wt = p(w|t) matrix
        phi_new = jax.ops.segment_sum(p_it, batch, phi.shape[0])
        phi_new -= phi * grad_reg(phi)  # (W, T)
        phi_new = norm(phi_new, axis=0)  # (W, T)

        return phi_it, phi_new, theta, n_t_new

    def _init_state(self, *, seed: int, data_size: int):
        key = jax.random.key(seed)

        self.phi = jax.random.uniform(
            key=key,
            shape=(self.vocab_size, self.n_topics),
        )  # (W, T)
        self.phi = norm(self.phi, axis=0)
        self.n_t = jnp.full(
            shape=(self.n_topics,),
            fill_value=data_size / self.n_topics,
        )  # (T, )
