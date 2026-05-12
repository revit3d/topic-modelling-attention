from abc import ABC, abstractmethod
from typing import Iterable, Callable

import jax
import jax.numpy as jnp

from cartm.core import get_context_weights_1d
from cartm.regularization import Regularization
from cartm.metrics import Metric


class ModelBase(ABC):
    def __init__(
        self,
        vocab_size: int,
        *,
        n_topics: int = 10,
        ctx_len: int = 10,
        gamma: float = 0.1,
        self_aware_context: bool = False,
        regularizers: list[Regularization] | None = None,
        metrics: list[Metric] | None = None,
    ):
        """
        Args:
            vocab_size: corpus vocabulary size, W.
            n_topics: number of topics, T.
            ctx_len: one-sided context size, C.
            gamma: parameter used for calculating weights of the word embeddings in the context.
            self_aware_context: whether to use the word itself in its context.
            regularizers: list of regularizations (see `add_regularization` method).
            metrics: list of metrics calculated on each step.

        Note:
            - Total context of a word on `i`-th index is ctx_len words to the left,\\
            `ctx_len` words to the right, and the word itself (if `self_aware_context` = True).
        """
        self.vocab_size = vocab_size
        self.ctx_len = ctx_len
        self.n_topics = n_topics

        self._gamma = gamma
        self._self_aware_context = self_aware_context

        self.context_weights = get_context_weights_1d(
            ctx_len=ctx_len, gamma=gamma, self_aware=self_aware_context
        )
        self.phi = None
        self.n_t = None

        self._regularizations = {}
        if regularizers is not None:
            for regularization in regularizers:
                self.add_regularization(regularization)

        self._metrics = {}
        if metrics is not None:
            for metric in metrics:
                self.add_metric(metric)

    @abstractmethod
    def _init_state(self, *, seed: int):
        pass

    @staticmethod
    @abstractmethod
    @jax.jit
    def _step(
        batch: jax.Array,
        ctx_bounds: jax.Array,
        token_mask: jax.Array,
        phi: jax.Array,
        n_t: jax.Array,
        ctx_weights: jax.Array,
        num_attn_passes: int,
    ) -> tuple:
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def _calc_metrics_batch(
        self,
        *,
        batch: jax.Array,
        phi: jax.Array,
        theta: jax.Array,
    ):
        pass

    def add_regularization(self, regularization: Regularization):
        """
        Add a regularization to the model.

        Note:
        - `regularization` has to be a child of base `Regularization` class.
        """
        if not isinstance(regularization, Regularization):
            raise TypeError(
                f"Regularization has to be a subclass of "
                f"the Regularization base class, got type {type(regularization)}"
            )
        self._regularizations[regularization.tag] = regularization

    def add_metric(self, metric: Metric):
        """
        Add a metric to the model.

        Note:
        - `metric` has to be a child of base `Metric` class.
        """
        if not isinstance(metric, Metric):
            raise TypeError(
                f"Metric has to be a subclass of "
                f"the Metric base class, got type {type(metric)}"
            )
        self._metrics[metric.tag] = metric

    def remove_regularization(self, tag: str):
        """Remove the regularization with specified tag."""
        try:
            self._regularizations.pop(tag)
        except KeyError:
            print(
                f"Regularization with tag {tag} is not present. "
                f"Did you mean to use remove_metric?"
            )

    def remove_metric(self, tag: str):
        """Remove the metric with the specified tag."""
        try:
            self._metrics.pop(tag)
        except KeyError:
            print(
                f"Metric with tag {tag} is not present. "
                f"Did you mean to use remove_regularization?"
            )

    def _compose_regularizations(self):
        regs = self._regularizations.values()
        reg_grad = jax.grad(
            lambda x: sum([1.0,] + [reg(x) for reg in regs])
        )
        return jax.jit(reg_grad)

    def _flush_metrics(
        self,
        *,
        verbose: int,
    ):
        if len(self._metrics) == 0:
            return

        if verbose > 1:
            print("  Metrics:")

        for tag, metric in self._metrics.items():
            value = metric.flush()
            if verbose > 1:
                print(f"    {tag}: {value:.04f}")

    def fit(
        self,
        batches: Iterable[tuple[jax.Array, jax.Array, jax.Array]],
        *,
        lr: float = 0.1,
        num_batches_before_update: int = -1,
        num_attn_passes: int = 1,
        max_iter: int = 1000,
        tol: float = 1e-3,
        verbose: int = 0,
        seed: int = 0,
    ):
        """
        Fit the model with the corpus of documents. Note that default model
        behavior is to accumulate statistics across all batches and then
        update state once per corpus pass. If you want to update state every
        n batches, see `num_batches_before_update` and `lr` parameters.

        Args:
            batches: Iterable returning tuples (data_batch, ctx_bounds_batch, token_mask), where
                - data_batch is an array of shape (I, ), containing tokenized words
                    of each document.
                - ctx_bounds_batch is an array of shape (B,) containing bounds for context.
                    Words beyond the bound are ignored in the context.
                - token_mask is an array of shape (B,) containing mask of valid tokens in batch
            lr: coefficient for updating phi in EMA mode:
                phi = phi_prev * (1 - lr) + phi_new * lr
            num_batches_before_update: if positive, batched algorithm updates phi
                with EMA logic, phi_new is calculated from statistics accumulated on
                num_batches_before_update batches.
            num_attn_passes: number of E-steps on each iteration.
            max_iter: max number of iterations.
            tol: early stopping threshold.
            verbose: write logs to stdout on each iteration.\n
                0 - silent\n
                1 - output general info about iterations\n
                2 - output metric values after each iteration
            seed: random seed.
        """
        if num_attn_passes <= 0:
            raise ValueError("num_attn_passes has to be a positive value.")

        n_w = jnp.zeros(self.vocab_size)
        for batch, _, token_mask in batches:
            n_w += jnp.bincount(
                batch,
                weights=token_mask.astype(jnp.float32),
                length=self.vocab_size,
            )
        self.p_w = n_w / jnp.sum(n_w)  # (W,)

        self._init_state(seed=seed)
        grad_regularization = self._compose_regularizations()

        for it in range(max_iter):
            phi_new, n_t_new = self._batched_step_wrapper(
                batches=batches,
                ctx_weights=self.context_weights,
                grad_reg=grad_regularization,
                num_attn_passes=num_attn_passes,
                lr=lr,
                num_batches_before_update=num_batches_before_update,
            )

            diff_norm = jnp.linalg.norm(phi_new - self.phi)
            if verbose > 0:
                print(
                    f"Iteration [{it + 1}/{max_iter}], phi update diff norm: {diff_norm:.04f}"
                )

            self._flush_metrics(verbose=verbose)

            self.phi = phi_new
            self.n_t = n_t_new
            if diff_norm < tol:
                break
