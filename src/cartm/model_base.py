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
        ctx_len: int,
        *,
        n_topics: int = 10,
        gamma: float = 0.6,
        self_aware_context: bool = False,
        regularizers: list[Regularization] | None = None,
        metrics: list[Metric] | None = None,
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
    def _init_state(self, *, seed: int, data_size: int):
        pass

    @staticmethod
    @abstractmethod
    def _step(
        batch: jax.Array,
        ctx_bounds: jax.Array,
        phi: jax.Array,
        n_t: jax.Array,
        ctx_weights: jax.Array,
        grad_reg: Callable,
        num_attn_passes: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        pass

    def add_regularization(self, regularization: Regularization):
        """
        Add a regularization to the model.

        Note:
        - `regularization` has to be a child of base `Regularization` class.
        """
        if not isinstance(regularization, Regularization):
            raise TypeError(
                f"Regularization [{regularization.__name__}] has to be a subclass of "
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
                f"Metric [{metric.__name__}] has to be a subclass of "
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
        for tag, metric in self._metrics.items():
            value = metric(
                phi_it=phi_it,
                phi_wt=phi_wt,
                theta=theta,
                batch=batch,
            )
            if verbose > 1:
                print(f"    {tag}: {value:.04f}")

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
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        phi_new = phi.copy()
        n_t_new = n_t.copy()
        phi_it = []
        theta = []
        batch_all = []

        for batch, ctx_bounds_batch in batches:
            phi_it_step, phi_step, theta_step, n_t_step = self._step(
                batch=batch,
                ctx_bounds=ctx_bounds_batch,
                phi=phi,
                n_t=n_t,
                ctx_weights=ctx_weights,
                grad_reg=grad_reg,
                num_attn_passes=num_attn_passes,
            )
            phi_new = phi_new * (1 - lr) + phi_step * lr
            n_t_new = n_t_new * (1 - lr) + n_t_step * lr
            phi_it.append(phi_it_step)
            theta.append(theta_step)
            batch_all.append(batch)

        phi_it = jnp.concatenate(phi_it).reshape(-1, self.n_topics)
        theta = jnp.concatenate(theta).reshape(-1, self.n_topics)
        batch_all = jnp.concatenate(batch_all).reshape(-1)
        return phi_it, phi_new, theta, n_t_new, batch_all

    def fit(
        self,
        data: jax.Array | Iterable[tuple[jax.Array, jax.Array]],
        ctx_bounds: jax.Array = None,
        *,
        lr: float = 0.1,
        num_attn_passes: int = 1,
        max_iter: int = 1000,
        tol: float = 1e-3,
        verbose: int = 0,
        seed: int = 0,
    ):
        """
        Fit the model with the corpus of documents.

        Args:
            data: array of shape (I, ), containing tokenized words of each document
                or iterable returning tuples (data_batch, ctx_bounds_batch).
            ctx_bounds: array of shape (B, ), containing bounds for context. Words
                beyond the bound are ignored in the context.
            lr: coefficient for updating phi in online mode:
                phi = phi_prev * (1 - lr) + phi_new * lr
            num_attn_passes: number of E-steps on each iteration.
            max_iter: max number of iterations.
            tol: early stopping threshold.
            verbose: write logs to stdout on each iteration.\n
                0 - silent\n
                1 - output general info about iterations\n
                2 - output metric values after each iteration
            seed: random seed.
        """
        assert num_attn_passes > 0
        self._init_state(seed=seed, data_size=len(data))
        grad_regularization = self._compose_regularizations()

        for it in range(max_iter):
            if ctx_bounds is None:
                # batched input
                phi_it, phi_new, theta, self.n_t, batch_for_metrics = self._batched_step_wrapper(
                    batches=data,
                    phi=self.phi,
                    n_t=self.n_t,
                    ctx_weights=self.context_weights,
                    grad_reg=grad_regularization,
                    num_attn_passes=num_attn_passes,
                    lr=lr,
                )
            else:
                # non-batched input
                phi_it, phi_new, theta, self.n_t = self._step(
                    batch=data,
                    ctx_bounds=ctx_bounds,
                    phi=self.phi,
                    n_t=self.n_t,
                    ctx_weights=self.context_weights,
                    grad_reg=grad_regularization,
                    num_attn_passes=num_attn_passes,
                )
                batch_for_metrics = data

            diff_norm = jnp.linalg.norm(phi_new - self.phi)
            if verbose > 0:
                print(
                    f"Iteration [{it + 1}/{max_iter}], phi update diff norm: {diff_norm:.04f}"
                )

            self._calc_metrics(
                batch=batch_for_metrics,
                phi_it=phi_it,
                phi_wt=phi_new,
                theta=theta,
                verbose=verbose,
            )

            self.phi = phi_new
            if diff_norm < tol:
                break
