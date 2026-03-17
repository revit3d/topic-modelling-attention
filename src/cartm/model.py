from functools import partial
from typing import Iterable, Callable

import jax
import jax.numpy as jnp

from . import regularization as reg
from . import metrics as mtc


class ContextTopicModel:
    """
    Topic model which uses local context of words.
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
        eps: float = 1e-12,
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

        Note:
            - Total context of a word on `i`-th index is ctx_len words to the left,\\
            `ctx_len` words to the right, and the word itself (if `self_aware_context` = True).
        """
        self.ctx_len = ctx_len
        self.vocab_size = vocab_size
        self.n_topics = n_topics
        self._gamma = gamma
        self._self_aware_context = self_aware_context
        self._eps = eps
        self.phi = None
        self.n_t = None

        self._context_weights_1d = self._get_context_weights_1d(self._gamma)

        self._regularizations = {}
        if regularizers is not None:
            for regularization in regularizers:
                self.add_regularization(regularization)

        self._metrics = {}
        if metrics is not None:
            for metric in metrics:
                self.add_metric(metric)

    def add_regularization(self, regularization: reg.Regularization):
        """
        Add a regularization to the model.

        Note:
        - `reg` has to be a child of base `Regularization` class.
        """
        if not isinstance(regularization, reg.Regularization):
            raise TypeError(
                f"Regularization [{regularization.__name__}] has to be a subclass of "
                f"the Regularization base class, got type {type(regularization)}"
            )
        self._regularizations[regularization.tag] = regularization

    def add_metric(self, metric: mtc.Metric):
        """
        Add a metric to the model.

        Note:
        - `metric` has to be a child of base `Metric` class.
        """
        if not isinstance(metric, mtc.Metric):
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

    @partial(jax.jit, static_argnums=(0, 2))
    def _norm(self, x: jax.Array, axis: int = 0) -> jax.Array:
        # take x+ = max(x, 0) element-wise (perform projection on positive simplex)
        x = jnp.maximum(x, 0.0)
        norm = jnp.sum(x, axis=axis, keepdims=True)
        safe_norm = jnp.where(norm > self._eps, norm, 1.0)
        x = jnp.where(norm > self._eps, x / safe_norm, 0.0)
        return x

    @partial(jax.jit, static_argnums=(0, 1))
    def _get_context_weights_1d(self, gamma: float) -> jax.Array:
        # w_i = gamma * (1 - gamma)**i
        suffix_context_weights = (
            jnp.cumprod(jnp.full(self.ctx_len, (1.0 - gamma))) * gamma
        )  # (C, )
        prefix_context_weights = suffix_context_weights[::-1]  # (C, )
        self_context_weight = jnp.array([gamma * self._self_aware_context], dtype=jnp.float32)

        context_weights = jnp.concatenate(
            [
                prefix_context_weights,
                self_context_weight,
                suffix_context_weights,
            ]
        )
        return jnp.array(context_weights)  # (2C + 1, )

    @partial(jax.jit, static_argnums=0)
    def _calc_attn(self, *, matrix: jax.Array, ctx_bounds: jax.Array) -> jax.Array:
        batch_size, embed_size = matrix.shape

        pad_zeros = jnp.zeros((self.ctx_len, embed_size), dtype=matrix.dtype)
        padded_matrix = jnp.concatenate([pad_zeros, matrix, pad_zeros], axis=0)

        doc_starts = jnp.zeros(batch_size, dtype=jnp.int32)
        doc_starts = doc_starts.at[ctx_bounds[:-1]].set(1)
        doc_ids = jnp.cumsum(doc_starts)

        pad_left = jnp.arange(-self.ctx_len, 0, dtype=jnp.int32)
        pad_right = doc_ids[-1] + 1 + jnp.arange(self.ctx_len, dtype=jnp.int32)
        padded_doc_ids = jnp.concatenate([pad_left, doc_ids, pad_right])

        def compute_attn_i(i):
            doc_window = jax.lax.dynamic_slice(padded_doc_ids, (i,), (2 * self.ctx_len + 1,))
            mask = (doc_window == doc_ids[i])

            weights = self._context_weights_1d * mask
            norm = jnp.sum(weights)
            safe_norm = jnp.where(norm > self._eps, norm, 1.0)
            weights = jnp.where(norm > self._eps, weights / safe_norm, 0.0)

            window = jax.lax.dynamic_slice(padded_matrix, (i, 0), (2 * self.ctx_len + 1, embed_size))
            return jnp.dot(weights, window)

        return jax.vmap(compute_attn_i)(jnp.arange(batch_size))

    @partial(jax.jit, static_argnums=0)
    def _calc_phi_hatch(self, *, phi: jax.Array, n_t: jax.Array) -> jax.Array:
        return self._norm(phi * n_t, axis=1)  # (W, T)

    @partial(jax.jit, static_argnums=0)
    def _calc_theta(
        self,
        *,
        phi_hatch: jax.Array,
        batch: jax.Array,
        ctx_bounds: jax.Array,
    ) -> jax.Array:
        return self._calc_attn(matrix=phi_hatch[batch], ctx_bounds=ctx_bounds)  # (I, T)

    @partial(jax.jit, static_argnums=0)
    def _calc_p_ti(
        self, *, phi: jax.Array, theta: jax.Array, batch: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        phi_it = phi[batch]  # (I, T)
        p_ti = self._norm(phi_it * theta, axis=1)  # (I, T)
        return p_ti, phi_it

    @partial(jax.jit, static_argnums=0)
    def _calc_n_t(self, *, p_ti: jax.Array) -> jax.Array:
        return jnp.sum(p_ti, axis=0)  # (T, )

    @partial(jax.jit, static_argnums=0, static_argnames="grad_reg")
    def _calc_phi(
        self,
        *,
        batch: jax.Array,
        phi: jax.Array,
        p_ti: jax.Array,
        grad_reg: Callable,
    ) -> jax.Array:
        phi_new = jnp.zeros_like(phi).at[batch].add(p_ti)
        phi_new -= phi * grad_reg(phi)  # (W, T)
        phi_new = self._norm(phi_new, axis=0)  # (W, T)
        return phi_new

    def _compose_regularizations(self):
        regs = self._regularizations.values()
        reg_grad = jax.grad(
            lambda x: sum([1.0,] + [reg(x) for reg in regs])
        )
        return jax.jit(reg_grad)

    def _calc_metrics(
        self,
        *,
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
            value = metric(phi_it=phi_it, phi_wt=phi_wt, theta=theta)
            if verbose > 1:
                print(f"    {tag}: {value:.04f}")

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
        # calculate phi' (words -> topics) matrix (phi with old p_{ti})
        phi_hatch = self._calc_phi_hatch(phi=phi, n_t=n_t)  # (I, T)

        # calculate theta_it = p(t|C_i) matrix
        theta = self._calc_theta(
            phi_hatch=phi_hatch,
            batch=batch,
            ctx_bounds=ctx_bounds,
        )  # (I, T)

        # update p_{ti} - topic probability distribution for i-th context
        # phi_it = p(C_i|t)
        p_ti, phi_it = self._calc_p_ti(
            phi=phi,
            theta=theta,
            batch=batch,
        )  # (I, T)

        # update n_{t} - topic probability distribution
        n_t_new = self._calc_n_t(p_ti=p_ti)  # (T, )

        # update phi_wt = p(w|t) matrix
        phi_new = self._calc_phi(
            batch=batch,
            phi=phi,
            p_ti=p_ti,
            grad_reg=grad_reg,
        )  # (W, T)

        return phi_it, phi_new, theta, n_t_new

    def _batched_step_wrapper(
        self,
        *,
        batches: Iterable[tuple[jax.Array, jax.Array]],
        phi: jax.Array,
        n_t: jax.Array,
        grad_reg: Callable,
        lr: float,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        phi_new = phi.copy()
        n_t_new = n_t.copy()
        phi_it = []
        theta = []

        for batch, ctx_bounds_batch in batches:
            phi_it_step, phi_step, theta_step, n_t_step = self._step(
                batch=batch,
                ctx_bounds=ctx_bounds_batch,
                phi=phi,
                n_t=n_t,
                grad_reg=grad_reg,
            )
            phi_new = phi_new * (1 - lr) + phi_step * lr
            n_t_new = n_t_new * (1 - lr) + n_t_step * lr
            phi_it.append(phi_it_step)
            theta.append(theta_step)

        phi_it = jnp.concatenate(phi_it).reshape(-1, self.n_topics)
        theta = jnp.concatenate(theta).reshape(-1, self.n_topics)
        return phi_it, phi_new, theta, n_t_new

    def _init_state(self, *, seed: int, data_size: int):
        key = jax.random.key(seed)
        self.phi = jax.random.uniform(
            key=key,
            shape=(self.vocab_size, self.n_topics),
        )  # (W, T)
        self.phi = self._norm(self.phi, axis=0)
        self.n_t = jnp.full(
            shape=(self.n_topics,),
            fill_value=data_size / self.n_topics,
        )  # (T, )

    def fit(
        self,
        data: jax.Array | Iterable[tuple[jax.Array, jax.Array]],
        ctx_bounds: jax.Array = None,
        *,
        lr: float = 0.1,
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
            max_iter: max number of iterations.
            tol: early stopping threshold.
            verbose: write logs to stdout on each iteration.\n
                0 - silent\n
                1 - output general info about iterations\n
                2 - output metric values after each iteration
            seed: random seed.
        """
        self._init_state(seed=seed, data_size=len(data))
        grad_regularization = self._compose_regularizations()

        for it in range(max_iter):
            if ctx_bounds is None:
                # batched input
                phi_it, phi_new, theta, self.n_t = self._batched_step_wrapper(
                    batches=data,
                    phi=self.phi,
                    n_t=self.n_t,
                    grad_reg=grad_regularization,
                    lr=lr,
                )
            else:
                # non-batched input
                phi_it, phi_new, theta, self.n_t = self._step(
                    batch=data,
                    ctx_bounds=ctx_bounds,
                    phi=self.phi,
                    n_t=self.n_t,
                    grad_reg=grad_regularization,
                )

            diff_norm = jnp.linalg.norm(phi_new - self.phi)
            if verbose > 0:
                print(
                    f"Iteration [{it + 1}/{max_iter}], phi update diff norm: {diff_norm:.04f}"
                )

            self._calc_metrics(
                phi_it=phi_it,
                phi_wt=phi_new,
                theta=theta,
                verbose=verbose,
            )

            self.phi = phi_new
            if diff_norm < tol:
                break
