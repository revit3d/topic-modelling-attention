from functools import partial
from typing import Sequence, Callable

import jax
import jax.numpy as jnp

from . import regularization as reg
from . import metrics as mtc


class ContextTopicModel():
    """
    Topic model which uses local context of words
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
            vocab_size: corpus vocabulary size, W
            ctx_len: one-sided context size, C
            n_topics: number of topics, T
            gamma: parameter used for calculating weights of the word embeddings in the context
            self_aware_context: whether to use the word itself in its context
            regularizers: list of regularizations (see `add_regularization` method)
            metrics: list of metrics calculated on each step
            eps: parameter set for balance between numerical stability and precision

        Note:
            - Total context of a word on `i`-th index is ctx_len words to the left,\\
            `ctx_len` words to the right, and the word itself (if `self_aware_context` = True)
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
        - `reg` has to be a child of base `Regularization` class
        """
        if not isinstance(regularization, reg.Regularization):
            raise TypeError(
                f'Regularization [{regularization.__name__}] has to be a subclass of '
                f'the Regularization base class, got type {type(regularization)}'
            )

        self._regularizations[regularization.tag] = regularization

    def add_metric(self, metric: mtc.Metric):
        """
        Add a metric to the model.

        Note:
        - `metric` has to be a child of base `Metric` class
        """
        if not isinstance(metric, mtc.Metric):
            raise TypeError(
                f'Metric [{metric.__name__}] has to be a subclass of '
                f'the Metric base class, got type {type(metric)}'
            )

        self._metrics[metric.tag] = metric

    def remove_regularization(self, tag: str):
        """Remove the regularization with specified tag."""
        try:
            self._regularizations.pop(tag)
        except KeyError:
            print(
                f'Regularization with tag {tag} is not present. '
                f'Did you mean to use remove_metric?'
            )

    def remove_metric(self, tag: str):
        """Remove the metric with the specified tag."""
        try:
            self._metrics.pop(tag)
        except KeyError:
            print(
                f'Metric with tag {tag} is not present. '
                f'Did you mean to use remove_regularization?'
            )

    @partial(jax.jit, static_argnums=0)
    def _norm(self, x: jax.Array) -> jax.Array:
        # take x+ = max(x, 0) element-wise (perform projection on positive simplex)
        x = jnp.maximum(x, jnp.zeros_like(x))
        # normalize values in non-zero rows to 1
        # (mapping from the positive simplex to the unit simplex)
        norm = x.sum(axis=0)
        x = jnp.where(norm > self._eps, x / norm, jnp.zeros_like(x))
        return x

    @partial(jax.jit, static_argnums=(0, 1))
    def _get_context_weights_1d(self, gamma: float) -> jax.Array:
        # w_i = gamma * (1 - gamma)**i
        suffix_context_weights = jnp.cumprod(jnp.full(self.ctx_len, (1 - gamma))) * gamma  # (C, )
        prefix_context_weights = suffix_context_weights[::-1]  # (C, )
        self_context_weight = jnp.array([self._gamma * self._self_aware_context])
        context_weights = jnp.concatenate([
            prefix_context_weights,
            self_context_weight,
            suffix_context_weights,
        ])
        return jnp.array(context_weights)  # (2C + 1, )

    # @partial(jax.jit, static_argnums=0)
    def _get_context_weights_2d(self, batch: jax.Array, attn_bounds: jax.Array) -> jax.Array:
        batch_size = batch.shape[0]

        # True where to attend
        attn_matrix = jnp.ones(
            shape=(batch_size + self.ctx_len * 2, self.ctx_len * 2 + 1),
            dtype=bool,
        )  # (I + 2C, 2C + 1)

        # prefix attention mask (ignore words from the previous document in context)
        prefix_bounds = attn_bounds[:-1] + self.ctx_len  # (B, )

        ignored_mask_prefix = jnp.ones((self.ctx_len, self.ctx_len), dtype=bool)  # (C, C)
        ignored_mask_prefix = jnp.rot90(~jnp.triu(ignored_mask_prefix))  # (C, C)
        # for broadcasting
        ignored_mask_prefix = jnp.tile(
            ignored_mask_prefix,
            reps=len(prefix_bounds),
        ).T  # (B * C, C)

        # context (row) indices where attention mask is needed (the beginning of a new document)
        shifts = jnp.ones((len(prefix_bounds), self.ctx_len), dtype=int)  # (B, C)
        shifts = shifts.at[:, 0].set(prefix_bounds)
        shifts = jnp.cumsum(shifts, axis=1)
        shifts = shifts.reshape(-1, 1)  # (B * C, 1)

        # words (column) indices in prefix context
        prefix_columns = jnp.arange(self.ctx_len)  # (C, )

        attn_matrix = attn_matrix.at[shifts, prefix_columns].set(ignored_mask_prefix)

        # suffix attention (ignore words from the next document in context)
        suffix_bounds = attn_bounds[1:]  # (B, )

        ignored_mask_suffix = jnp.ones((self.ctx_len, self.ctx_len), dtype=bool)  # (C, C)
        ignored_mask_suffix = jnp.rot90(~jnp.tril(ignored_mask_suffix))  # (C, C)
        # for broadcasting
        ignored_mask_suffix = jnp.tile(
            ignored_mask_suffix,
            reps=len(suffix_bounds),
        ).T  # (B * C, C)

        # context (row) indices where attention mask is needed (the end of a document)
        shifts = jnp.ones((len(suffix_bounds), self.ctx_len), dtype=int)  # (B, C)
        shifts = shifts.at[:, 0].set(suffix_bounds)
        shifts = jnp.cumsum(shifts, axis=1)
        shifts = shifts.reshape(-1, 1)  # (B * C, 1)

        # words (column) indices in suffix context
        suffix_columns = jnp.arange(self.ctx_len + 1, self.ctx_len * 2 + 1)  # (C, )

        # apply mask in reverse order
        attn_matrix = attn_matrix.at[shifts[::-1], suffix_columns].set(ignored_mask_suffix[::-1])

        # remove padding
        attn_matrix = attn_matrix[self.ctx_len:-self.ctx_len]  # (I, 2C + 1)

        # calculate context weights with respect to attention and normalize weights
        context_matrix = self._context_weights_1d * attn_matrix  # (I, 2C + 1)
        context_matrix = self._norm(context_matrix.T).T
        return context_matrix  # (I, 2C + 1)

    @partial(jax.jit, static_argnums=0)
    def _get_context_tensor(self, batch: jax.Array) -> jax.Array:
        """
        Stacks 2d-data into a 3d-tensor along a new (context) axis,
        shifting the data along the new axis. The constructed tensor
        if helpful for fast context convolution with given weights.
        """
        batch_size = batch.shape[0]
        pad_token = -1  # assuming we don't have negative tokens in vocabulary

        # shifts for rolling the batch along new dimension
        shifts = jnp.arange(0, -2 * self.ctx_len - 1, -1)  # (2C + 1, )

        # pad batch for shifting
        max_shift = self.ctx_len * 2 + batch_size
        padded_batch = jnp.full(
            (max_shift, self.n_topics),
            fill_value=pad_token,
            dtype=batch.dtype,
        )  # (I + 2C, T)
        padded_batch = padded_batch.at[self.ctx_len:self.ctx_len + batch_size].set(batch)

        # rolling and clipping each "slice" of batch
        def shift_batch(shift):
            return jnp.roll(padded_batch, shift, axis=0)[:batch_size]

        # apply vmap over all shifts
        stacked_tensor = jax.vmap(shift_batch)(shifts).transpose(1, 0, 2)
        return stacked_tensor  # (I, 2C + 1, T)

    @partial(jax.jit, static_argnums=0)
    def _calc_phi_hatch(self, phi: jax.Array, n_t: jax.Array) -> jax.Array:
        return self._norm(phi.T * n_t[:, None]).T  # (W, T)

    # @partial(jax.jit, static_argnums=0)
    def _calc_theta(
            self,
            phi_hatch: jax.Array,
            batch: jax.Array,
            ctx_bounds: jax.Array,
    ) -> jax.Array:
        phi_it_hatch = jnp.take_along_axis(
            phi_hatch,
            indices=batch[:, None],
            axis=0,
        )  # (I, T)
        phi_it_hatch_with_context = self._get_context_tensor(phi_it_hatch)  # (I, 2C + 1, T)
        context_matrix = self._get_context_weights_2d(
            batch=batch,
            attn_bounds=ctx_bounds,
        )  # (I, 2C + 1)
        theta_it = context_matrix[..., None] * phi_it_hatch_with_context  # (I, 2C + 1, T)
        theta_it = jnp.sum(theta_it, axis=1)  # (I, T)
        return theta_it

    @partial(jax.jit, static_argnums=0)
    def _calc_p_ti(
            self,
            phi: jax.Array,
            theta: jax.Array,
            batch: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        phi_it = jnp.take_along_axis(
            phi,
            indices=batch[:, None],
            axis=0,
        )  # (I, T)
        p_ti = self._norm((phi_it * theta).T).T  # (I, T)
        return p_ti, phi_it

    @partial(jax.jit, static_argnums=0)
    def _calc_n_t(self, p_ti: jax.Array) -> jax.Array:
        return jnp.sum(p_ti, axis=0)  # (T, )

    def _calc_phi(
            self,
            batch: jax.Array,
            phi: jax.Array,
            p_ti: jax.Array,
            grad_reg: Callable,
    ) -> jax.Array:
        phi_new = jnp.add.at(
            jnp.zeros_like(phi),
            batch,
            p_ti,
            inplace=False,
        )  # (W, T)
        phi_new += phi * grad_reg(phi)  # (W, T)
        phi_new = self._norm(phi_new)  # (W, T)
        return phi_new

    def _compose_regularizations(self):
        regs = self._regularizations.values()
        reg_grad = jax.grad(lambda x: sum([1.0, ] + [reg(x) for reg in regs]))
        return jax.jit(reg_grad)

    def _calc_metrics(self, phi_it: jax.Array, phi_wt: jax.Array, theta: jax.Array):
        if len(self._metrics) == 0:
            return

        print('  Metrics:')
        for tag, metric in self._metrics.items():
            value = metric(phi_it=phi_it, phi_wt=phi_wt, theta=theta)
            print(f'    {tag}: {value:.04f}')

    def _step(
            self,
            batch: jax.Array,
            ctx_bounds: jax.Array,
            phi: jax.Array,
            n_t: jax.Array,
            grad_reg: Callable,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        # calculate phi' (words -> topics) matrix (phi with old p_{ti})
        phi_hatch = self._calc_phi_hatch(phi=self.phi, n_t=self.n_t)  # (W, T)

        # calculate theta_it = p(t|C_i) matrix
        theta = self._calc_theta(
            phi_hatch=phi_hatch,
            batch=batch,
            ctx_bounds=ctx_bounds,
        )  # (I, T)

        # update p_{ti} - topic probability distribution for i-th context
        # phi_it = p(C_i|t)
        p_ti, phi_it = self._calc_p_ti(
            phi=self.phi,
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
            batches: Sequence[tuple[jax.Array, jax.Array]],
            grad_reg: Callable,
            lr: float,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        phi_new = self.phi.copy()
        phi_it = []
        theta = []

        for batch, ctx_bounds_batch in batches:
            phi_it_step, phi_step, theta_step = self._step(
                batch=batch,
                ctx_bounds=ctx_bounds_batch,
                grad_reg=grad_reg,
            )
            phi_new = phi_new * (1 - lr) + phi_step * lr
            phi_it.append(phi_it_step)
            theta.append(theta_step)

        phi_it = jnp.concatenate(phi_it).reshape(-1, self.n_topics)
        theta = jnp.concatenate(theta).reshape(-1, self.n_topics)
        return phi_it, phi_new, theta

    def fit(
            self,
            data: jax.Array | Sequence[tuple[jax.Array, jax.Array]],
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
            verbose: write logs to stdout on each iteration.
                0 - silent
                1 - output general info about iterations
                2 - output metric values after each iteration
            seed: random seed.
        """
        key = jax.random.key(seed)
        self.phi = jax.random.uniform(
            key=key,
            shape=(self.vocab_size, self.n_topics),
        )  # (W, T)
        self.phi = self._norm(self.phi)
        self.n_t = jnp.full(
            shape=(self.n_topics, ),
            fill_value=len(data) / self.n_topics,
        )  # (T, )
        grad_regularization = self._compose_regularizations()

        for it in range(max_iter):
            if ctx_bounds is None:
                # batched input
                phi_it, phi_new, theta, self.n_t = self._batched_step_wrapper(
                    batches=data,
                    grad_reg=grad_regularization,
                    lr=lr,
                )
            else:
                # non-batched input
                phi_it, phi_new, theta, self.n_t = self._step(
                    batch=data,
                    ctx_bounds=ctx_bounds,
                    grad_reg=grad_regularization,
                )

            if verbose > 0:
                diff_norm = jnp.linalg.norm(phi_new - self.phi)
                print(f'Iteration [{it + 1}/{max_iter}], phi update diff norm: {diff_norm:.04f}')
                if verbose > 1:
                    self._calc_metrics(phi_it, phi_new, theta)

            self.phi = phi_new
            if diff_norm < tol:
                break
