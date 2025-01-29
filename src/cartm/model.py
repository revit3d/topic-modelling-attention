import numpy as np

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

        self._context_weights_1d = self._get_context_weights_1d(self._gamma)

        self._regularizations = dict()
        if regularizers is not None:
            for reg in regularizers:
                self.add_regularization(reg)
        
        self._metrics = dict()
        if metrics is not None:
            for metric in metrics:
                self.add_metric(metric)

    def add_regularization(self, reg: reg.Regularization):
        """
        Add a regularization to the model.

        Note:
        - `reg` has to be a child of base `Regularization` class
        """
        if not isinstance(reg, reg.Regularization):
            raise TypeError(
                f'Regularization [{reg.__name__}] has to be a subclass of '
                f'the Regularization base class, got type {type(reg)}'
            )

        self._regularizations[reg.tag] = reg

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

    def _norm(self, x: jax.Array) -> jax.Array:
        assert jnp.any(~jnp.isnan(x))
        # take x+ = max(x, 0) element-wise (perform projection on positive simplex)
        x = jnp.maximum(x, jnp.zeros_like(x))
        # normalize values in non-zero rows to 1
        # (mapping from the positive simplex to the unit simplex)
        norm = x.sum(axis=0)
        x = jnp.where(norm > self._eps, x / norm, jnp.zeros_like(x))
        return x

    def _get_context_weights_1d(self, gamma: float) -> jax.Array:
        # w_i = gamma * (1 - gamma)**i
        suffix_context_weights = np.cumprod(np.full(self.ctx_len, (1 - gamma))) * gamma  # (C, )
        prefix_context_weights = suffix_context_weights[::-1]  # (C, )
        context_weights = np.concatenate([
            prefix_context_weights,
            [self._gamma * self._self_aware_context],  # whether to ignore the word itself
            suffix_context_weights,
        ])
        return jnp.array(context_weights)  # (2C + 1, )

    def _get_context_weights_2d(self, batch_size: int, attn_bounds: jax.Array) -> jax.Array:
        # True where to attend
        attn_matrix = np.ones(
            shape=(batch_size, self.ctx_len * 2 + 1),
            dtype=bool,
        )  # (I, 2C + 1)

        # prefix attention mask (ignore words from the previous document in context)
        prefix_bounds = attn_bounds[:-1]  # (B, )

        ignored_mask_prefix = jnp.ones((self.ctx_len, self.ctx_len), dtype=bool)  # (C, C)
        ignored_mask_prefix = jnp.rot90(~np.triu(ignored_mask_prefix))  # (C, C)
        # for broadcasting
        ignored_mask_prefix = jnp.tile(
            ignored_mask_prefix,
            reps=len(prefix_bounds),
        ).T  # (B * C, C)

        # context (row) indices where attention mask is needed (the beginning of a new document)
        shifts = np.ones((len(prefix_bounds), self.ctx_len), dtype=int)  # (B, C)
        shifts[:, 0] = prefix_bounds
        shifts = jnp.cumsum(shifts, axis=1)
        shifts = shifts.reshape(-1, 1)  # (B * C, 1)

        # words (column) indices in prefix context
        prefix_columns = jnp.arange(self.ctx_len)  # (C, )

        attn_matrix[shifts, prefix_columns] = ignored_mask_prefix

        # suffix attention (ignore words from the next document in context)
        suffix_bounds = attn_bounds[1:] - self.ctx_len  # (B, )

        ignored_mask_suffix = jnp.ones((self.ctx_len, self.ctx_len), dtype=bool)  # (C, C)
        ignored_mask_suffix = jnp.rot90(~np.tril(ignored_mask_suffix))  # (C, C)
        # for broadcasting
        ignored_mask_suffix = jnp.tile(
            ignored_mask_suffix,
            reps=len(suffix_bounds),
        ).T  # (B * C, C)

        # context (row) indices where attention mask is needed (the end of a document)
        shifts = np.ones((len(suffix_bounds), self.ctx_len), dtype=int)  # (B, C)
        shifts[:, 0] = suffix_bounds
        shifts = jnp.cumsum(shifts, axis=1)
        shifts = shifts.reshape(-1, 1)  # (B * C, 1)

        # words (column) indices in suffix context
        suffix_columns = jnp.arange(self.ctx_len + 1, self.ctx_len * 2 + 1)  # (C, )

        attn_matrix[shifts, suffix_columns] = ignored_mask_suffix

        # calculate context weights with respect to attention and normalize weights
        context_matrix = self._context_weights_1d * attn_matrix  # (I, 2C + 1)
        context_matrix /= context_matrix.sum(axis=1, keepdims=True)
        return context_matrix  # (I, 2C + 1)

    def _get_context_tensor(self, batch: jax.Array) -> jax.Array:
        """
        Stacks 2d-data into a 3d-tensor along a new (context) axis,
        shifting the data along the new axis. The constructed tensor
        if helpful for fast context convolution with given weights.
        """
        batch_size = batch.shape[0]
        pad_token = 0  # -1

        # shift for each 2d "slice" of data
        shifts = jnp.arange(self.ctx_len * 2, -1, -1)  # (2C + 1, )
        max_shift = self.ctx_len * 2 + batch_size

        stacked_tensor = np.full(
            (max_shift, self.ctx_len * 2 + 1, self.n_topics),
            fill_value=pad_token,
            dtype=jnp.float64,
        )  # (I + 2C, 2C + 1, T)

        row_indices = jnp.arange(batch_size)[:, None] + shifts[None, :]  # (I, 2C + 1)
        col_indices = jnp.arange(self.ctx_len * 2 + 1)  # (2C + 1)

        stacked_tensor[row_indices, col_indices] = batch[:, None]

        # discard contexts for padding
        pad_context_mask = jnp.all(
            jnp.isclose(stacked_tensor[:, self.ctx_len, :], pad_token),
            axis=1,
        )  # (I + 2C, )
        stacked_tensor = stacked_tensor[~pad_context_mask]
        assert stacked_tensor.shape[0] == batch.shape[0]
        return stacked_tensor  # (I, 2C + 1, T)

    def _calc_phi_hatch(self) -> jax.Array:
        return self._norm(self.phi.T * self.n_t[:, None]).T  # (W, T)

    def _calc_theta(
            self,
            phi_hatch: jax.Array,
            batch: jax.Array,
            ctx_bounds: jax.Array,
    ) -> jax.Array:
        batch_size = batch.shape[0]

        phi_it_hatch = jnp.take_along_axis(
            phi_hatch,
            indices=batch[:, None],
            axis=0,
        )  # (I, T)
        phi_it_hatch_with_context = self._get_context_tensor(phi_it_hatch)  # (I, 2C + 1, T)

        context_matrix = self._get_context_weights_2d(
            batch_size=batch_size,
            attn_bounds=ctx_bounds,
        )  # (I, 2C + 1)
        theta_it = context_matrix[..., None] * phi_it_hatch_with_context  # (I, 2C + 1, T)
        theta_it = jnp.sum(theta_it, axis=1)  # (I, T)
        return theta_it

    def _calc_p_ti(self, theta: jax.Array, batch: jax.Array) -> jax.Array:
        phi_it = jnp.take_along_axis(
            self.phi,
            indices=batch[:, None],
            axis=0,
        )  # (I, T)
        p_ti = self._norm((phi_it * theta).T).T  # (I, T)
        return p_ti, phi_it

    def _calc_n_t(self, p_ti: jax.Array) -> jax.Array:
        return jnp.sum(p_ti, axis=0)  # (T, )

    def _calc_phi(
            self,
            p_ti: jax.Array,
            batch: jax.Array,
            grad_regularization: callable,
    ) -> jax.Array:
        phi_new = jnp.add.at(
            jnp.zeros_like(self.phi),
            batch,
            p_ti,
            inplace=False,
        )  # (W, T)
        phi_new += self.phi * grad_regularization(self.phi)  # (W, T)
        phi_new = self._norm(phi_new)  # (W, T)
        return phi_new

    def _compose_regularizations(self):
        regs = self._regularizations.values()
        sum_reg = lambda x: sum([1.0, ] + [reg(x) for reg in regs])
        return jax.jit(jax.grad(sum_reg))

    def _calc_metrics(self, phi_it: jax.Array, phi_wt: jax.Array, theta: jax.Array):
        if len(self._metrics) == 0:
            return

        print('  Metrics:')
        for tag, metric in self._metrics.items():
            value = metric(phi_it=phi_it, phi_wt=phi_wt, theta=theta)
            print(f'    {tag}: {value:.04f}')

    def fit(
        self,
        data: jax.Array,
        ctx_bounds: jax.Array,
        max_iter: int = 1000,
        tol: float = 1e-3,
        verbose: bool = False,
        seed: int = 0,
    ):
        """
        Fit the model with the corpus of documents.

        Args:
            data: array of shape (I, ), containing tokenized words of each document.
            ctx_bounds: array of shape (B, ), containing bounds for context. Words
                beyond the bound are ignored in the context.
            max_iter: max number of iterations.
            tol: early stopping threshold.
            verbose: write logs to stdout on each iteration.
            seed: random seed.
        """
        key = jax.random.key(seed)
        self.phi = jax.random.uniform(
            key=key,
            shape=(self.vocab_size, self.n_topics),
        )  # (W, T)
        self.n_t = jnp.full(
            shape=(self.n_topics, ),
            fill_value=len(data) / self.n_topics,
        )  # (T, )
        grad_regularization = self._compose_regularizations()

        self.phi = self._norm(self.phi)
        for it in range(max_iter):
            # calculate phi' (words -> topics) matrix (phi with old p_{ti})
            phi_hatch = self._calc_phi_hatch()  # (W, T)

            # calculate theta_it = p(t|C_i) matrix
            theta = self._calc_theta(
                phi_hatch=phi_hatch,
                batch=data,
                ctx_bounds=ctx_bounds,
            )  # (I, T)

            # update p_{ti} - topic probability distribution for i-th context
            # phi_it = p(C_i|t)
            p_ti, phi_it = self._calc_p_ti(theta, batch=data)  # (I, T)

            # update n_{t} - topic probability distribution
            self.n_t = self._calc_n_t(p_ti)  # (T, )

            # update phi_wt = p(w|t) matrix
            phi_new = self._calc_phi(
                p_ti=p_ti,
                batch=data,
                grad_regularization=grad_regularization,
            )  # (W, T)

            if verbose:
                diff_norm = jnp.linalg.norm(phi_new - self.phi)
                print(f'Iteration [{it + 1}/{max_iter}], phi update diff norm: {diff_norm:.04f}')
                self._calc_metrics(phi_it, phi_new, theta)

            self.phi = phi_new
            if diff_norm < tol:
                break
