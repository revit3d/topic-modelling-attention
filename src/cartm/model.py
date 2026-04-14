from functools import partial
from typing import Iterable, Callable

import jax
import jax.numpy as jnp
import optax

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
        learnable_context: bool = False,
        word_specific_context: bool = False,
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
        self.ctx_len = ctx_len
        self.vocab_size = vocab_size
        self.n_topics = n_topics
        self._gamma = gamma
        self._self_aware_context = self_aware_context
        self._eps = eps
        self._learnable_context = learnable_context
        self.word_specific_context = word_specific_context
        self.opt_state = None
        self.phi = None
        self.n_t = None
        self.ctx_weights_logits = None

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
    def _calc_attn(
        self,
        *,
        matrix: jax.Array,
        batch: jax.Array,
        ctx_bounds: jax.Array,
        ctx_weights_logits: jax.Array,
    ) -> jax.Array:
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

            if self._learnable_context:
                if self.word_specific_context:
                    logits = ctx_weights_logits[batch[i]]
                else:
                    logits = ctx_weights_logits
                if not self._self_aware_context:
                    logits = logits.at[self.ctx_len].set(-jnp.inf)
                dynamic_weights = jax.nn.softmax(logits, axis=-1)
            else:
                dynamic_weights = self._context_weights_1d

            weights = dynamic_weights * mask
            norm = jnp.sum(weights, axis=-1, keepdims=True)
            safe_norm = jnp.where(norm > self._eps, norm, 1.0)
            weights = jnp.where(norm > self._eps, weights / safe_norm, 0.0)

            window = jax.lax.dynamic_slice(padded_matrix, (i, 0), (2 * self.ctx_len + 1, embed_size))
            return jnp.dot(weights, window)  # (T, )

        return jax.vmap(compute_attn_i)(jnp.arange(batch_size))

    @partial(jax.jit, static_argnums=0)
    def _calc_phi_hatch(self, *, phi_it: jax.Array, n_t: jax.Array) -> jax.Array:
        return self._norm(phi_it * n_t, axis=1)  # (I, T)

    @partial(jax.jit, static_argnums=0)
    def _calc_theta(
        self,
        *,
        p_it: jax.Array,
        batch: jax.Array,
        ctx_bounds: jax.Array,
        ctx_weights_logits: jax.Array,
    ) -> jax.Array:
        return self._calc_attn(
            matrix=p_it,
            batch=batch,
            ctx_bounds=ctx_bounds,
            ctx_weights_logits=ctx_weights_logits,
        )  # (I, T)

    @partial(jax.jit, static_argnums=0)
    def _calc_p_it(
        self, *, p_it: jax.Array, theta: jax.Array, n_t: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        p_it = self._norm(p_it * theta / (n_t + self._eps), axis=1)  # (I, T)
        return p_it

    @partial(jax.jit, static_argnums=0)
    def _calc_n_t(self, *, p_it: jax.Array) -> jax.Array:
        return jnp.sum(p_it, axis=0)  # (T, )

    @partial(jax.jit, static_argnums=0, static_argnames="grad_reg")
    def _calc_phi(
        self,
        *,
        batch: jax.Array,
        phi: jax.Array,
        p_it: jax.Array,
        grad_reg: Callable,
    ) -> jax.Array:
        phi_new = jnp.zeros_like(phi).at[batch].add(p_it)
        phi_new -= phi * grad_reg(phi)  # (W, T)
        phi_new = self._norm(phi_new, axis=0)  # (W, T)
        return phi_new

    @partial(jax.jit, static_argnums=0)
    def _calc_nll(
        self, 
        ctx_weights_logits: jax.Array, 
        phi: jax.Array, 
        n_t: jax.Array, 
        batch: jax.Array, 
        ctx_bounds: jax.Array
    ) -> jax.Array:
        """Computes the Negative Log-Likelihood of the batch to evaluate context weights."""
        p_it = self._calc_phi_hatch(phi_it=phi[batch], n_t=n_t)
        theta = self._calc_theta(
            p_it=p_it,
            batch=batch,
            ctx_bounds=ctx_bounds,
            ctx_weights_logits=ctx_weights_logits,
        )

        # P(w_i | C_i) = Sum over topics [ P(w_i | t) * P(t | C_i) ]
        prob_w = jnp.sum(phi[batch] * theta, axis=1)
        return -jnp.mean(jnp.log(prob_w + self._eps))

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

    @partial(jax.jit, static_argnums=0, static_argnames=("grad_reg", "num_attn_passes"))
    def _step(
        self,
        *,
        batch: jax.Array,
        ctx_bounds: jax.Array,
        phi: jax.Array,
        n_t: jax.Array,
        ctx_weights_logits: jax.Array,
        grad_reg: Callable,
        num_attn_passes: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        # phi_it = p(C_i|t)
        phi_it = phi[batch]

        # calculate phi' (words -> topics) matrix (phi with old p_{ti})
        p_it = self._calc_phi_hatch(phi_it=phi_it, n_t=n_t)  # (I, T)

        for _ in range(num_attn_passes):
            # calculate theta_it = p(t|C_i) matrix
            theta = self._calc_theta(
                p_it=p_it,
                batch=batch,
                ctx_bounds=ctx_bounds,
                ctx_weights_logits=ctx_weights_logits,
            )  # (I, T)

            # update p_{ti} - topic probability distribution for i-th context
            p_it = self._calc_p_it(
                p_it=p_it,
                theta=theta,
                n_t=n_t,
            )  # (I, T)

        # update n_{t} - topic probability distribution
        n_t_new = self._calc_n_t(p_it=p_it)  # (T, )

        # update phi_wt = p(w|t) matrix
        phi_new = self._calc_phi(
            batch=batch,
            phi=phi,
            p_it=p_it,
            grad_reg=grad_reg,
        )  # (W, T)

        if self._learnable_context:
            grad_weights_fn = jax.grad(self._calc_nll, argnums=0)
            logits_grad = grad_weights_fn(ctx_weights_logits, phi, n_t, batch, ctx_bounds)
        else:
            logits_grad = jnp.zeros_like(self._context_weights_1d)

        return phi_it, phi_new, theta, n_t_new, logits_grad

    def _batched_step_wrapper(
        self,
        *,
        batches: Iterable[tuple[jax.Array, jax.Array]],
        phi: jax.Array,
        n_t: jax.Array,
        ctx_weights_logits: jax.Array,
        grad_reg: Callable,
        num_attn_passes: int,
        lr: float,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        phi_new = phi.copy()
        n_t_new = n_t.copy()
        logits_grad = []
        phi_it = []
        theta = []

        for batch, ctx_bounds_batch in batches:
            phi_it_step, phi_step, theta_step, n_t_step, logits_grad_step = self._step(
                batch=batch,
                ctx_bounds=ctx_bounds_batch,
                phi=phi,
                n_t=n_t,
                ctx_weights_logits=ctx_weights_logits,
                grad_reg=grad_reg,
                num_attn_passes=num_attn_passes,
            )
            phi_new = phi_new * (1 - lr) + phi_step * lr
            n_t_new = n_t_new * (1 - lr) + n_t_step * lr
            logits_grad.append(logits_grad_step)
            phi_it.append(phi_it_step)
            theta.append(theta_step)

        phi_it = jnp.concatenate(phi_it).reshape(-1, self.n_topics)
        theta = jnp.concatenate(theta).reshape(-1, self.n_topics)
        logits_grad = jnp.mean(jnp.array(logits_grad), axis=0)
        return phi_it, phi_new, theta, n_t_new, logits_grad

    def _init_state(self, *, seed: int, data_size: int, lr_weights: float):
        key = jax.random.key(seed)
        key, subkey = jax.random.split(key)

        self.phi = jax.random.uniform(
            key=key,
            shape=(self.vocab_size, self.n_topics),
        )  # (W, T)
        self.phi = self._norm(self.phi, axis=0)
        self.n_t = jnp.full(
            shape=(self.n_topics,),
            fill_value=data_size / self.n_topics,
        )  # (T, )

        static_weights = self._get_context_weights_1d(self._gamma)
        if self._learnable_context:
            base_logits = jnp.log(static_weights + self._eps)
            if self.word_specific_context:
                # Матрица (V, 2C+1). Каждое слово в словаре имеет свою "кривую" внимания
                t_weights = jnp.tile(base_logits, (self.vocab_size, 1))
                # ВАЖНО: Добавляем шум, чтобы слова могли учиться индивидуально
                noise = jax.random.normal(subkey, t_weights.shape) * 0.1
                self.ctx_weights_logits = noise
            else:
                self.ctx_weights_logits = base_logits
        else:
            self._context_weights_1d = static_weights

        if self._learnable_context:
            self.optimizer = optax.adam(learning_rate=lr_weights)
            self.opt_state = self.optimizer.init(self.ctx_weights_logits)

    def fit(
        self,
        data: jax.Array | Iterable[tuple[jax.Array, jax.Array]],
        ctx_bounds: jax.Array = None,
        *,
        lr: float = 0.1,
        lr_weights: float = 0.05,
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
            max_iter: max number of iterations.
            tol: early stopping threshold.
            verbose: write logs to stdout on each iteration.\n
                0 - silent\n
                1 - output general info about iterations\n
                2 - output metric values after each iteration
            seed: random seed.
        """
        self._init_state(seed=seed, data_size=len(data), lr_weights=lr_weights)
        grad_regularization = self._compose_regularizations()

        for it in range(max_iter):
            if ctx_bounds is None:
                # batched input
                phi_it, phi_new, theta, self.n_t, logits_grad = self._batched_step_wrapper(
                    batches=data,
                    phi=self.phi,
                    n_t=self.n_t,
                    ctx_weights_logits=self.ctx_weights_logits,
                    grad_reg=grad_regularization,
                    num_attn_passes=num_attn_passes,
                    lr=lr,
                )
            else:
                # non-batched input
                phi_it, phi_new, theta, self.n_t, logits_grad = self._step(
                    batch=data,
                    ctx_bounds=ctx_bounds,
                    phi=self.phi,
                    n_t=self.n_t,
                    ctx_weights_logits=self.ctx_weights_logits,
                    grad_reg=grad_regularization,
                    num_attn_passes=num_attn_passes,
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
            if self._learnable_context:
                updates, self.opt_state = self.optimizer.update(logits_grad, self.opt_state)
                self.ctx_weights_logits = optax.apply_updates(self.ctx_weights_logits, updates)
            if diff_norm < tol:
                break
