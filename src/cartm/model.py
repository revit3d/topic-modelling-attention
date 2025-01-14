import numpy as np

import jax
import jax.numpy as jnp

from . import regularization
from . import metrics


class ContextTopicModel():
    """
    Topic model which uses local context of words
    """

    def __init__(
        self,
        ctx_len: int,
        max_len: int,
        vocab_size: int,
        pad_token: int,
        n_topics: int = 10,
        regularizers: list = None,
        metrics: list = None,
        eps: float = 1e-12,
    ):
        """
        Args:
            ctx_len: one-sided context size
            max_len: max length of a document, W_d
            vocab_size: corpus vocabulary size, W
            n_topics: number of topics, T
            regularizers: list of regularizations (see `add_regularization` method)
            metrics: list of metrics calculated on each step
            eps: parameter set for balance between numerical stability and precision

        Note:
            - Total context of a word on `i`-th index is ctx_len words to the left,\\
            `ctx_len` words to the right, and the word itself
            - All documents should be padded to `max_len` length
        """
        self.ctx_len = ctx_len
        self.seq_len = max_len
        self.n_topics = n_topics
        self.vocab_size = vocab_size
        self.pad_token = pad_token
        self._eps = eps

        self._context_coeffs = self._create_context_coeff_matrix()

        self._regularizations = dict()
        if regularizers is not None:
            for reg in regularizers:
                self.add_regularization(reg)
        
        self._metrics = dict()
        if metrics is not None:
            for metric in metrics:
                self.add_metric(metric)

    def add_regularization(self, reg: regularization.Regularization):
        """
        Add a regularization to the model.

        Note:
        - `reg` has to be a child of base `Regularization` class
        """
        if not isinstance(reg, regularization.Regularization):
            raise TypeError(
                f'Regularization [{reg.__name__}] has to be a subclass of '
                f'Regularization base class, got type {type(reg)}'
            )

        try:
            self._regularizations[reg.tag] = reg
        except Exception:
            raise

    def add_metric(self, metric: metrics.Metric):
        """
        Add a metric to the model.

        Note:
        - `metric` has to be a child of base `Metric` class
        """
        if not isinstance(metric, metrics.Metric):
            raise TypeError(
                f'Metric [{metric.__name__}] has to be a subclass of '
                f'Metric base class, got type {type(metric)}'
            )

        try:
            self._metrics[metric.tag] = metric
        except Exception:
            raise

    def _norm(self, x: jax.Array) -> jax.Array:
        assert jnp.any(~jnp.isnan(x)), jnp.sum(x)
        # take x+ = max(x, 0) element-wise (perform projection on positive simplex)
        x = jnp.maximum(x, jnp.zeros_like(x))
        # normalize values in non-zero rows to 1 (mapping from the positive simplex to the unit simplex)
        norm = x.sum(axis=0)
        x = jnp.where(norm > self._eps, x / norm, jnp.zeros_like(x))
        return x

    def _calc_phi_hatch(self):
        return self._norm(self.phi.T * self.n_t[:, None]).T

    def _calc_theta_new(self, phi_hatch, data):
        # we can interpret phi as a bunch of embeddings for words,
        # thus creating tensor of embeddings of words in documents
        # and calculating context (document embedding), based on this
        data_emb = jnp.take_along_axis(phi_hatch[None, ...], indices=data[..., None], axis=1)  # (D, W_d, T)
        theta_new = jnp.sum(
            data_emb[:, None, :, :] * self._context_coeffs[None, :, :, None],
            axis=2,
        )  # (D, W_d, T)
        # now we see each context window as a new document, I - number of context windows
        theta_new = theta_new.reshape(-1, theta_new.shape[-1])  # (I, T)
        return theta_new

    def _calc_p_ti(self, theta_new, data):
        data_emb = jnp.take_along_axis(self.phi[None, ...], indices=data[..., None], axis=1)  # (D, W_d, T)
        p_ti = data_emb.reshape(-1, data_emb.shape[-1])  # (I, T)
        p_ti = self._norm((p_ti * theta_new).T).T  # (I, T)
        return p_ti

    def _calc_n_t(self, p_ti):
        return jnp.sum(p_ti, axis=0)  # (T, )

    def _calc_phi(self, p_ti, data, grad_regularization):
        indices = data.flatten()  # (I, )
        phi_new = jnp.add.at(jnp.zeros_like(self.phi), indices, p_ti, inplace=False)  # (W, T)
        phi_new += self.phi * grad_regularization(self.phi)  # (W, T)
        phi_new = self._norm(phi_new)  # (W, T)
        return phi_new

    def _create_context_coeff_matrix(self) -> jax.Array:
        gamma = 1 / self.ctx_len

        # construct tril matrix (suffix context)
        tril_matrix = np.zeros((self.seq_len, self.seq_len))
        for i in np.arange(1, self.ctx_len + 1):
            tril_matrix[np.arange(i, self.seq_len), np.arange(self.seq_len - i)] = gamma * (1 - gamma) ** i

        # contstruct full matrix (self + prefix + suffix context)
        full_matrix = tril_matrix + tril_matrix.T + np.eye(tril_matrix.shape[0]) * gamma

        # normalize weights and transpose
        full_matrix /= full_matrix.sum(axis=0)
        full_matrix = full_matrix.T
        return jnp.array(full_matrix)

    def _compose_regularizations(self):
        regs = self._regularizations.values()
        sum_reg = lambda x: sum([1.0, ] + [reg(x) for reg in regs])
        return jax.jit(jax.grad(sum_reg))

    def fit(
        self,
        data: jax.Array,
        max_iter: int = 1000,
        tol: float = 1e-3,
        verbose: bool = False,
        seed: int = 0,
    ):
        """
        Fit the model with the corpus of documents.

        The corpus is represented as a matrix of shape (D, W_d), where D\\
        is the number of documents and W_d is the max length of document\\
        in the corpus. Thus, all documents must be aligned to the same\\
        size with pad_token.

        Args:
            data: matrix of shape (D, W_d), containing tokenized words of each document
            max_iter: max number of iterations
            tol: early stopping threshold
            verbose: write logs to stdout on each iteration
            seed: random seed
        """
        key = jax.random.key(seed)
        self.phi = jax.random.uniform(
            key=key,
            shape=(self.vocab_size, self.n_topics),
        )  # (W, T)
        self.n_t = jnp.full(
            shape=(self.n_topics, ),
            fill_value=jnp.sum(data, dtype=jnp.float32) / self.n_topics,
        )  # (T, )
        grad_regularization = self._compose_regularizations()

        self.phi = self._norm(self.phi)
        for it in range(max_iter):
            # calculate phi' (words -> topics) matrix (phi with old p_{ti})
            phi_hatch = self._calc_phi_hatch()  # (W, T)

            # create theta (documents -> topics) matrix
            theta_new = self._calc_theta_new(phi_hatch, data)  # (I, T)

            # update p_{ti} - topic probability distribution for i-th context
            p_ti = self._calc_p_ti(theta_new, data)  # (I, T)

            # update n_{t} - topic probability distribution
            self.n_t = self._calc_n_t(p_ti)  # (T, )

            # update phi (words -> topics) matrix (phi with new p_{ti})
            phi_new = self._calc_phi(p_ti, data, grad_regularization)  # (W, T)

            if verbose:
                diff_norm = jnp.linalg.norm(phi_new - self.phi)
                print(f'Iteration [{it + 1}/{max_iter}], phi update diff norm: {diff_norm:.04f}')
                if len(self._metrics) > 0:
                    print('Metrics:')
                    theta_doc = theta_new.reshape(data.shape + (-1, )).sum(axis=1)  # (D, T)
                    for tag, metric in self._metrics.items():
                        value = metric(phi_new, theta_doc)
                        print(f'\t{tag}: {value:.04f}')

            self.phi = phi_new
            if diff_norm < tol:
                break
