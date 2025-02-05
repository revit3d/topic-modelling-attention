import jax
import jax.numpy as jnp

from .regularization_base import Regularization


class SparsityRegularization(Regularization):
    def __init__(
            self,
            alpha: jax.Array,
            tau: float,
            *,
            tag: str = None,
            eps: float = 1e-12):
        """
        Regularization that approximates the distribution p(w|t) \\
        to a given prior.

        Args:
            alpha: matrix with a prior distribution on p(w|t).
            tau: value that controls regularization's strength.
            tag: regularizer's name to be displayed in logs.
            eps: parameter used for numerical stability.
        """
        if tag is None:
            tag = self.__class__.__name__
        super().__init__(tag=tag, tau=tau)

        self.alpha = alpha
        self._eps = eps

    def _call_impl(self, phi_wt: jax.Array) -> float:
        return jnp.sum(self.alpha * jnp.log(phi_wt + self._eps))
