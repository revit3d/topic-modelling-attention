import jax
import jax.numpy as jnp

from cartm.core import EPSILON
from cartm.regularization.regularization_base import Regularization


class SparsityRegularization(Regularization):
    def __init__(
        self, alpha: jax.Array, tau: float, *, tag: str | None = None
    ):
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

    def _call_impl(self, phi_wt: jax.Array) -> jax.Array:
        return jnp.sum(self.alpha * jnp.log(phi_wt + EPSILON))
