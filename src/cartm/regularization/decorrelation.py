import jax
import jax.numpy as jnp

from .regularization_base import Regularization


class DecorrelationRegularization(Regularization):
    def __init__(self, tau: float, tag: str = None):
        """
        Regularization that decorrelates topics.

        Args:
            tau: value that controls regularization's strength.
            tag: regularizer's name to be displayed in logs.
        """
        if tag is None:
            tag = self.__class__.__name__
        super().__init__(tag=tag, tau=tau)

    def _call_impl(self, phi_wt: jax.Array) -> float:
        corr_matrix = phi_wt.T @ phi_wt  # (T, T)
        # remove duplicates and diagonal terms
        corr_triu = jnp.triu(corr_matrix, k=1)
        return jnp.sum(corr_triu)
