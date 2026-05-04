import jax
import jax.numpy as jnp

from cartm.core import EPSILON


def calc_norm_vector_primitive(x: jax.Array):
    assert len(x.shape) == 1

    x_plus = []
    for x_i in x:
        x_plus.append(max(x_i, 0))
    x_plus = jnp.array(x_plus)

    x_norm = x_plus / (jnp.sum(x_plus) + EPSILON)
    return x_norm


def calc_norm_matrix_primitive(x: jax.Array):
    assert len(x.shape) == 2

    x_norm = []
    for x_j in x.T:
        x_j_norm = calc_norm_vector_primitive(x=x_j)
        x_norm.append(x_j_norm)
    return jnp.stack(x_norm).T
