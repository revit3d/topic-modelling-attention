import pytest

import jax
import jax.numpy as jnp
import numpy as np

from tests.config import TestConfig
from tests.math_primitives import calc_norm_matrix_primitive


@pytest.fixture(scope="session")
def config():
    return TestConfig()


@pytest.fixture(scope="session")
def data(config: TestConfig):
    key = jax.random.key(config.seed)
    return jax.random.randint(
        key=key,
        shape=(config.n_words,),
        minval=0,
        maxval=config.vocab_size,
    )


@pytest.fixture(scope="session")
def doc_bounds(config: TestConfig):
    rng = np.random.default_rng(config.seed)
    doc_bounds = rng.choice(
        np.arange(1, config.n_words),
        size=config.n_documents - 1,
        replace=False,
    )
    doc_bounds_ohe = np.zeros(config.n_words, dtype=bool)
    doc_bounds_ohe[doc_bounds] = True
    return jnp.asarray(doc_bounds_ohe)


@pytest.fixture(scope="session")
def phi(config: TestConfig):
    key = jax.random.key(config.seed)
    phi = jax.random.uniform(key=key, shape=(config.vocab_size, config.n_topics))
    phi = calc_norm_matrix_primitive(phi)
    return phi


@pytest.fixture(scope="session")
def n_t(config: TestConfig):
    return jnp.ones(config.n_topics)


@pytest.fixture
def theta(config):
    key = jax.random.key(config.seed)
    theta = jax.random.uniform(key=key, shape=(config.n_words, config.n_topics))
    theta = theta / theta.sum(axis=1, keepdims=True)
    return theta
