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
    doc_bounds = np.random.choice(
        config.n_words - 1,
        size=config.n_documents - 1,
        replace=False,
    ) + 1
    doc_bounds_ohe = np.zeros(config.n_words, dtype=np.bool)
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
