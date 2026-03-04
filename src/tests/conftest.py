import pytest

import jax
import jax.numpy as jnp

from .config import TestConfig


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
    key = jax.random.key(config.seed)
    return jnp.concatenate(
        [
            jnp.array([0]),
            jax.random.randint(
                key=key,
                shape=(config.n_documents - 1,),
                minval=1,
                maxval=config.n_words - 1,
            ),
            jnp.array([config.n_words]),
        ]
    ).sort()
