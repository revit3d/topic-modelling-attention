from dataclasses import dataclass


@dataclass
class TestConfig:
    vocab_size = 40
    n_words = 100
    n_documents = 20
    n_topics = 12
    ctx_len = 5
    gamma = 0.6
    seed = 42
