from dataclasses import dataclass


@dataclass(frozen=True)
class TestConfig:
    vocab_size: int = 30
    n_words: int = 150
    n_documents: int = 20
    n_topics: int = 12
    ctx_len: int = 5
    gamma: float = 0.6
    num_attn_passes: int = 1
    seed: int = 42
