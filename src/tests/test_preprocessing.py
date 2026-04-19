import pytest

import jax
import jax.numpy as jnp
from nltk.stem import PorterStemmer

from cartm.preprocessing import CorpusLoader, BatchedCorpusLoader


@pytest.fixture
def raw_data() -> list[str]:
    return [
        "Deep into the darkness peering,",
        "Long I stood there, wondering, fearing,",
        "Doubting, dreaming dreams no mortals",
        "Ever dared to dream before;",
    ]


@pytest.fixture
def expected_words() -> list[list[str]]:
    return [
        ["deep", "dark", "peer"],
        ["long", "stood", "wonder", "fear"],
        ["doubt", "dream", "dream", "mortal"],
        ["ever", "dare", "dream"],
    ]


@pytest.fixture
def expected_vocabulary() -> dict:
    return {
        "deep": 0,
        "dark": 1,
        "peer": 2,
        "long": 3,
        "stood": 4,
        "wonder": 5,
        "fear": 6,
        "doubt": 7,
        "dream": 8,
        "mortal": 9,
        "ever": 10,
        "dare": 11,
    }


@pytest.fixture
def tokenized_data() -> jax.Array:
    return jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11, 8])


@pytest.fixture
def document_bounds() -> jax.Array:
    return jnp.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0])


@pytest.fixture
def preprocessor() -> CorpusLoader:
    stemmer = PorterStemmer()
    return CorpusLoader(token_normalizer=lambda doc: stemmer.stem(doc))


def test_preprocess_text(raw_data, expected_words, preprocessor):
    for text, expected in zip(raw_data, expected_words):
        tokens = preprocessor.process_doc(text)
        assert tokens == expected


def test_fit(raw_data, expected_vocabulary, preprocessor):
    preprocessor.fit(raw_data)
    vocab_out = preprocessor.vocabulary

    assert len(vocab_out) == len(expected_vocabulary)
    assert set(vocab_out.keys()) == set(expected_vocabulary.keys())


def test_fit_transform(raw_data, expected_words, document_bounds, preprocessor):
    tokens_out1 = preprocessor.fit_transform(raw_data, return_doc_bounds=False)
    tokens_out2, doc_bounds_out2 = preprocessor.fit_transform(raw_data)

    vocabulary_out = preprocessor.vocabulary
    reverse_vocabulary = {value: key for key, value in vocabulary_out.items()}
    reconstructed_texts1 = [reverse_vocabulary[token.item()] for token in tokens_out1]
    reconstructed_texts2 = [reverse_vocabulary[token.item()] for token in tokens_out2]
    expected_words_flattened = [word for doc in expected_words for word in doc]

    assert len(reconstructed_texts1) == len(expected_words_flattened)
    assert reconstructed_texts1 == expected_words_flattened

    assert len(reconstructed_texts2) == len(expected_words_flattened)
    assert reconstructed_texts2 == expected_words_flattened

    assert document_bounds.shape[0] == len(tokens_out2)
    assert doc_bounds_out2.shape == document_bounds.shape
    assert doc_bounds_out2.tolist() == document_bounds.tolist()


def test_batch_loader(tokenized_data, document_bounds):
    data, doc_bounds = tokenized_data, document_bounds
    batch_loader = BatchedCorpusLoader(data, doc_bounds, batch_size=4)

    assert len(batch_loader) == 4

    batch1_data, batch1_bounds = batch_loader[0]
    batch2_data, batch2_bounds = batch_loader[1]
    batch3_data, batch3_bounds = batch_loader[2]
    batch4_data, batch4_bounds = batch_loader[3]

    assert batch1_data.tolist() == [0, 1, 2, 3]
    assert batch1_bounds.tolist() == [0, 0, 0, 1]

    assert batch2_data.tolist() == [4, 5, 6, 7]
    assert batch2_bounds.tolist() == [0, 0, 0, 1]

    assert batch3_data.tolist() == [8, 8, 9, 10]
    assert batch3_bounds.tolist() == [0, 0, 0, 1]

    assert batch4_data.tolist() == [11, 8]
    assert batch4_bounds.tolist() == [0, 0]
