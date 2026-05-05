import numpy as np
import pytest
from nltk.stem import PorterStemmer

import cartm.preprocessing as pp
from cartm.preprocessing import CorpusDataLoader, build_bow_from_loader


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
def expected_vocabulary_words() -> set[str]:
    return {
        "dare",
        "dark",
        "deep",
        "doubt",
        "dream",
        "ever",
        "fear",
        "long",
        "mortal",
        "peer",
        "stood",
        "wonder",
    }


@pytest.fixture
def custom_stopwords() -> set[str]:
    return {"the", "into", "i", "there", "no", "to", "before"}


@pytest.fixture
def loader(raw_data, custom_stopwords) -> CorpusDataLoader:
    stemmer = PorterStemmer()
    return CorpusDataLoader(
        raw_data,
        batch_size=4,
        token_normalizer=stemmer.stem,
        stopwords=custom_stopwords,
    )


def _ids(vocabulary: dict[str, int], words: list[str]) -> list[int]:
    return [vocabulary[word] for word in words]


def _batch_to_lists(batch):
    data, bounds, mask = batch
    return data.tolist(), bounds.tolist(), mask.tolist()


def test_process_doc(raw_data, expected_words, loader):
    for text, expected in zip(raw_data, expected_words):
        assert loader.process_doc(text) == expected


def test_default_stopwords_are_loaded_and_lowered(monkeypatch):
    monkeypatch.setattr(pp.default_stopwords, "words", lambda language: ["The"])

    loader = pp.CorpusDataLoader(
        ["The Cat"],
        tokenizer=str.split,
    )

    assert loader.process_doc("The Cat") == ["cat"]


def test_fit_infers_sorted_vocabulary_with_pad(
    loader,
    raw_data,
    expected_vocabulary_words,
):
    loader.fit()

    vocabulary = loader.vocabulary

    assert vocabulary is not None
    assert vocabulary["<PAD>"] == -1
    assert set(vocabulary) == expected_vocabulary_words | {"<PAD>"}

    # Inferred non-pad terms are sorted alphabetically and shifted by one
    # because id 0 is reserved for <PAD>.
    expected_sorted_words = sorted(expected_vocabulary_words)
    assert [vocabulary[word] for word in expected_sorted_words] == list(
        range(len(expected_sorted_words))
    )

    assert loader.vocab_size == len(expected_vocabulary_words) + 1
    assert loader.n_docs_ == len(raw_data)
    assert loader.doc_freq_ is not None
    assert loader.doc_freq_["dream"] == 2
    assert loader.doc_freq_["deep"] == 1


def test_vocabulary_property_returns_copy(loader):
    loader.fit()

    vocabulary = loader.vocabulary
    assert vocabulary is not None

    vocabulary["new-token"] = 10_000

    assert "new-token" not in loader.vocabulary


def test_iter_encoded_docs_preserves_document_order(
    loader,
    expected_words,
):
    loader.fit()

    vocabulary = loader.vocabulary
    assert vocabulary is not None
    reverse_vocabulary = {idx: token for token, idx in vocabulary.items()}

    encoded_docs = list(loader.iter_encoded_docs())
    decoded_docs = [
        [reverse_vocabulary[token_id] for token_id in doc_ids]
        for doc_ids in encoded_docs
    ]

    assert decoded_docs == expected_words


def test_iter_requires_fitted_vocabulary(raw_data, custom_stopwords):
    loader = CorpusDataLoader(raw_data, stopwords=custom_stopwords)

    with pytest.raises(ValueError, match="Vocabulary is not fitted"):
        list(loader)

    with pytest.raises(ValueError, match="Vocabulary is not fitted"):
        list(loader.iter_encoded_docs())


def test_iter_batches_documents_without_splitting(loader):
    loader.fit()

    vocabulary = loader.vocabulary
    assert vocabulary is not None

    batches = list(loader)

    assert len(batches) == 4

    batch1_data, batch1_bounds, batch1_mask = _batch_to_lists(batches[0])
    batch2_data, batch2_bounds, batch2_mask = _batch_to_lists(batches[1])
    batch3_data, batch3_bounds, batch3_mask = _batch_to_lists(batches[2])
    batch4_data, batch4_bounds, batch4_mask = _batch_to_lists(batches[3])

    assert batch1_data == _ids(vocabulary, ["deep", "dark", "peer", "<PAD>"])
    assert batch1_bounds == [False, False, False, True]
    assert batch1_mask == [True, True, True, False]

    assert batch2_data == _ids(vocabulary, ["long", "stood", "wonder", "fear"])
    assert batch2_bounds == [False, False, False, False]
    assert batch2_mask == [True, True, True, True]

    assert batch3_data == _ids(vocabulary, ["doubt", "dream", "dream", "mortal"])
    assert batch3_bounds == [False, False, False, False]
    assert batch3_mask == [True, True, True, True]

    assert batch4_data == _ids(vocabulary, ["ever", "dare", "dream", "<PAD>"])
    assert batch4_bounds == [False, False, False, True]
    assert batch4_mask == [True, True, True, False]


def test_iter_batches_multiple_documents_in_one_batch(raw_data, custom_stopwords):
    stemmer = PorterStemmer()
    loader = CorpusDataLoader(
        raw_data,
        batch_size=8,
        token_normalizer=stemmer.stem,
        stopwords=custom_stopwords,
    ).fit()

    vocabulary = loader.vocabulary
    assert vocabulary is not None

    batches = list(loader)

    assert len(batches) == 2

    batch1_data, batch1_bounds, batch1_mask = _batch_to_lists(batches[0])
    batch2_data, batch2_bounds, batch2_mask = _batch_to_lists(batches[1])

    assert batch1_data == (
        _ids(vocabulary, ["deep", "dark", "peer", "long", "stood", "wonder", "fear", "<PAD>"])
    )
    assert batch1_bounds == [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
    ]
    assert batch1_mask == [True, True, True, True, True, True, True, False]

    assert batch2_data == (
        _ids(vocabulary, ["doubt", "dream", "dream", "mortal", "ever", "dare", "dream", "<PAD>"])
    )
    assert batch2_bounds == [
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        True,
    ]
    assert batch2_mask == [True, True, True, True, True, True, True, False]


def test_callable_source_is_reiterated_for_fit_and_each_iteration(
    raw_data,
    custom_stopwords,
):
    calls = 0

    def source():
        nonlocal calls
        calls += 1
        return iter(raw_data)

    stemmer = PorterStemmer()
    loader = CorpusDataLoader(
        source,
        batch_size=4,
        token_normalizer=stemmer.stem,
        stopwords=custom_stopwords,
    )

    loader.fit()
    first_pass = [_batch_to_lists(batch) for batch in loader]
    second_pass = [_batch_to_lists(batch) for batch in loader]

    assert calls == 3
    assert first_pass == second_pass


def test_rejects_one_shot_iterator():
    data = (doc for doc in ["alpha beta", "gamma delta"])

    with pytest.raises(TypeError, match="one-shot|re-iterable"):
        CorpusDataLoader(data, stopwords=())


def test_accepts_callable_returning_fresh_generator():
    def source():
        return (doc for doc in ["alpha beta", "gamma delta"])

    loader = CorpusDataLoader(
        source,
        tokenizer=str.split,
        stopwords=(),
    ).fit()

    assert list(loader.iter_encoded_docs()) == [[0, 1], [3, 2]]


def test_provided_vocabulary_is_used_without_fit():
    loader = CorpusDataLoader(
        ["alpha beta gamma"],
        vocabulary={"alpha": 0, "beta": 1},
        tokenizer=str.split,
        stopwords=(),
    )

    assert loader.vocabulary == {"<PAD>": -1, "alpha": 0, "beta": 1}
    assert list(loader.iter_encoded_docs()) == [[0, 1]]


def test_fit_force_rebuilds_provided_vocabulary():
    loader = CorpusDataLoader(
        ["beta"],
        vocabulary={"alpha": 0},
        tokenizer=str.split,
        stopwords=(),
    )

    loader.fit()
    assert loader.vocabulary == {"<PAD>": -1, "alpha": 0}

    loader.fit(force=True)
    assert loader.vocabulary == {"<PAD>": -1, "beta": 0}
    assert loader.n_docs_ == 1


@pytest.mark.parametrize(
    ("vocabulary", "exc_type", "message"),
    [
        ({1: 0}, TypeError, "keys"),
        ({"a": "0"}, TypeError, "values"),
        ({"a": 1}, ValueError, "contiguous"),
        ({"a": 0, "b": 2}, ValueError, "contiguous"),
    ],
)
def test_rejects_invalid_vocabulary(vocabulary, exc_type, message):
    with pytest.raises(exc_type, match=message):
        CorpusDataLoader(
            ["alpha"],
            vocabulary=vocabulary,
            stopwords=(),
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"batch_size": 0}, "batch_size"),
        ({"min_token_len": 0}, "min_token_len"),
        ({"min_token_len": 3, "max_token_len": 2}, "max_token_len"),
        ({"min_df": 0}, "min_df"),
        ({"max_df": 0}, "max_df"),
        ({"min_df": 1.1}, "min_df"),
        ({"max_df": -0.1}, "max_df"),
    ],
)
def test_rejects_invalid_init_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        CorpusDataLoader(["alpha beta"], stopwords=(), **kwargs)


@pytest.mark.parametrize("hook", ["preprocessor", "tokenizer", "token_normalizer"])
def test_rejects_non_callable_pipeline_hooks(hook):
    with pytest.raises(TypeError, match=hook):
        CorpusDataLoader(
            ["alpha beta"],
            stopwords=(),
            **{hook: object()},
        )


def test_fit_applies_integer_document_frequency_filters():
    docs = [
        "apple banana common",
        "banana common",
        "common rare",
    ]

    loader = CorpusDataLoader(
        docs,
        tokenizer=str.split,
        stopwords=(),
        min_df=2,
        max_df=2,
    ).fit()

    assert loader.vocabulary == {"<PAD>": -1, "banana": 0}


def test_fit_applies_float_document_frequency_filters():
    docs = [
        "apple banana common",
        "banana common",
        "common rare",
    ]

    loader = CorpusDataLoader(
        docs,
        tokenizer=str.split,
        stopwords=(),
        min_df=0.5,  # ceil(0.5 * 3) == 2
        max_df=2 / 3,  # floor(2 / 3 * 3) == 2
    ).fit()

    assert loader.vocabulary == {"<PAD>": -1, "banana": 0}


def test_fit_raises_when_resolved_min_df_exceeds_max_df():
    loader = CorpusDataLoader(
        ["a b", "a", "a"],
        tokenizer=str.split,
        stopwords=(),
        min_token_len=1,
        min_df=1.0,
        max_df=0.5,
    )

    with pytest.raises(ValueError, match="min_df must be <= max_df"):
        loader.fit()


def test_fit_raises_on_empty_corpus():
    loader = CorpusDataLoader([], stopwords=())

    with pytest.raises(ValueError, match="empty corpus"):
        loader.fit()


def test_fit_raises_when_vocabulary_empty_after_filters():
    loader = CorpusDataLoader(
        ["a", "I"],
        tokenizer=str.split,
        stopwords=(),
        min_token_len=2,
    )

    with pytest.raises(ValueError, match="Vocabulary is empty"):
        loader.fit()


def test_long_document_raises_when_split_documents_false():
    loader = CorpusDataLoader(
        ["a b c"],
        batch_size=2,
        tokenizer=str.split,
        stopwords=(),
        min_token_len=1,
        split_documents=False,
    ).fit()

    with pytest.raises(ValueError, match="split_documents=True"):
        list(loader)


def test_long_document_is_split_when_split_documents_true():
    loader = CorpusDataLoader(
        ["a b c", "d"],
        batch_size=2,
        tokenizer=str.split,
        stopwords=(),
        min_token_len=1,
        split_documents=True,
    ).fit()

    vocabulary = loader.vocabulary
    assert vocabulary is not None

    batches = list(loader)

    assert len(batches) == 2

    batch1_data, batch1_bounds, batch1_mask = _batch_to_lists(batches[0])
    batch2_data, batch2_bounds, batch2_mask = _batch_to_lists(batches[1])

    assert batch1_data == _ids(vocabulary, ["a", "b"])
    assert batch1_bounds == [False, False]
    assert batch1_mask == [True, True]

    assert batch2_data == _ids(vocabulary, ["c", "d"])
    assert batch2_bounds == [False, True]
    assert batch2_mask == [True, True]


def test_build_bow_from_loader_streams_encoded_documents(loader):
    loader.fit()

    vocabulary = loader.vocabulary
    assert vocabulary is not None

    bow = build_bow_from_loader(loader)
    dense = bow.toarray()

    assert bow.shape == (4, loader.vocab_size)
    assert bow.dtype == np.uint32

    assert dense[:, vocabulary["<PAD>"]].sum() == 0

    assert dense[0, vocabulary["deep"]] == 1
    assert dense[0, vocabulary["dark"]] == 1
    assert dense[0, vocabulary["peer"]] == 1

    assert dense[2, vocabulary["dream"]] == 2
    assert dense[3, vocabulary["dream"]] == 1
    assert dense[1, vocabulary["dream"]] == 0


def test_build_bow_from_loader_preserves_empty_encoded_documents():
    loader = CorpusDataLoader(
        ["known", "unknown", "known known"],
        vocabulary={"<PAD>": 0, "known": 1},
        tokenizer=str.split,
        stopwords=(),
    )

    bow = build_bow_from_loader(loader)

    assert bow.shape == (3, 2)
    assert bow.toarray().tolist() == [
        [0, 1],
        [0, 0],
        [0, 2],
    ]
