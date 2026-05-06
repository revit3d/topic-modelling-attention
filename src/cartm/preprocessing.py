from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from typing import TypeAlias

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax import Array
from nltk.corpus import stopwords as default_stopwords


Batch: TypeAlias = tuple[Array, Array, Array]
DocumentSource: TypeAlias = Iterable[str] | Callable[[], Iterable[str]]

_TOKEN_RE = re.compile(r"[a-zA-Z]+")


class CorpusDataLoader:
    """
    Lazy text corpus loader. It handles all preprocessing logic:
    1. preprocess raw documents;
    2. tokenize and normalize tokens;
    3. map tokens to vocabulary ids;
    4. pack token ids into fixed-size batches;
    5. emit JAX arrays suitable for `ModelBase.fit`.

    Note that the loader stores fitted vocabulary.

    Important:
        The data source must be re-iterable because `ModelBase.fit` iterates
        over batches multiple times.

        Good:
            CorpusDataLoader(list_of_docs, ...)
            CorpusDataLoader(lambda: open_texts(path), ...)

        Bad:
            CorpusDataLoader(open_texts(path), ...)
            CorpusDataLoader(iter(list_of_docs), ...)
    """

    def __init__(  # noqa: C901
        self,
        data: DocumentSource,
        *,
        batch_size: int = 10000,
        split_documents: bool = False,
        lower: bool = True,
        vocabulary: dict[str, int] | None = None,
        preprocessor: Callable[[str], str] | None = None,
        tokenizer: Callable[[str], Iterable[str]] | None = None,
        token_normalizer: Callable[[str], str] | None = None,
        stopwords: Iterable[str] | None = None,
        pad_token_id: int = -1,
        min_token_len: int = 2,
        max_token_len: int = 20,
        min_df: int | float = 1,
        max_df: int | float = 1.0,
    ):
        """
        Returned batch tuple:

            data_batch:       (batch_size,)
            doc_bounds_batch: (batch_size,)
            valid_mask:       (batch_size,)

        `valid_mask == False` marks padding tokens.

        Args:
            data: re-iterable source which iterates over documents.
            batch_size: size of a single batch.
            split_documents: if True, each document with
                length > batch_size will be split into two or more batches.
                If false, an error will be raised if such document
                will be encountered in data.
            lower: convert all characters to lowercase per token after tokenization.
            vocabulary: mapping (e.g., a dict) where keys are terms and values
                are unique integers from 0 to len(vocabulary). If not given,
                a vocabulary is inferred from the input documents.
            preprocessor: override the preprocessing stage.
            tokenizer: override the tokenizer stage.
            token_normalizer: normalizer applied to each token
                in splitted text, typically a stemmer or a lemmatizer.
            stopwords: terms to be ignored in tokenized data.
                If None, uses default english stopwords from nltk module.
            pad_token_id: token used for padding.
            min_token_len: if length of a normalized token is less than
                min_token_len, it is ignored.
            max_token_len: if length of a normalized token is more than
                max_token_len, it is ignored.
            min_df: if document frequency of a normalized token is less than
                min_df, it is ignored. If float in range [0.0, 1.0],
                the parameter represents a proportion of documents,
                integer absolute counts.
            max_df: if document frequency of a normalized token is more than
                max_df, it is ignored. If float in range [0.0, 1.0],
                the parameter represents a proportion of documents,
                integer absolute counts.
        """
        self._data = data

        if not callable(data):
            try:
                data_iter = iter(data)
            except TypeError as exc:
                raise TypeError("data must be iterable or a callable returning an iterable") from exc

            if data_iter is data:
                raise TypeError(
                    "CorpusDataLoader must be re-iterable. "
                    "You passed a one-shot iterator/generator. "
                    "Pass a sequence or a callable that returns a fresh iterator instead."
                )

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.batch_size = batch_size
        self.pad_token_id = pad_token_id
        self.split_documents = split_documents

        self._lower = lower
        self._vocab = dict(vocabulary) if vocabulary is not None else None
        self._preprocessor = preprocessor
        self._tokenizer = tokenizer
        self._token_normalizer = token_normalizer
        self._min_token_len = min_token_len
        self._max_token_len = max_token_len
        self._min_df = min_df
        self._max_df = max_df

        self.n_docs_: int | None = None
        self.doc_freq_: Counter[str] | None = None

        if self._vocab is not None:
            self._validate_vocabulary(self._vocab)

        if preprocessor is not None and not callable(preprocessor):
            raise TypeError(f"preprocessor must be callable, got {type(preprocessor)}")

        if tokenizer is not None and not callable(tokenizer):
            raise TypeError(f"tokenizer must be callable, got {type(tokenizer)}")

        if token_normalizer is not None and not callable(token_normalizer):
            raise TypeError(
                f"token_normalizer must be callable, got {type(token_normalizer)}"
            )

        if stopwords is None:
            stopwords = default_stopwords.words("english")

        self._stopwords = set(stopwords)
        if self._lower:
            self._stopwords = {w.lower() for w in self._stopwords}

        if min_token_len < 1:
            raise ValueError("min_token_len must be >= 1")

        if max_token_len < min_token_len:
            raise ValueError("max_token_len must be >= min_token_len")

        self._validate_df_threshold("min_df", min_df)
        self._validate_df_threshold("max_df", max_df)

    @staticmethod
    def _validate_vocabulary(vocabulary: dict[str, int]) -> None:
        if not all(isinstance(k, str) for k in vocabulary):
            raise TypeError("Vocabulary keys must be strings")

        if not all(isinstance(v, int) for v in vocabulary.values()):
            raise TypeError("Vocabulary values must be integer ids")

        ids = list(vocabulary.values())
        if sorted(ids) != list(range(len(ids))):
            raise ValueError(
                "Vocabulary should contain contiguous token ids starting at 0"
            )

    @staticmethod
    def _validate_df_threshold(name: str, value: int | float) -> None:
        if isinstance(value, float):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} as float must be in [0.0, 1.0]")
        elif value < 1:
            raise ValueError(f"{name} as int must be >= 1")

    def _iter_documents(self) -> Iterator[str]:
        docs = self._data() if callable(self._data) else self._data
        return iter(docs)

    def _preprocess_text(self, text: str) -> str:
        if self._preprocessor is not None:
            text = self._preprocessor(text)
        return text

    def _tokenize(self, text: str) -> list[str]:
        if self._tokenizer is not None:
            tokens = self._tokenizer(text)
        else:
            tokens = _TOKEN_RE.findall(text)

        cleaned: list[str] = []
        for token in tokens:
            if self._lower:
                token = token.lower()

            if token in self._stopwords:
                continue

            if self._token_normalizer is not None:
                token = self._token_normalizer(token)
                if token in self._stopwords:
                    continue

            if not self._min_token_len <= len(token) <= self._max_token_len:
                continue

            if token:
                cleaned.append(token)

        return cleaned

    def process_doc(self, doc: str) -> list[str]:
        text = self._preprocess_text(doc)
        return self._tokenize(text)

    def _resolve_df_thresholds(self, n_docs: int) -> tuple[int, int]:
        if isinstance(self._min_df, float):
            min_df = math.ceil(self._min_df * n_docs)
        else:
            min_df = self._min_df

        if isinstance(self._max_df, float):
            max_df = math.floor(self._max_df * n_docs)
        else:
            max_df = self._max_df

        if min_df > max_df:
            raise ValueError("min_df must be <= max_df")

        return min_df, max_df

    def fit(self, *, force: bool = False) -> CorpusDataLoader:
        """
        Learn vocabulary by streaming over the raw corpus once.
        """
        if self._vocab is not None and not force:
            return self

        n_docs = 0
        doc_freq: Counter[str] = Counter()

        for doc in self._iter_documents():
            n_docs += 1
            tokens = self.process_doc(doc)
            doc_freq.update(set(tokens))

        if n_docs == 0:
            raise ValueError("Cannot fit vocabulary on an empty corpus")

        min_df, max_df = self._resolve_df_thresholds(n_docs)

        vocab_words = [
            word
            for word, df in doc_freq.items()
            if min_df <= df <= max_df
        ]
        vocab_words.sort()

        if len(vocab_words) == 0:
            raise ValueError(
                "Vocabulary is empty after applying min_df/max_df/token filters"
            )

        self._vocab = {word: i for i, word in enumerate(vocab_words)}
        self.n_docs_ = n_docs
        self.doc_freq_ = doc_freq

        return self

    def _require_vocabulary(self) -> dict[str, int]:
        if self._vocab is None:
            raise ValueError("Vocabulary is not fitted. Call `fit()` first.")
        return self._vocab

    def iter_encoded_docs(self) -> Iterator[list[int]]:
        """
        Lazily yield one encoded document at a time.

        Empty encoded documents are yielded as empty lists. `__iter__` skips
        them because they do not contribute tokens to model batches.
        """
        vocab = self._require_vocabulary()

        for doc in self._iter_documents():
            tokens = self.process_doc(doc)
            yield [vocab[token] for token in tokens if token in vocab]

    def _make_batch(
        self,
        token_ids: list[int],
        doc_bounds: list[bool],
    ) -> Batch:
        real_len = len(token_ids)

        if real_len == 0:
            raise ValueError("Cannot create an empty batch")

        if real_len > self.batch_size:
            raise ValueError(
                f"Internal error: batch length {real_len} exceeds "
                f"batch_size={self.batch_size}"
            )

        data = np.full(
            self.batch_size,
            fill_value=self.pad_token_id,
            dtype=np.int32,
        )
        bounds = np.zeros(self.batch_size, dtype=bool)
        valid_mask = np.zeros(self.batch_size, dtype=bool)

        data[:real_len] = np.asarray(token_ids, dtype=np.int32)
        bounds[:real_len] = np.asarray(doc_bounds, dtype=bool)
        valid_mask[:real_len] = True

        if real_len < self.batch_size:
            bounds[real_len] = True

        return (
            jnp.asarray(data, dtype=jnp.int32),
            jnp.asarray(bounds, dtype=jnp.bool_),
            jnp.asarray(valid_mask, dtype=jnp.bool_),
        )

    def __iter__(self) -> Iterator[Batch]:
        """
        Lazily preprocess and batch the corpus.

        This method is re-iterable: every call creates a fresh
        pass over the raw data source.
        """
        _ = self._require_vocabulary()

        batch_tokens: list[int] = []
        batch_bounds: list[bool] = []

        def flush() -> Batch:
            batch = self._make_batch(batch_tokens, batch_bounds)
            batch_tokens.clear()
            batch_bounds.clear()
            return batch

        def append_segment(segment: list[int]) -> None:
            if len(segment) == 0:
                return

            starts_new_doc_inside_batch = len(batch_tokens) > 0

            batch_tokens.extend(segment)
            batch_bounds.extend(
                [starts_new_doc_inside_batch] + [False] * (len(segment) - 1)
            )

        for doc_ids in self.iter_encoded_docs():
            if len(doc_ids) == 0:
                continue

            if not self.split_documents:
                if len(doc_ids) > self.batch_size:
                    raise ValueError(
                        f"Found encoded document of length {len(doc_ids)}, "
                        f"but batch_size={self.batch_size}. "
                        "Either increase batch_size or pass split_documents=True."
                    )

                if (
                    len(batch_tokens) > 0
                    and len(batch_tokens) + len(doc_ids) > self.batch_size
                ):
                    yield flush()

                append_segment(doc_ids)
                continue

            offset = 0
            while offset < len(doc_ids):
                if len(batch_tokens) == self.batch_size:
                    yield flush()

                free = self.batch_size - len(batch_tokens)
                take = min(free, len(doc_ids) - offset)

                append_segment(doc_ids[offset:offset + take])
                offset += take

                if len(batch_tokens) == self.batch_size:
                    yield flush()

        if len(batch_tokens) > 0:
            yield flush()

    @property
    def vocabulary(self) -> dict[str, int] | None:
        """
        Token -> id mapping.
        """
        if self._vocab is None:
            return None
        return dict(self._vocab)

    @property
    def vocab_size(self) -> int:
        return len(self._require_vocabulary())


def build_bow_from_loader(loader: CorpusDataLoader) -> sp.csr_matrix:
    """
    Build sparse bow by streaming encoded documents from a fitted loader.
    """
    rows: list[int] = []
    cols: list[int] = []
    values: list[int] = []

    n_docs = 0

    for doc_ids in loader.iter_encoded_docs():
        counts = Counter(doc_ids)

        for token_id, count in counts.items():
            rows.append(n_docs)
            cols.append(token_id)
            values.append(count)

        n_docs += 1

    if len(rows) == 0:
        return sp.csr_matrix((n_docs, loader.vocab_size), dtype=np.uint32)

    return sp.csr_matrix(
        (
            np.asarray(values, dtype=np.uint32),
            (
                np.asarray(rows, dtype=np.int32),
                np.asarray(cols, dtype=np.int32),
            ),
        ),
        shape=(n_docs, loader.vocab_size),
        dtype=np.uint32,
    )
