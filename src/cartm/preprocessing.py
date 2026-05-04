import re
import math
from collections import Counter
from typing import Sequence, Callable, Iterable

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax import Array

from nltk.corpus import stopwords as default_stopwords


def build_bow(
    tokenized_data: Array,
    document_bounds: Array,
    vocab_size: int,
) -> sp.csr_matrix:
    tokenized_data = np.asarray(tokenized_data, dtype=np.int32)
    document_bounds = np.asarray(document_bounds, dtype=bool)

    n_docs = np.sum(document_bounds) + 1

    doc_lens = np.bincount(np.cumsum(document_bounds))
    rows = np.repeat(np.arange(n_docs, dtype=np.int32), doc_lens)
    cols = tokenized_data
    data = np.ones_like(cols, dtype=np.uint8)

    bow = sp.csr_matrix((data, (rows, cols)), shape=(n_docs, vocab_size), dtype=np.uint32)
    bow.sum_duplicates()

    return bow


class CorpusLoader:
    def __init__(  # noqa (C901)
        self,
        *,
        lower: bool = True,
        vocabulary: dict | None = None,
        preprocessor: Callable[[str], str] | None = None,
        tokenizer: Callable[[str], list[str]] | None = None,
        token_normalizer: Callable[[str], str] | None = None,
        stopwords: Iterable[str] | None = None,
        min_token_len: int = 2,
        max_token_len: int = 20,
        min_df: int | float = 1,
        max_df: int | float = 1.0,
    ):
        """
        Convert sequence of raw documents into a sequence of tokens
        suitable for fitting the model or batching via BatchedCorpusLoader.

        Args:
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
        self._lower = lower
        self._vocab = vocabulary
        self._preprocessor = preprocessor
        self._tokenizer = tokenizer
        self._token_normalizer = token_normalizer
        self._min_token_len = min_token_len
        self._max_token_len = max_token_len
        self._min_df = min_df
        self._max_df = max_df
        self._cached_tokens = None

        if vocabulary is not None:
            if not all(isinstance(k, str) for k in vocabulary):
                raise TypeError("Vocabulary keys must be strings")
            if not all(isinstance(v, int) for v in vocabulary.values()):
                raise TypeError("Vocabulary values must be integer ids")
            ids = list(vocabulary.values())
            if sorted(ids) != list(range(len(ids))):
                raise ValueError("Vocabulary should contain contiguous token ids starting at 0")

        if preprocessor is not None and not callable(preprocessor):
            raise TypeError(f"Preprocessor must be callable, got {type(preprocessor)}")

        if tokenizer is not None and not callable(tokenizer):
            raise TypeError(f"Tokenizer should be callable, got {type(tokenizer)}")

        if token_normalizer is not None and not callable(token_normalizer):
            raise TypeError(f"Token normalizer should be callable, got {type(token_normalizer)}")

        if stopwords is None:
            stopwords = default_stopwords.words("english")
        self._stopwords = set(stopwords)
        if self._lower:
            self._stopwords = {w.lower() for w in self._stopwords}

        if min_token_len < 1:
            raise ValueError("min_token_len must be >= 1")
        if max_token_len < min_token_len:
            raise ValueError("max_token_len must be >= min_token_len")

        if isinstance(min_df, float):
            if not (0.0 <= min_df <= 1.0):
                raise ValueError("min_df as float must be in [0.0, 1.0]")
        elif min_df < 1:
            raise ValueError("min_df as int must be >= 1")

        if isinstance(max_df, float):
            if not (0.0 <= max_df <= 1.0):
                raise ValueError("max_df as float must be in [0.0, 1.0]")
        elif max_df < 1:
            raise ValueError("max_df as int must be >= 1")

    def _preprocess_text(self, text: str) -> str:
        """Apply preprocessing to a single document."""
        if self._preprocessor is not None:
            text = self._preprocessor(text)
        return text

    def _tokenize(self, text: str) -> list[str]:
        """Apply tokenization to a single document."""
        if self._tokenizer is not None:
            tokens = self._tokenizer(text)
        else:
            tokens = re.findall(r"[a-zA-Z]+", text)

        cleaned = []
        for token in tokens:
            if self._lower:
                token = token.lower()

            if token in self._stopwords:
                continue

            if self._token_normalizer is not None:
                token = self._token_normalizer(token)
                if token in self._stopwords:
                    continue

            if not (self._min_token_len <= len(token) <= self._max_token_len):
                continue

            if token:
                cleaned.append(token)

        return cleaned

    def process_doc(self, doc: str) -> list[str]:
        text = self._preprocess_text(doc)
        return self._tokenize(text)

    def fit(self, data: Sequence[str]) -> "CorpusLoader":
        """Learn a vocabulary dictionary of all tokens in the raw documents."""
        n_docs = len(data)
        doc_freq = Counter()

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

        self._cached_tokens = []
        for doc in data:
            tokens = self.process_doc(doc)
            self._cached_tokens.append(tokens)
            doc_freq.update(set(tokens))

        vocab_words = [
            word
            for word, df in doc_freq.items()
            if df >= min_df and df <= max_df
        ]
        vocab_words.sort()

        self._vocab = {word: i for i, word in enumerate(vocab_words)}
        return self

    def _transform_impl(
        self,
        data: Iterable[str] | Iterable[list[str]],
        return_doc_bounds: bool = True,
        preprocess: bool = True,
    ) -> Array | tuple[Array, Array]:
        if self._vocab is None:
            raise ValueError("Vocabulary is not fitted. Call fit() first.")

        flat_data = []
        doc_bounds = []
        for doc in data:
            tokens = self.process_doc(doc) if preprocess else doc
            encoded = [self._vocab[word] for word in tokens if word in self._vocab]
            flat_data.extend(encoded)
            doc_bounds.append(len(flat_data))
        doc_bounds = doc_bounds[:-1]

        flat_data = np.array(flat_data, dtype=np.int32)
        doc_bounds_ohe = np.zeros_like(flat_data, dtype=bool)
        doc_bounds_ohe[doc_bounds] = True
        flat_data_jnp = jnp.array(flat_data, dtype=jnp.int32)
        doc_bounds_jnp = jnp.array(doc_bounds_ohe, dtype=jnp.bool_)

        if return_doc_bounds:
            return flat_data_jnp, doc_bounds_jnp
        return flat_data_jnp

    def transform(
        self,
        data: Iterable[str],
        *,
        return_doc_bounds: bool = True,
    ) -> Array | tuple[Array, Array]:
        """
        Return a flattened list of all terms from all documents.

        Args:
            data: a sequence of strings.
            return_doc_bounds: if True, also returns boolean array where
                True stands at an index of the first token in each document
                except for the first one.
        """
        return self._transform_impl(
            data=data,
            return_doc_bounds=return_doc_bounds,
            preprocess=True,
        )

    def fit_transform(
        self,
        data: Sequence[str],
        *,
        return_doc_bounds: bool = True,
    ) -> Array | tuple[Array, Array]:
        """
        Learn the vocabulary dictionary and return a flattened list of all
        terms from all documents.

        Args:
            data: a sequence of strings.
            return_doc_bounds: if True, also returns boolean array where
                True stands at an index of the first token in each document
                except for the first one.
        """
        self.fit(data)
        return self._transform_impl(
            data=self._cached_tokens,
            return_doc_bounds=return_doc_bounds,
            preprocess=False,
        )

    @property
    def vocabulary(self) -> dict[str, int] | None:
        """Mapping token -> id."""
        return self._vocab


class BatchedCorpusLoader:
    def __init__(  # noqa (C901)
        self,
        data: Array,
        doc_bounds: Array,
        *,
        batch_size: int = 10000,
        pad_token_id: int = 0,
        split_documents: bool = False,
    ):
        """
        Split tokenized data into fixed-shape, document-aligned batches.

        Returned batch tuple:

            data_batch:       (batch_size,)
            doc_bounds_batch: (batch_size,)
            valid_mask:       (batch_size,)

        `valid_mask == False` marks padding tokens.

        Args:
            data: array of tokens with shape (I, ),
                where I is total number of words in corpus.
            doc_bounds: array of shape (I, ),
                containing ohe of document bounds.
            batch_size: size of a single batch.
            pad_token_id: token used for padding data_batch
                to batch_size.
            split_documents: if True, each document with
                length > batch_size will be split into two or more batches.
                If false, an error will be raised if such document
                will be encountered in data.
        """
        self.batch_size = batch_size
        self.pad_token_id = pad_token_id
        self.split_documents = split_documents
        self._batches = []

        data_np = np.asarray(data, dtype=np.int32)
        bounds_np = np.asarray(doc_bounds, dtype=bool)

        if data_np.ndim != 1:
            raise ValueError("data must be 1-dimensional")
        if bounds_np.shape != data_np.shape:
            raise ValueError("doc_bounds must have the same shape as data")

        data_len = data_np.shape[0]
        if data_len == 0:
            return

        if not split_documents:
            doc_starts = np.concatenate([[0], np.flatnonzero(bounds_np)])
            doc_starts = np.unique(doc_starts)
            doc_starts.sort()

            doc_ends = np.concatenate([doc_starts[1:], [data_len]])

            doc_lens = doc_ends - doc_starts
            max_doc_len = doc_lens.max().item()

            if max_doc_len > self.batch_size:
                raise ValueError(
                    f"Found document of length {max_doc_len}, but batch_size={self.batch_size}. "
                    "To avoid splitting documents, batch_size must be at least the longest "
                    "tokenized document length. To ignore this error, pass split_documents=True."
                )

        def emit_batch(start_idx: int, end_idx: int):
            real_len = end_idx - start_idx
            assert real_len > 0

            data_batch = np.full(
                self.batch_size,
                fill_value=self.pad_token_id,
                dtype=np.int32,
            )
            bounds_batch = np.zeros(self.batch_size, dtype=bool)
            valid_mask = np.zeros(self.batch_size, dtype=bool)

            data_batch[:real_len] = data_np[start_idx:end_idx]
            bounds_batch[:real_len] = bounds_np[start_idx:end_idx]
            valid_mask[:real_len] = True

            bounds_batch[0] = False
            if real_len < self.batch_size:
                bounds_batch[real_len] = True

            self._batches.append(
                (
                    jnp.asarray(data_batch, dtype=jnp.int32),
                    jnp.asarray(bounds_batch, dtype=jnp.bool_),
                    jnp.asarray(valid_mask, dtype=jnp.bool_),
                )
            )

        batch_start = int(doc_starts[0])
        batch_end = batch_start
        used = 0

        for doc_start, doc_end in zip(doc_starts, doc_ends):
            doc_len = doc_end - doc_start

            if doc_len == 0:
                continue

            if used + doc_len > self.batch_size:
                emit_batch(batch_start, batch_end)
                used = 0

            if used == 0:
                batch_start = doc_start

            batch_end = doc_end
            used += doc_len

        if used > 0:
            emit_batch(batch_start, batch_end)

    def __len__(self):
        return len(self._batches)

    def __getitem__(self, idx) -> tuple[Array, Array, Array]:
        return self._batches[idx]

    def __iter__(self):
        return iter(self._batches)
