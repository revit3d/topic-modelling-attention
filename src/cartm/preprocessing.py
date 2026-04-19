import re
from typing import Sequence, Callable, Iterable

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax import Array

from nltk import word_tokenize
from nltk.corpus import stopwords as default_stopwords
from nltk.stem import PorterStemmer


def build_bow(
    tokenized_data: Array,
    document_bounds: Array,
    vocab_size: int,
) -> sp.csr_matrix:
    tokenized_data = np.asarray(tokenized_data, dtype=np.int32)
    document_bounds = np.asarray(document_bounds, dtype=np.bool)

    n_docs = np.sum(document_bounds) + 1

    doc_lens = np.bincount(np.cumsum(document_bounds))
    rows = np.repeat(np.arange(n_docs, dtype=np.int32), doc_lens)
    cols = tokenized_data
    data = np.ones_like(cols, dtype=np.uint8)

    bow = sp.csr_matrix((data, (rows, cols)), shape=(n_docs, vocab_size), dtype=np.uint32)
    bow.sum_duplicates()

    return bow


class DatasetPreprocessor:
    def __init__(
        self,
        *,
        lower: bool = True,
        vocabulary: dict | None = None,
        preprocessor: Callable[[str], str] | None = None,
        tokenizer: Callable[[str], list[str]] | None = None,
        stopwords: Iterable[str] | None = None,
    ):
        """
        Convert sequence of raw documents into a sequence of tokens
        suitable for fitting the model or batching via BatchLoader.

        Args:
            lower: convert all characters to lowercase before tokenizing.
            vocabulary: mapping (e.g., a dict) where keys are terms and values
                are unique integers from 0 to len(vocabulary). If not given,
                a vocabulary is determined from the input documents.
            preprocessor: override the preprocessing stage.
            tokenizer: override the tokenizer stage.
            stopwords: terms to be ignored in tokenized data.
        """
        self._lower = lower
        self._vocab = vocabulary
        self._stemmer = PorterStemmer()

        if preprocessor is not None and not callable(preprocessor):
            raise TypeError(
                f"Preprocessor should be callable if provided, "
                f"got type {type(preprocessor)}."
            )
        self._preprocessor = preprocessor

        if tokenizer is not None and not callable(tokenizer):
            raise TypeError(
                f"Tokenizer should be callable if provided, "
                f"got type {type(tokenizer)}."
            )
        self._tokenizer = tokenizer

        if stopwords is None:
            self._stopwords = set(default_stopwords.words("english"))
        else:
            self._stopwords = set(stopwords)

    def fit(self, data: Sequence[str]) -> dict:
        """Learn a vocabulary dictionary of all tokens in the raw documents."""
        texts_tokenized = [self._preprocess_text(doc) for doc in data]
        self._vocab = self._create_vocabulary(texts_tokenized)
        return self.vocabulary

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
            return_doc_bounds: if True, returns ohe of document bounds
                as the second value (True at the index of the first token
                in each document).
        """
        texts_tokenized = [self._preprocess_text(doc) for doc in data]

        if self._vocab is None:
            self._vocab = self._create_vocabulary(texts_tokenized)

        flat_data = []
        doc_bounds = []
        for text in texts_tokenized:
            encoded = [self._vocab[word] for word in text if word in self._vocab]
            flat_data.extend(encoded)
            doc_bounds.append(len(flat_data))
        doc_bounds = doc_bounds[:-1]

        flat_data = np.array(flat_data, dtype=np.int32)
        doc_bounds_ohe = np.zeros_like(flat_data, dtype=np.bool)
        doc_bounds_ohe[doc_bounds] = True
        flat_data_jnp = jnp.array(flat_data, dtype=jnp.int32)
        doc_bounds_jnp = jnp.array(doc_bounds_ohe, dtype=jnp.bool)

        if return_doc_bounds:
            return flat_data_jnp, doc_bounds_jnp
        return flat_data_jnp

    def _preprocess_text(self, text: str) -> list[str]:
        """Apply preprocessing and tokenization to a single document."""
        # Preprocessing stage
        if self._preprocessor is None:
            if self._lower:
                text = text.lower()
                text = re.sub(r"[^a-z]", " ", text)
            else:
                text = re.sub(r"[^A-Za-z]", " ", text)
        else:
            text = self._preprocessor(text)

        # tokenization stage
        if self._tokenizer is None:
            text_tokenized = word_tokenize(text)
        else:
            text_tokenized = self._tokenizer(text)

        # removing stopwords
        text_tokenized = [
            word for word in text_tokenized if word not in self._stopwords
        ]

        if self._tokenizer is None:
            text_tokenized = [self._stemmer.stem(token) for token in text_tokenized]

        return text_tokenized

    @staticmethod
    def _create_vocabulary(texts: list[list[str]]) -> dict[str, int]:
        """Create vocabulary from all unique terms in tokenized corpus."""
        unique_words = {word for text in texts for word in text}
        return {word: token for token, word in enumerate(sorted(unique_words))}

    @property
    def vocabulary(self):
        """Mapping used for tokenizing terms."""
        return self._vocab


class BatchLoader:
    def __init__(self, data: Array, doc_bounds: Array, *, batch_size: int = 10000):
        """
        Split tokenized data into batches. Instance of this class can be passed
        directly to ContextTopicModel for batched fitting.

        Args:
            data: array of tokens with shape (I, ),
                where I is total number of words in corpus.
            doc_bounds: array of shape (I, ),
                containing ohe of document bounds.
            batch_size: size of a single batch.
        """
        self.batch_size = batch_size
        self._batches = []

        data_len = data.shape[0]
        num_batches = (data_len + batch_size - 1) // batch_size

        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, data_len)

            data_batch = data[start_idx:end_idx]
            doc_bounds_batch = doc_bounds[start_idx:end_idx]

            self._batches.append((data_batch, doc_bounds_batch))

    def __len__(self):
        return len(self._batches)

    def __getitem__(self, idx) -> tuple[Array, Array]:
        return self._batches[idx]
