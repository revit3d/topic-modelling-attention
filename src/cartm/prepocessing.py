import re
from typing import Sequence, Callable

import jax.numpy as jnp
from jax import Array

from nltk.corpus import stopwords as default_stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


class DatasetPreprocessor:
    def __init__(
            self,
            *,
            lower: bool = True,
            vocabulary: dict = None,
            preprocessor: Callable = None,
            tokenizer: Callable = None,
            stopwords: list | set = None,
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
        self._preprocessor = preprocessor
        self._data = None
        self._doc_bounds = None

        if tokenizer is None:
            self._tokenizer = word_tokenize
        else:
            self._tokenizer = tokenizer

        if stopwords is None:
            self._stopwords = set(default_stopwords.words("english"))
        else:
            self._stopwords = set(stopwords)

    def fit(
            self,
            data: Sequence[str],
    ) -> dict:
        """
        Learn a vocabulary dictionary of all tokens in the raw documents.

        Args:
            data: a sequence of strings.
        """
        texts_tokenized = []
        for doc in data:
            texts_tokenized.append(self._preprocess_text(doc))
        self._vocab = self._create_vocabulary(texts_tokenized)
        return self._vocab

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
            return_doc_bounds: if True, returns indices of document bounds
                as the second value (with the first value 0 and the last
                value is len(data)).
        """
        texts_tokenized = []
        for doc in data:
            texts_tokenized.append(self._preprocess_text(doc))

        if self._vocab is None:
            self._vocab = self._create_vocabulary(texts_tokenized)

        self._data = []
        self._doc_bounds = [0, ]
        for text in texts_tokenized:
            for word in text:
                self._data.append(self._vocab[word])
            self._doc_bounds.append(len(self._data))

        self._data = jnp.array(self._data, dtype=int)
        self._doc_bounds = jnp.array(self._doc_bounds, dtype=int)

        if return_doc_bounds:
            return self._data, self._doc_bounds
        return self._data

    def _preprocess_text(self, text: str) -> list[str]:
        """Apply preprocessing and tokenizing to a single document."""
        if self._lower:
            text = text.lower()

        if self._preprocessor is None:
            text = re.sub(r'[^a-z]', ' ', text)
        else:
            text = self._preprocessor(text)

        text_tokenized = self._tokenizer(text)

        lemmatizer = WordNetLemmatizer()
        text_tokenized = [lemmatizer.lemmatize(word) for word in text_tokenized]

        text_tokenized = [word for word in text_tokenized if word not in self._stopwords]

        return text_tokenized

    @classmethod
    def _create_vocabulary(cls, texts: list[list[str]]) -> dict:
        """Create vocabulary from all unique terms in tokenized corpus."""
        _vocab = {}
        for text in texts:
            for word in text:
                _vocab[word] = _vocab.get(word, len(_vocab))
        return _vocab

    @property
    def vocabulary(self):
        """
        Mapping used for tokenizing terms.
        """
        return self._vocab


class BatchLoader:
    def __init__(
            self,
            data: Array,
            doc_bounds: Array,
            *,
            batch_size: int = 10000
    ):
        """
        Split tokenized data into batches. Instance of this class can be passed
        directly to ContextTopicModel for batched fitting.

        Args:
            data: array of tokens with shape (I, ),
                where I is total number of words in corpus.
            doc_bounds: array of shape (B, ),
                containing indices of document bounds.
            batch_size: size of a single batch.
        """
        self.batch_size = batch_size
        self._batches = []

        num_batches = jnp.ceil(len(data) / batch_size).astype(int)
        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = (i + 1) * self.batch_size
            end_idx = min(end_idx, len(data))

            data_batch = data[start_idx:end_idx]
            bounds_batch_mask = (doc_bounds >= start_idx) & (doc_bounds < end_idx)
            doc_bounds_batch = doc_bounds[bounds_batch_mask].copy()
            doc_bounds_batch -= start_idx  # absolute bounds to batch-relative bounds

            # add bounds at the beginning and ending of the batch
            if len(doc_bounds_batch) == 0 or doc_bounds_batch[0] != 0:
                doc_bounds_batch = jnp.concatenate([
                    jnp.array([0]),
                    doc_bounds_batch,
                ], dtype=int)
            if doc_bounds_batch[-1] != self.batch_size:
                doc_bounds_batch = jnp.concatenate([
                    doc_bounds_batch,
                    jnp.array([end_idx - start_idx]),
                ], dtype=int)

            self._batches.append((data_batch, doc_bounds_batch))

    def __len__(self):
        return len(self._batches)

    def __getitem__(self, idx) -> tuple[Array, Array]:
        return self._batches[idx]
