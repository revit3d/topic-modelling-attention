import re

import jax.numpy as jnp
from jax import Array

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


class Dataset:
    def __init__(self, data: list[str]):
        texts_tokenized = []
        for doc in data:
            texts_tokenized.append(self.preprocess_text(doc))
        self.vocab = self.create_vocab(texts_tokenized)

        self._data = []
        self._doc_bounds = [0, ]
        for text in texts_tokenized:
            for word in text:
                self._data.append(self.vocab[word])
            self._doc_bounds.append(len(self._data))

        self._data = jnp.array(self._data, dtype=int)
        self._doc_bounds = jnp.array(self._doc_bounds, dtype=int)

    @staticmethod
    def preprocess_text(text: str) -> list[str]:
        text = text.lower()  # lower characters
        text = re.sub(r'[^a-z]', ' ', text)  # remove special characters
        text_tokenized = word_tokenize(text)  # split by words

        lemmatizer = WordNetLemmatizer()  # lemmatize
        text_tokenized = [lemmatizer.lemmatize(word) for word in text_tokenized]

        english_stopwords = set(stopwords.words("english") + ['ha', 'wa'])    # remove stopwords
        text_tokenized = [word for word in text_tokenized if word not in english_stopwords]
        return text_tokenized

    @classmethod
    def create_vocab(cls, texts: list[list[str]]) -> dict:
        vocab = {}
        for text in texts:
            for word in text:
                vocab[word] = vocab.get(word, len(vocab))
        return vocab

    @property
    def data(self):
        return self._data, self._doc_bounds

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx) -> Array:
        tokens = self._data[idx]
        return tokens


class BatchLoader:
    def __init__(self, data: Array, doc_bounds: Array, *, batch_size: int = 10000):
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

    def __getitem__(self, idx):
        return self._batches[idx]
