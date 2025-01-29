import jax.numpy as jnp

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


class ArticlesDataset():
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
        vocab = dict()
        for text in texts:
            for word in text:
                vocab[word] = vocab.get(word, len(vocab))
        return vocab

    @property
    def data(self):
        return self._data, self._doc_bounds

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

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index) -> jnp.ndarray:
        tokens = self._data[index]
        return tokens
