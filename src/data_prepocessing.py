import numpy as np
import jax.numpy as jnp

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


class ArticlesDataset():
    UNK_TOKEN = 'UNK'
    PAD_TOKEN = 'PAD'

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
        vocab = {
            cls.UNK_TOKEN: 0,
            cls.PAD_TOKEN: 1,
        }
        for text in texts:
            for word in text:
                vocab[word] = vocab.get(word, len(vocab))
        return vocab

    @property
    def unk_token_id(self):
        return self.vocab[self.UNK_TOKEN]

    @property
    def pad_token_id(self):
        return self.vocab[self.PAD_TOKEN]

    @property
    def data(self):
        return self._data

    def __init__(self, data: list[str], maxlen: int):
        texts_tokenized = []
        for doc in data:
            texts_tokenized.append(self.preprocess_text(doc))
        self.vocab = self.create_vocab(texts_tokenized)

        if maxlen is None:
            maxlen = max([len(doc) for doc in texts_tokenized])
        self.maxlen = maxlen

        self._data = np.full((len(texts_tokenized), self.maxlen), fill_value=self.pad_token_id)
        for text_idx, text in enumerate(texts_tokenized):
            for word_idx, word in enumerate(text):
                if word_idx >= self.maxlen:
                    break
                self._data[text_idx][word_idx] = self.vocab[word]
        self._data = jnp.array(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> jnp.ndarray:
        tokens = self.data[index]
        return tokens
