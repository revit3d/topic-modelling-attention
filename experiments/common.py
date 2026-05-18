from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp
import jax
import jax.numpy as jnp

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from datasets import load_dataset as hf_load_dataset

from cartm import AttentiveTopicModel
from cartm.core import EPSILON, norm, calc_attn
from cartm.preprocessing import CorpusDataLoader, build_bow_from_loader
from cartm.regularization import DecorrelationRegularization


Batch = tuple[jax.Array, jax.Array, jax.Array]


@dataclass
class PreparedData:
    dataset_name: str
    train_texts: list[str]
    test_texts: list[str]
    train_texts_filtered: list[str]
    test_texts_filtered: list[str]
    y_train: np.ndarray
    y_test: np.ndarray
    loader: CorpusDataLoader
    train_tokens: jax.Array
    train_bounds: jax.Array
    test_tokens: jax.Array
    test_bounds: jax.Array
    train_bow: sp.csr_matrix
    test_bow: sp.csr_matrix
    train_tfidf: sp.csr_matrix
    test_tfidf: sp.csr_matrix
    vocab: dict[str, int]
    id2word: dict[int, str]
    min_token_len: int = 3
    max_token_len: int = 20

    def make_loader(
        self,
        split: Literal["train", "test"],
        *,
        batch_size: int = 10000,
        split_documents: bool = False,
    ) -> CorpusDataLoader:
        texts = self.train_texts_filtered if split == "train" else self.test_texts_filtered

        return CorpusDataLoader(
            texts,
            batch_size=batch_size,
            split_documents=split_documents,
            lower=True,
            vocabulary=self.vocab,
            min_token_len=self.min_token_len,
            max_token_len=self.max_token_len,
            min_df=1,
            max_df=1.0,
            pad_token_id=0,
        )


def doc_spans(bounds: np.ndarray | jax.Array, n_tokens: int) -> list[tuple[int, int]]:
    bounds = np.asarray(bounds, dtype=bool)[:n_tokens]
    if n_tokens == 0:
        return []

    starts_inside = np.flatnonzero(bounds)
    starts_inside = starts_inside[starts_inside != 0]

    starts = np.r_[0, starts_inside]
    ends = np.r_[starts_inside, n_tokens]
    return list(zip(starts, ends))


def _make_padded_batch(
    token_ids: list[int],
    doc_bounds: list[bool],
    *,
    batch_size: int,
    pad_token_id: int = 0,
) -> Batch:
    real_len = len(token_ids)
    if real_len == 0:
        raise ValueError("Cannot create an empty batch")
    if real_len > batch_size:
        raise ValueError(f"Batch length {real_len} exceeds batch_size={batch_size}")

    data = np.full(batch_size, pad_token_id, dtype=np.int32)
    bounds = np.zeros(batch_size, dtype=bool)
    valid_mask = np.zeros(batch_size, dtype=bool)

    data[:real_len] = np.asarray(token_ids, dtype=np.int32)
    bounds[:real_len] = np.asarray(doc_bounds, dtype=bool)
    valid_mask[:real_len] = True

    # Padding starts a fake new doc. It is masked out, but this prevents
    # padding from being treated as continuation of the last real document.
    if real_len < batch_size:
        bounds[real_len] = True

    return (
        jnp.asarray(data, dtype=jnp.int32),
        jnp.asarray(bounds, dtype=jnp.bool_),
        jnp.asarray(valid_mask, dtype=jnp.bool_),
    )


class TokenBatchLoader:
    """
    Lazy re-iterable batcher for already encoded flat token arrays.

    Emits:
        token_ids:   (batch_size,)
        ctx_bounds:  (batch_size,)
        token_mask:  (batch_size,)

    This is needed by experiments that operate on already-tokenized corpora,
    e.g. truncation/boundary-detection experiments.
    """

    def __init__(
        self,
        data: jax.Array | np.ndarray,
        doc_bounds: jax.Array | np.ndarray,
        *,
        batch_size: int = 10000,
        split_documents: bool = False,
        pad_token_id: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.tokens = np.asarray(data, dtype=np.int32)
        self.bounds = np.asarray(doc_bounds, dtype=bool)
        self.batch_size = batch_size
        self.split_documents = split_documents
        self.pad_token_id = pad_token_id

    def __iter__(self) -> Iterator[Batch]:
        batch_tokens: list[int] = []
        batch_bounds: list[bool] = []

        def flush() -> Batch:
            batch = _make_padded_batch(
                batch_tokens,
                batch_bounds,
                batch_size=self.batch_size,
                pad_token_id=self.pad_token_id,
            )
            batch_tokens.clear()
            batch_bounds.clear()
            return batch

        def append_segment(segment: np.ndarray, *, starts_new_doc: bool) -> None:
            if len(segment) == 0:
                return
            batch_tokens.extend(segment.tolist())
            batch_bounds.extend(
                [starts_new_doc] + [False] * (len(segment) - 1)
            )

        for start, end in doc_spans(self.bounds, len(self.tokens)):
            doc = self.tokens[start:end]
            if len(doc) == 0:
                continue

            if not self.split_documents:
                if len(doc) > self.batch_size:
                    raise ValueError(
                        f"Found document of length {len(doc)}, "
                        f"but batch_size={self.batch_size}. "
                        "Increase batch_size or use split_documents=True."
                    )

                if batch_tokens and len(batch_tokens) + len(doc) > self.batch_size:
                    yield flush()

                append_segment(doc, starts_new_doc=len(batch_tokens) > 0)
                continue

            offset = 0
            is_continuation = False
            while offset < len(doc):
                if len(batch_tokens) == self.batch_size:
                    yield flush()

                free = self.batch_size - len(batch_tokens)
                take = min(free, len(doc) - offset)

                starts_new_doc = len(batch_tokens) > 0 and not is_continuation
                append_segment(doc[offset:offset + take], starts_new_doc=starts_new_doc)

                offset += take
                is_continuation = True

                if len(batch_tokens) == self.batch_size:
                    yield flush()

        if batch_tokens:
            yield flush()

    def __len__(self) -> int:
        return sum(1 for _ in self)


def flatten_loader_to_arrays(loader: CorpusDataLoader) -> tuple[jax.Array, jax.Array]:
    """
    Convert a fitted CorpusDataLoader to old-style flat tokens + global doc bounds.

    Used only by experiments that explicitly need flat token arrays.
    """
    tokens: list[int] = []
    bounds: list[bool] = []

    for doc_ids in loader.iter_encoded_docs():
        if len(doc_ids) == 0:
            continue

        starts_new_doc = len(tokens) > 0
        tokens.extend(doc_ids)
        bounds.extend([starts_new_doc] + [False] * (len(doc_ids) - 1))

    return (
        jnp.asarray(tokens, dtype=jnp.int32),
        jnp.asarray(bounds, dtype=jnp.bool_),
    )


def build_bow_from_tokens(
    tokens: jax.Array | np.ndarray,
    bounds: jax.Array | np.ndarray,
    vocab_size: int,
) -> sp.csr_matrix:
    tokens_np = np.asarray(tokens, dtype=np.int32)
    spans = doc_spans(bounds, len(tokens_np))

    rows: list[int] = []
    cols: list[int] = []
    values: list[int] = []

    for doc_id, (start, end) in enumerate(spans):
        counts = Counter(tokens_np[start:end])
        for token_id, count in counts.items():
            rows.append(doc_id)
            cols.append(int(token_id))
            values.append(int(count))

    if len(rows) == 0:
        return sp.csr_matrix((len(spans), vocab_size), dtype=np.uint32)

    return sp.csr_matrix(
        (
            np.asarray(values, dtype=np.uint32),
            (
                np.asarray(rows, dtype=np.int32),
                np.asarray(cols, dtype=np.int32),
            ),
        ),
        shape=(len(spans), vocab_size),
        dtype=np.uint32,
    )


def load_text_classification_dataset(
    name: Literal["20ng", "ag_news", "dbpedia14"],
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    if name == "20ng":
        train = fetch_20newsgroups(
            subset="train",
            remove=("headers", "footers", "quotes"),
        )
        test = fetch_20newsgroups(
            subset="test",
            remove=("headers", "footers", "quotes"),
        )
        return train.data, test.data, np.asarray(train.target), np.asarray(test.target)

    if name == "ag_news":
        ds = hf_load_dataset("ag_news")
        train_texts = list(ds["train"]["text"])
        test_texts = list(ds["test"]["text"])
        y_train = np.asarray(ds["train"]["label"])
        y_test = np.asarray(ds["test"]["label"])
        return train_texts, test_texts, y_train, y_test

    if name == "dbpedia14":
        ds = hf_load_dataset("dbpedia_14")
        train_texts = [
            f"{title}. {content}"
            for title, content in zip(ds["train"]["title"], ds["train"]["content"])
        ]
        test_texts = [
            f"{title}. {content}"
            for title, content in zip(ds["test"]["title"], ds["test"]["content"])
        ]
        y_train = np.asarray(ds["train"]["label"])
        y_test = np.asarray(ds["test"]["label"])
        return train_texts, test_texts, y_train, y_test

    raise ValueError(f"Unknown dataset: {name}")


def _tokenize_and_filter_empty_docs(
    texts: list[str],
    labels: np.ndarray,
    loader: CorpusDataLoader,
) -> tuple[list[str], list[list[str]], np.ndarray]:
    kept_texts = []
    tokenized_docs = []
    kept_labels = []

    vocab = loader.vocabulary
    assert vocab is not None

    for text, y in zip(texts, labels):
        toks = loader.process_doc(text)
        toks = [tok for tok in toks if tok in vocab]
        if len(toks) == 0:
            continue
        kept_texts.append(text)
        tokenized_docs.append(toks)
        kept_labels.append(y)

    return kept_texts, tokenized_docs, np.asarray(kept_labels)


def prepare_data(
    dataset_name: Literal["20ng", "ag_news", "dbpedia14"],
    *,
    min_df: int | float = 5,
    max_df: int | float = 0.5,
    min_token_len: int = 3,
    max_token_len: int = 20,
) -> PreparedData:
    train_texts, test_texts, y_train, y_test = load_text_classification_dataset(dataset_name)

    loader = CorpusDataLoader(
        train_texts,
        lower=True,
        min_df=min_df,
        max_df=max_df,
        min_token_len=min_token_len,
        max_token_len=max_token_len,
        pad_token_id=0,
    )
    loader.fit()

    train_texts_filtered, _, y_train = _tokenize_and_filter_empty_docs(
        train_texts, y_train, loader
    )
    test_texts_filtered, _, y_test = _tokenize_and_filter_empty_docs(
        test_texts, y_test, loader
    )

    vocab = loader.vocabulary
    assert vocab is not None
    vocab_size = len(vocab)
    id2word = {v: k for k, v in vocab.items()}

    train_loader = CorpusDataLoader(
        train_texts_filtered,
        lower=True,
        vocabulary=vocab,
        min_token_len=min_token_len,
        max_token_len=max_token_len,
        pad_token_id=0,
    )
    test_loader = CorpusDataLoader(
        test_texts_filtered,
        lower=True,
        vocabulary=vocab,
        min_token_len=min_token_len,
        max_token_len=max_token_len,
        pad_token_id=0,
    )

    train_tokens, train_bounds = flatten_loader_to_arrays(train_loader)
    test_tokens, test_bounds = flatten_loader_to_arrays(test_loader)

    train_bow = build_bow_from_loader(train_loader)
    test_bow = build_bow_from_loader(test_loader)

    tfidf = TfidfTransformer(norm="l2")
    train_tfidf = tfidf.fit_transform(train_bow)
    test_tfidf = tfidf.transform(test_bow)

    print("=== Prepared data summary ===")
    print(f"Dataset name: {dataset_name}")
    print(f"Train docs: {len(train_texts)}")
    print(f"Test docs: {len(test_texts)}")
    print(f"Train docs filtered: {len(train_texts_filtered)}")
    print(f"Test docs filtered: {len(test_texts_filtered)}")
    print(f"Classes: {np.unique(np.concatenate([y_train, y_test]))}")
    print(f"Vocabulary: {vocab_size}")
    print(f"Num tokens train: {len(train_tokens)}")
    print(f"Num tokens test: {len(test_tokens)}")

    return PreparedData(
        dataset_name=dataset_name,
        train_texts=train_texts,
        test_texts=test_texts,
        train_texts_filtered=train_texts_filtered,
        test_texts_filtered=test_texts_filtered,
        y_train=y_train,
        y_test=y_test,
        loader=loader,
        train_tokens=train_tokens,
        train_bounds=train_bounds,
        test_tokens=test_tokens,
        test_bounds=test_bounds,
        train_bow=train_bow,
        test_bow=test_bow,
        train_tfidf=train_tfidf,
        test_tfidf=test_tfidf,
        vocab=vocab,
        id2word=id2word,
        min_token_len=min_token_len,
        max_token_len=max_token_len,
    )


def aggregate_doc_topics(
    token_topics: np.ndarray,
    bounds: np.ndarray | jax.Array,
) -> np.ndarray:
    spans = doc_spans(bounds, token_topics.shape[0])
    doc_topics = []
    for start, end in spans:
        doc_topics.append(token_topics[start:end].mean(axis=0))
    doc_topics = np.asarray(doc_topics, dtype=np.float32)
    denom = doc_topics.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return doc_topics / denom


def normalize_cols(x: np.ndarray) -> np.ndarray:
    denom = x.sum(axis=0, keepdims=True)
    denom[denom == 0] = 1.0
    return x / denom


def infer_doc_topics_aartm(
    model: AttentiveTopicModel,
    batches: Iterable[Batch],
    *,
    num_attn_passes: int = 1,
) -> np.ndarray:
    p_it_all = []
    bounds_all = []

    first_batch = True

    for batch_tokens, batch_bounds, token_mask in batches:
        batch_safe = jnp.where(token_mask, batch_tokens, 0)

        p_it_batch = norm(model.phi[batch_safe], axis=1) * token_mask[:, None]

        for _ in range(num_attn_passes):
            theta_batch = calc_attn(
                matrix=p_it_batch,
                ctx_bounds=batch_bounds,
                ctx_weights=model.context_weights,
                token_mask=token_mask,
            )
            p_it_batch = norm(
                p_it_batch * theta_batch / (model.n_t + EPSILON),
                axis=1,
            )
            p_it_batch = p_it_batch * token_mask[:, None]

        valid_np = np.asarray(jax.device_get(token_mask), dtype=bool)

        p_np = np.asarray(jax.device_get(p_it_batch))[valid_np]
        b_np = np.asarray(jax.device_get(batch_bounds), dtype=bool)[valid_np]

        if len(b_np) > 0:
            if not first_batch:
                b_np[0] = True
            first_batch = False

        p_it_all.append(p_np)
        bounds_all.append(b_np)

    if len(p_it_all) == 0:
        return np.empty((0, model.n_topics), dtype=np.float32)

    p_it = np.concatenate(p_it_all, axis=0)
    bounds = np.concatenate(bounds_all, axis=0)

    return aggregate_doc_topics(p_it, bounds)


def aartm_phi_pwt(
    model: AttentiveTopicModel,
    train_tokens: jax.Array | None = None,
) -> np.ndarray:
    if getattr(model, "p_w", None) is None:
        if train_tokens is None:
            raise ValueError("model.p_w is missing; pass train_tokens to estimate it.")
        n_w = jnp.bincount(train_tokens, length=model.vocab_size)
        p_w = n_w / jnp.sum(n_w)
    else:
        p_w = model.p_w

    phi_wt = model.renormalize_phi(p_w=p_w, phi=model.phi)
    return np.asarray(jax.device_get(phi_wt))


def top_words(phi_wt: np.ndarray, id2word: dict[int, str], top_k: int = 10) -> list[list[str]]:
    idx = np.argpartition(phi_wt, -top_k, axis=0)[-top_k:]  # (k, T)
    idx = idx[::-1].T  # (T, k)
    result = []
    for topic_words in idx:
        result.append([id2word[int(i)] for i in topic_words])
    return result


def save_top_words(
    phi_wt: np.ndarray,
    id2word: dict[int, str],
    out_path: str | Path,
    top_k: int = 10,
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    topics = top_words(phi_wt, id2word, top_k=top_k)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, words in enumerate(topics):
            f.write(f"Topic {i}\t" + ", ".join(words) + "\n")


def topic_diversity(phi_wt: np.ndarray, top_k: int = 25) -> float:
    top_idx = np.argpartition(phi_wt, -top_k, axis=0)[-top_k:]
    unique_words = np.unique(top_idx)
    return float(len(unique_words) / (phi_wt.shape[1] * top_k))


def topic_sparsity(phi_wt: np.ndarray, eps: float = 1e-12) -> float:
    return float((np.abs(phi_wt) < eps).mean())


def mean_nearest_hellinger(phi_wt: np.ndarray) -> float:
    topics = phi_wt.T  # (T, W)
    T = topics.shape[0]
    dist = np.full((T, T), np.inf, dtype=np.float32)
    sqrt_topics = np.sqrt(np.maximum(topics, 0.0))
    for i in range(T):
        for j in range(T):
            if i == j:
                continue
            d2 = ((sqrt_topics[i] - sqrt_topics[j]) ** 2).sum() / 2.0
            dist[i, j] = np.sqrt(d2)
    return float(dist.min(axis=1).mean())


@jax.jit(static_argnames=("num_attn_passes",))
def _aartm_log_likelihood_batch(
    batch: jax.Array,
    ctx_bounds: jax.Array,
    token_mask: jax.Array,
    phi_tw: jax.Array,
    phi_wt: jax.Array,
    n_t: jax.Array,
    ctx_weights: jax.Array,
    num_attn_passes: int,
) -> tuple[jax.Array, jax.Array]:
    """
    Batch log-likelihood for AARTM-style models.

    phi_tw is model.phi = p(t|w)
    phi_wt is renormalized phi = p(w|t)

    Uses the same theta inference logic as infer_doc_topics_aartm.
    """
    batch_safe = jnp.where(token_mask, batch, 0)

    p_it = norm(phi_tw[batch_safe], axis=1) * token_mask[:, None]
    theta = jnp.zeros_like(p_it)

    for _ in range(num_attn_passes):
        theta = calc_attn(
            matrix=p_it,
            ctx_bounds=ctx_bounds,
            ctx_weights=ctx_weights,
            token_mask=token_mask,
        )
        p_it = norm(
            p_it * theta / (n_t + EPSILON),
            axis=1,
        )
        p_it = p_it * token_mask[:, None]

    # p(w_i | C_i) = sum_t p(w_i | t) p(t | C_i)
    p_wi = jnp.sum(theta * phi_wt[batch_safe], axis=1)

    log_likelihood = jnp.sum(jnp.log(p_wi + EPSILON) * token_mask)
    num_words = jnp.sum(token_mask)

    return log_likelihood, num_words


def aartm_perplexity(
    model: AttentiveTopicModel,
    batches: Iterable[Batch],
    *,
    num_attn_passes: int = 1,
    phi_wt: np.ndarray | jax.Array | None = None,
    train_tokens: jax.Array | None = None,
) -> float:
    """
    Offline perplexity for AttentiveTopicModel / AttentiveTopicModelNoNWT.

    Returns:
        exp(- log_likelihood / num_words)
    """
    if num_attn_passes <= 0:
        raise ValueError("num_attn_passes must be positive.")

    if phi_wt is None:
        phi_wt = aartm_phi_pwt(model, train_tokens)

    phi_wt = jnp.asarray(phi_wt)

    total_log_likelihood = 0.0
    total_words = 0

    for batch, ctx_bounds, token_mask in batches:
        ll_batch, n_batch = _aartm_log_likelihood_batch(
            batch=batch,
            ctx_bounds=ctx_bounds,
            token_mask=token_mask,
            phi_tw=model.phi,
            phi_wt=phi_wt,
            n_t=model.n_t,
            ctx_weights=model.context_weights,
            num_attn_passes=num_attn_passes,
        )

        total_log_likelihood += float(jax.device_get(ll_batch))
        total_words += int(jax.device_get(n_batch))

    if total_words == 0:
        return float("nan")

    return float(np.exp(-total_log_likelihood / total_words))


def npmi_score(phi_wt: np.ndarray, bow: sp.csr_matrix, top_k: int = 10) -> float:
    bow = bow.sign().astype(np.uint8).tocsc(copy=False)
    df = np.asarray(bow.getnnz(axis=0)).ravel().astype(np.float32)
    n_docs = bow.shape[0]

    top_words = np.argpartition(phi_wt, -top_k, axis=0)[-top_k:].T  # (T, k)
    selected, inv = np.unique(top_words, return_inverse=True)
    inv = inv.reshape(top_words.shape)

    X_sub = bow[:, selected]
    cooc = (X_sub.T @ X_sub).toarray().astype(np.float32)
    df_sub = df[selected]

    p_i = df_sub / n_docs
    p_ij = cooc / n_docs
    denom = p_i[:, None] * p_i[None, :]

    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(p_ij / denom)
        npmi = pmi / (-np.log(p_ij))

    npmi = np.where(cooc > 0, npmi, -1.0)

    triu = np.triu_indices(top_k, k=1)
    topic_scores = []
    for topic_idx in inv:
        local = npmi[np.ix_(topic_idx, topic_idx)]
        topic_scores.append(local[triu].mean())

    return float(np.mean(topic_scores))


def classification_scores(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int = 0,
) -> dict[str, float]:
    clf = LogisticRegression(
        max_iter=2000,
        random_state=seed,
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
    }


def fit_aartm(
    data: PreparedData,
    *,
    n_topics: int,
    ctx_len: int,
    gamma: float,
    self_aware_context: bool,
    num_attn_passes: int,
    max_iter: int,
    tol: float,
    seed: int,
    batch_size: int = -1,
    decorrelation_tau: float = 0.0,
) -> tuple[AttentiveTopicModel, float]:
    regs = build_regularizers(decorrelation_tau, "tw")

    model = AttentiveTopicModel(
        vocab_size=len(data.vocab),
        ctx_len=ctx_len,
        n_topics=n_topics,
        gamma=gamma,
        self_aware_context=self_aware_context,
        regularizers=regs,
    )

    if batch_size is None or batch_size <= 0:
        batch_size = int(len(data.train_tokens))

    batches = data.make_loader(
        "train",
        batch_size=batch_size,
    )

    elapsed = fit_topic_model(
        model,
        batches=batches,
        num_attn_passes=num_attn_passes,
        max_iter=max_iter,
        tol=tol,
        seed=seed,
        batch_size=batch_size,
    )

    return model, elapsed


def fit_lda(
    data: PreparedData,
    *,
    n_topics: int,
    max_iter: int,
    seed: int,
) -> tuple[LatentDirichletAllocation, float]:
    model = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=max_iter,
        learning_method="batch",
        random_state=seed,
        n_jobs=8,
        evaluate_every=-1,
    )
    t0 = perf_counter()
    model.fit(data.train_bow)
    elapsed = perf_counter() - t0
    return model, elapsed


def fit_nmf(
    data: PreparedData,
    *,
    n_topics: int,
    max_iter: int,
    seed: int,
) -> tuple[NMF, float]:
    model = NMF(
        n_components=n_topics,
        init="nndsvda",
        max_iter=max_iter,
        random_state=seed,
    )
    t0 = perf_counter()
    model.fit(data.train_tfidf)
    elapsed = perf_counter() - t0
    return model, elapsed


def evaluate_aartm(
    model: AttentiveTopicModel,
    data: PreparedData,
    *,
    batch_size: int,
    num_attn_passes: int,
    seed: int,
    **kwargs,
) -> dict[str, float]:
    phi_wt = aartm_phi_pwt(model, data.train_tokens)

    batches_train = data.make_loader("train", batch_size=batch_size)
    X_train = infer_doc_topics_aartm(
        model,
        batches_train,
        num_attn_passes=num_attn_passes,
    )

    batches_test = data.make_loader("test", batch_size=batch_size)
    X_test = infer_doc_topics_aartm(
        model,
        batches_test,
        num_attn_passes=num_attn_passes,
    )

    metrics = {
        "npmi_10": npmi_score(phi_wt, data.train_bow, top_k=10),
        "topic_diversity_25": topic_diversity(phi_wt, top_k=25),
        "topic_sparsity": topic_sparsity(phi_wt),
        "topic_hellinger": mean_nearest_hellinger(phi_wt),
    }
    metrics.update(classification_scores(X_train, data.y_train, X_test, data.y_test, seed=seed))
    return metrics


def evaluate_lda(
    model: LatentDirichletAllocation,
    data: PreparedData,
    *,
    seed: int,
) -> dict[str, float]:
    phi_tw = model.components_
    phi_wt = normalize_cols(phi_tw.T)

    X_train = model.transform(data.train_bow)
    X_test = model.transform(data.test_bow)
    X_train = X_train / np.maximum(X_train.sum(axis=1, keepdims=True), 1e-12)
    X_test = X_test / np.maximum(X_test.sum(axis=1, keepdims=True), 1e-12)

    metrics = {
        "npmi_10": npmi_score(phi_wt, data.train_bow, top_k=10),
        "topic_diversity_25": topic_diversity(phi_wt, top_k=25),
        "topic_sparsity": topic_sparsity(phi_wt),
        "topic_hellinger": mean_nearest_hellinger(phi_wt),
    }
    metrics.update(classification_scores(X_train, data.y_train, X_test, data.y_test, seed=seed))
    return metrics


def evaluate_nmf(
    model: NMF,
    data: PreparedData,
    *,
    seed: int,
) -> dict[str, float]:
    phi_tw = model.components_
    phi_wt = normalize_cols(phi_tw.T)

    X_train = model.transform(data.train_tfidf)
    X_test = model.transform(data.test_tfidf)
    X_train = X_train / np.maximum(X_train.sum(axis=1, keepdims=True), 1e-12)
    X_test = X_test / np.maximum(X_test.sum(axis=1, keepdims=True), 1e-12)

    metrics = {
        "npmi_10": npmi_score(phi_wt, data.train_bow, top_k=10),
        "topic_diversity_25": topic_diversity(phi_wt, top_k=25),
        "topic_sparsity": topic_sparsity(phi_wt),
        "topic_hellinger": mean_nearest_hellinger(phi_wt),
    }
    metrics.update(classification_scores(X_train, data.y_train, X_test, data.y_test, seed=seed))
    return metrics


def aggregate_results(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metric_cols = [c for c in df.columns if c not in group_cols]
    agg = df.groupby(group_cols)[metric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in agg.columns
    ]
    return agg


def parse_df_arg(x: str):
    return float(x) if "." in x else int(x)


class DocumentBatchedCorpusLoader:

    def __init__(self, data: jax.Array, doc_bounds: jax.Array, *, batch_size: int = 10000):
        self.batch_size = batch_size
        self._batches = []

        tokens = np.asarray(data, dtype=np.int32)
        bounds = np.asarray(doc_bounds, dtype=bool)
        spans = doc_spans(bounds, len(tokens))

        current_docs = []
        current_tokens = 0

        for start, end in spans:
            doc = tokens[start:end]
            doc_len = len(doc)

            if len(current_docs) > 0 and current_tokens + doc_len > batch_size:
                self._flush(current_docs)
                current_docs = []
                current_tokens = 0

            current_docs.append(doc)
            current_tokens += doc_len

            # if a single document is already very large, keep it in its own batch
            if doc_len >= batch_size:
                self._flush(current_docs)
                current_docs = []
                current_tokens = 0

        if len(current_docs) > 0:
            self._flush(current_docs)

    def _flush(self, docs: list[np.ndarray]):
        flat = np.concatenate(docs)
        bounds = np.zeros(len(flat), dtype=bool)

        offset = 0
        for i, doc in enumerate(docs):
            if i > 0:
                bounds[offset] = True
            offset += len(doc)

        self._batches.append(
            (
                jnp.asarray(flat, dtype=jnp.int32),
                jnp.asarray(bounds, dtype=jnp.bool_),
            )
        )

    def __len__(self):
        return len(self._batches)

    def __getitem__(self, idx):
        return self._batches[idx]

    def __iter__(self):
        return iter(self._batches)


def build_regularizers(decorrelation_tau: float, mode: str | None = None):
    regs = []
    if decorrelation_tau > 0:
        regs.append(DecorrelationRegularization(tau=decorrelation_tau))
    return regs if regs else None


def fit_topic_model(
    model,
    tokens: jax.Array | None = None,
    bounds: jax.Array | None = None,
    *,
    batches: Iterable[Batch] | None = None,
    num_attn_passes: int,
    max_iter: int,
    tol: float,
    seed: int,
    batch_size: int = 10000,
) -> float:
    t0 = perf_counter()

    if batches is None:
        if tokens is None or bounds is None:
            raise ValueError("Either batches or tokens+bounds must be provided.")

        if batch_size is None or batch_size <= 0:
            batch_size = int(len(tokens))

        batches = TokenBatchLoader(
            tokens,
            bounds,
            batch_size=batch_size,
            pad_token_id=0,
        )

    model.fit(
        batches,
        num_attn_passes=num_attn_passes,
        max_iter=max_iter,
        tol=tol,
        verbose=0,
        seed=seed,
    )

    return perf_counter() - t0


def infer_token_topics_aartm(
    model: AttentiveTopicModel,
    tokens: jax.Array,
    bounds: jax.Array,
    *,
    num_attn_passes: int = 1,
) -> np.ndarray:
    token_mask = jnp.ones(tokens.shape, dtype=jnp.bool_)
    p_it = norm(model.phi[tokens], axis=1)
    for _ in range(num_attn_passes):
        theta = calc_attn(
            matrix=p_it,
            ctx_bounds=bounds,
            ctx_weights=model.context_weights,
            token_mask=token_mask,
        )
        p_it = norm(p_it * theta / (model.n_t + EPSILON), axis=1)
    return np.asarray(jax.device_get(p_it))


def truncate_corpus(
    tokens: jax.Array,
    bounds: jax.Array,
    *,
    max_tokens_per_doc: int,
) -> tuple[jax.Array, jax.Array]:
    tokens_np = np.asarray(tokens, dtype=np.int32)
    bounds_np = np.asarray(bounds, dtype=bool)

    spans = doc_spans(bounds_np, len(tokens_np))
    new_tokens = []
    new_boundaries = []

    for start, end in spans:
        truncated = tokens_np[start:min(end, start + max_tokens_per_doc)]
        if len(truncated) == 0:
            continue
        if len(new_tokens) > 0:
            new_boundaries.append(len(new_tokens))
        new_tokens.extend(truncated.tolist())

    new_tokens = np.asarray(new_tokens, dtype=np.int32)
    new_bounds = np.zeros(len(new_tokens), dtype=bool)
    if len(new_boundaries) > 0:
        new_bounds[np.asarray(new_boundaries, dtype=np.int32)] = True

    return (
        jnp.asarray(new_tokens, dtype=jnp.int32),
        jnp.asarray(new_bounds, dtype=jnp.bool_),
    )


def tokenize_docs_with_vocab(texts, labels, loader):
    vocab = loader.vocabulary
    assert vocab is not None

    docs = []
    y = []
    for text, label in zip(texts, labels):
        toks = loader.process_doc(text)
        toks = [tok for tok in toks if tok in vocab]
        if len(toks) == 0:
            continue
        docs.append(toks)
        y.append(label)

    return docs, np.asarray(y)


def make_synthetic_boundary_dataset(
    texts,
    labels,
    loader,
    *,
    per_side_tokens: int = 64,
    n_pairs: int = 500,
    seed: int = 0,
):
    tokenized_docs, y = tokenize_docs_with_vocab(texts, labels, loader)

    by_class = {}
    n_passed = 0
    for toks, label in zip(tokenized_docs, y):
        if len(toks) >= per_side_tokens:
            n_passed += 1
            by_class.setdefault(int(label), []).append(toks)
    print(f"=== Passed documents: {n_passed} ===")

    classes = [c for c, docs in by_class.items() if len(docs) > 0]
    if len(classes) < 2:
        raise RuntimeError("Need at least two non-empty classes for boundary dataset.")

    rng = np.random.default_rng(seed)
    mixed_docs = []
    true_boundaries = []

    for _ in range(n_pairs):
        c1, c2 = rng.choice(classes, size=2, replace=False)
        left = by_class[c1][rng.integers(len(by_class[c1]))][:per_side_tokens]
        right = by_class[c2][rng.integers(len(by_class[c2]))][:per_side_tokens]

        mixed_docs.append(left + right)
        true_boundaries.append(len(left))

    vocab = loader.vocabulary
    assert vocab is not None

    flat_tokens: list[int] = []
    boundary_positions: list[int] = []

    for doc in mixed_docs:
        if len(flat_tokens) > 0:
            boundary_positions.append(len(flat_tokens))

        flat_tokens.extend([vocab[token] for token in doc])

    flat_tokens_np = np.asarray(flat_tokens, dtype=np.int32)
    bounds_np = np.zeros(len(flat_tokens_np), dtype=bool)

    if len(boundary_positions) > 0:
        bounds_np[np.asarray(boundary_positions, dtype=np.int32)] = True

    return (
        mixed_docs,
        jnp.asarray(flat_tokens_np, dtype=jnp.int32),
        jnp.asarray(bounds_np, dtype=jnp.bool_),
        np.asarray(true_boundaries, dtype=np.int32),
    )


def hellinger_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sqrt(((np.sqrt(p) - np.sqrt(q)) ** 2).sum() / 2.0))


def evaluate_boundary_detection(
    token_topics: np.ndarray,
    bounds: jax.Array,
    true_boundaries: np.ndarray,
    *,
    window: int = 16,
    tolerances: tuple[int, ...] = (5, 10),
):
    spans = doc_spans(bounds, token_topics.shape[0])

    pred_boundaries = []
    abs_errors = []
    hit_counts = {tol: 0 for tol in tolerances}

    for doc_id, (start, end) in enumerate(spans):
        doc_topics = token_topics[start:end]
        doc_len = len(doc_topics)

        if doc_len < 2 * window + 1:
            continue

        scores = np.full(doc_len, -np.inf, dtype=np.float32)
        for pos in range(window, doc_len - window):
            left = doc_topics[pos - window:pos].mean(axis=0)
            right = doc_topics[pos:pos + window].mean(axis=0)

            left = left / max(left.sum(), 1e-12)
            right = right / max(right.sum(), 1e-12)

            scores[pos] = hellinger_distance(left, right)

        pred = int(np.argmax(scores))
        true = int(true_boundaries[doc_id])

        pred_boundaries.append(pred)
        abs_errors.append(abs(pred - true))
        for tol in tolerances:
            if abs(pred - true) <= tol:
                hit_counts[tol] += 1

    n = max(len(abs_errors), 1)
    result = {
        "boundary_mae": float(np.mean(abs_errors)) if abs_errors else float("nan"),
        "n_eval_docs": int(len(abs_errors)),
    }
    for tol in tolerances:
        result[f"boundary_hit@{tol}"] = hit_counts[tol] / n

    return result

