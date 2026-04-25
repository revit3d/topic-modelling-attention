from __future__ import annotations

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

from cartm import AttentiveTopicModel, ContextTopicModel
from cartm.core import EPSILON, norm, calc_attn
from cartm.preprocessing import CorpusLoader, build_bow
from experiments.paper_utils import DocumentBatchedCorpusLoader
from cartm.regularization import DecorrelationRegularization

try:
    from datasets import load_dataset as hf_load_dataset
except ImportError:
    hf_load_dataset = None


@dataclass
class PreparedData:
    dataset_name: str
    train_texts: list[str]                 # raw original
    test_texts: list[str]                  # raw original
    train_texts_filtered: list[str]        # aligned with y_train / train_tokens
    test_texts_filtered: list[str]         # aligned with y_test / test_tokens
    y_train: np.ndarray
    y_test: np.ndarray
    loader: CorpusLoader
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

    if hf_load_dataset is None:
        raise ImportError(
            "The `datasets` package is required for ag_news and dbpedia14. "
            "Install it with `pip install datasets`."
        )

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
    loader: CorpusLoader,
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

    loader = CorpusLoader(
        lower=True,
        min_df=min_df,
        max_df=max_df,
        min_token_len=min_token_len,
        max_token_len=max_token_len,
    )
    loader.fit(train_texts)

    train_texts_filtered, tokenized_train, y_train = _tokenize_and_filter_empty_docs(
        train_texts, y_train, loader
    )
    test_texts_filtered, tokenized_test, y_test = _tokenize_and_filter_empty_docs(
        test_texts, y_test, loader
    )

    train_tokens, train_bounds = loader._transform_impl(
        tokenized_train,
        return_doc_bounds=True,
        preprocess=False,
    )
    test_tokens, test_bounds = loader._transform_impl(
        tokenized_test,
        return_doc_bounds=True,
        preprocess=False,
    )

    vocab = loader.vocabulary
    assert vocab is not None
    vocab_size = len(vocab)
    id2word = {v: k for k, v in vocab.items()}

    train_bow = build_bow(train_tokens, train_bounds, vocab_size)
    test_bow = build_bow(test_tokens, test_bounds, vocab_size)

    tfidf = TfidfTransformer(norm="l2")
    train_tfidf = tfidf.fit_transform(train_bow)
    test_tfidf = tfidf.transform(test_bow)

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
    )


def doc_spans(bounds: np.ndarray | jax.Array, n_tokens: int) -> list[tuple[int, int]]:
    bounds = np.asarray(bounds, dtype=bool)
    starts = np.r_[0, np.flatnonzero(bounds)]
    ends = np.r_[np.flatnonzero(bounds), n_tokens]
    return list(zip(starts, ends))


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
    tokens: jax.Array,
    bounds: jax.Array,
    *,
    num_attn_passes: int = 1,
) -> np.ndarray:
    p_it = norm(model.phi[tokens], axis=1)
    for _ in range(num_attn_passes):
        theta = calc_attn(
            matrix=p_it,
            ctx_bounds=bounds,
            ctx_weights=model.context_weights,
        )
        p_it = norm(p_it * theta / (model.n_t + EPSILON), axis=1)
    return aggregate_doc_topics(np.asarray(jax.device_get(p_it)), bounds)


def infer_doc_topics_cartm(
    model: ContextTopicModel,
    tokens: jax.Array,
    bounds: jax.Array,
    *,
    num_attn_passes: int = 1,
) -> np.ndarray:
    p_it = norm(model.phi[tokens] * model.n_t, axis=1)
    for _ in range(num_attn_passes):
        theta = calc_attn(
            matrix=p_it,
            ctx_bounds=bounds,
            ctx_weights=model.context_weights,
        )
        p_it = norm(p_it * theta / (model.n_t + EPSILON), axis=1)
    return aggregate_doc_topics(np.asarray(jax.device_get(p_it)), bounds)


def aartm_phi_pwt(model: AttentiveTopicModel, train_tokens: jax.Array) -> np.ndarray:
    phi_wt, _ = model.renormalize_phi(batch=train_tokens, phi=model.phi)
    return np.asarray(jax.device_get(phi_wt))


def cartm_phi_pwt(model: ContextTopicModel) -> np.ndarray:
    return np.asarray(jax.device_get(model.phi))


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
        multi_class="auto",
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
    regs = []
    if decorrelation_tau > 0:
        regs.append(DecorrelationRegularization(tau=decorrelation_tau))

    model = AttentiveTopicModel(
        vocab_size=len(data.vocab),
        ctx_len=ctx_len,
        n_topics=n_topics,
        gamma=gamma,
        self_aware_context=self_aware_context,
        regularizers=regs if regs else None,
    )

    if batch_size is not None and batch_size > 0:
        batches = DocumentBatchedCorpusLoader(
            data.train_tokens,
            data.train_bounds,
            batch_size=batch_size,
        )

    t0 = perf_counter()
    if batch_size is not None and batch_size > 0:
        model.fit(
            data=batches,
            ctx_bounds=None,
            num_attn_passes=num_attn_passes,
            max_iter=max_iter,
            tol=tol,
            seed=seed,
            verbose=0,
        )
    else:
        model.fit(
            data=data.train_tokens,
            ctx_bounds=data.train_bounds,
            num_attn_passes=num_attn_passes,
            max_iter=max_iter,
            tol=tol,
            seed=seed,
            verbose=0,
        )
    elapsed = perf_counter() - t0
    return model, elapsed


def fit_cartm(
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
) -> tuple[ContextTopicModel, float]:
    regs = []
    if decorrelation_tau > 0:
        regs.append(DecorrelationRegularization(tau=decorrelation_tau))

    model = ContextTopicModel(
        vocab_size=len(data.vocab),
        ctx_len=ctx_len,
        n_topics=n_topics,
        gamma=gamma,
        self_aware_context=self_aware_context,
        regularizers=regs if regs else None,
    )

    t0 = perf_counter()
    if batch_size is not None and batch_size > 0:
        batches = DocumentBatchedCorpusLoader(
            data.train_tokens,
            data.train_bounds,
            batch_size=batch_size,
        )
        model.fit(
            data=batches,
            ctx_bounds=None,
            num_attn_passes=num_attn_passes,
            max_iter=max_iter,
            tol=tol,
            seed=seed,
            verbose=0,
        )
    else:
        model.fit(
            data=data.train_tokens,
            ctx_bounds=data.train_bounds,
            num_attn_passes=num_attn_passes,
            max_iter=max_iter,
            tol=tol,
            seed=seed,
            verbose=0,
        )
    elapsed = perf_counter() - t0
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
    num_attn_passes: int,
    seed: int,
) -> dict[str, float]:
    phi_wt = aartm_phi_pwt(model, data.train_tokens)
    X_train = infer_doc_topics_aartm(model, data.train_tokens, data.train_bounds, num_attn_passes=num_attn_passes)
    X_test = infer_doc_topics_aartm(model, data.test_tokens, data.test_bounds, num_attn_passes=num_attn_passes)

    metrics = {
        "npmi_10": npmi_score(phi_wt, data.train_bow, top_k=10),
        "topic_diversity_25": topic_diversity(phi_wt, top_k=25),
        "topic_sparsity": topic_sparsity(phi_wt),
        "topic_hellinger": mean_nearest_hellinger(phi_wt),
    }
    metrics.update(classification_scores(X_train, data.y_train, X_test, data.y_test, seed=seed))
    return metrics


def evaluate_cartm(
    model: ContextTopicModel,
    data: PreparedData,
    *,
    num_attn_passes: int,
    seed: int,
) -> dict[str, float]:
    phi_wt = cartm_phi_pwt(model)
    X_train = infer_doc_topics_cartm(model, data.train_tokens, data.train_bounds, num_attn_passes=num_attn_passes)
    X_test = infer_doc_topics_cartm(model, data.test_tokens, data.test_bounds, num_attn_passes=num_attn_passes)

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
