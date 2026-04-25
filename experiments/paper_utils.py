from __future__ import annotations

from time import perf_counter
from typing import Iterable

import numpy as np
import jax
import jax.numpy as jnp
from sklearn.feature_extraction.text import TfidfTransformer

from cartm import AttentiveTopicModel, ContextTopicModel
from cartm.core import EPSILON, norm, calc_attn
from cartm.preprocessing import build_bow
from cartm.regularization import DecorrelationRegularization
from experiments.common import doc_spans


def parse_df_arg(x: str):
    return float(x) if "." in x else int(x)


class DocumentBatchedCorpusLoader:
    """
    Batch loader that preserves whole documents inside a batch.
    This is preferable for context-based models.
    """

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


def build_regularizers(decorrelation_tau: float):
    regs = []
    if decorrelation_tau > 0:
        regs.append(DecorrelationRegularization(tau=decorrelation_tau))
    return regs if regs else None


def fit_topic_model(
    model,
    tokens: jax.Array,
    bounds: jax.Array,
    *,
    num_attn_passes: int,
    max_iter: int,
    tol: float,
    seed: int,
    batch_size: int = -1,
) -> float:
    t0 = perf_counter()

    if batch_size is not None and batch_size > 0:
        batches = DocumentBatchedCorpusLoader(tokens, bounds, batch_size=batch_size)
        model.fit(
            data=batches,
            ctx_bounds=None,
            num_attn_passes=num_attn_passes,
            max_iter=max_iter,
            tol=tol,
            verbose=0,
            seed=seed,
        )
    else:
        model.fit(
            data=tokens,
            ctx_bounds=bounds,
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
    p_it = norm(model.phi[tokens], axis=1)
    for _ in range(num_attn_passes):
        theta = calc_attn(
            matrix=p_it,
            ctx_bounds=bounds,
            ctx_weights=model.context_weights,
        )
        p_it = norm(p_it * theta / (model.n_t + EPSILON), axis=1)
    return np.asarray(jax.device_get(p_it))


def infer_token_topics_cartm(
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


def build_truncated_bow_tfidf(
    tokens: jax.Array,
    bounds: jax.Array,
    vocab_size: int,
):
    bow = build_bow(tokens, bounds, vocab_size)
    tfidf = TfidfTransformer(norm="l2")
    tfidf_mat = tfidf.fit_transform(bow)
    return bow, tfidf_mat


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
    for toks, label in zip(tokenized_docs, y):
        if len(toks) >= per_side_tokens:
            by_class.setdefault(int(label), []).append(toks)

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

    tokens, bounds = loader._transform_impl(
        mixed_docs,
        return_doc_bounds=True,
        preprocess=False,
    )

    return mixed_docs, tokens, bounds, np.asarray(true_boundaries, dtype=np.int32)


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
