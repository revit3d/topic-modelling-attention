from __future__ import annotations

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from experiments.common import classification_scores


def phi_to_topic_words(
    phi_wt: np.ndarray,
    id2word: dict[int, str],
    top_k: int = 25,
) -> list[list[str]]:
    k = min(top_k, phi_wt.shape[0])
    idx = np.argpartition(phi_wt, -k, axis=0)[-k:]  # (k, T)
    scores = np.take_along_axis(phi_wt, idx, axis=0)
    order = np.argsort(-scores, axis=0)
    idx = np.take_along_axis(idx, order, axis=0)  # (k, T)

    topics = []
    for t in range(idx.shape[1]):
        topics.append([id2word[int(w)] for w in idx[:, t]])
    return topics


def save_topic_words_list(
    topic_words: list[list[str]],
    out_path: str | Path,
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, words in enumerate(topic_words):
            f.write(f"Topic {i}\t" + ", ".join(words) + "\n")


def topic_diversity_from_topic_words(
    topic_words: list[list[str]],
    top_k: int = 25,
) -> float:
    used = [w for topic in topic_words for w in topic[:top_k]]
    if len(used) == 0:
        return float("nan")
    return float(len(set(used)) / len(used))


def npmi_from_topic_words(
    topic_words: list[list[str]],
    vocab: dict[str, int],
    bow: sp.csr_matrix,
    top_k: int = 10,
) -> float:
    bow = bow.sign().astype(np.uint8).tocsc(copy=False)
    df = np.asarray(bow.getnnz(axis=0)).ravel().astype(np.float32)
    n_docs = bow.shape[0]

    topic_scores = []
    for topic in topic_words:
        ids = [vocab[w] for w in topic[:top_k] if w in vocab]
        ids = list(dict.fromkeys(ids))
        if len(ids) < 2:
            continue

        X_sub = bow[:, ids]
        cooc = (X_sub.T @ X_sub).toarray().astype(np.float32)

        p_i = df[ids] / n_docs
        p_ij = cooc / n_docs
        denom = p_i[:, None] * p_i[None, :]

        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log(p_ij / denom)
            npmi = pmi / (-np.log(p_ij))

        npmi = np.where(cooc > 0, npmi, -1.0)
        triu = np.triu_indices(len(ids), k=1)
        topic_scores.append(float(npmi[triu].mean()))

    if len(topic_scores) == 0:
        return float("nan")
    return float(np.mean(topic_scores))


def evaluate_topic_words_and_doc_topics(
    *,
    topic_words: list[list[str]],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_bow: sp.csr_matrix,
    vocab: dict[str, int],
    seed: int,
) -> dict[str, float]:
    metrics = {
        "npmi_10": npmi_from_topic_words(topic_words, vocab, train_bow, top_k=10),
        "topic_diversity_25": topic_diversity_from_topic_words(topic_words, top_k=25),
    }
    metrics.update(
        classification_scores(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            seed=seed,
        )
    )
    return metrics
