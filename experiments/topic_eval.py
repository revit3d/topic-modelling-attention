from __future__ import annotations

from pathlib import Path
import warnings
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


def topic_eval_texts_from_data(data) -> list[list[str]]:
    """
    Tokenized train texts for topic-word metrics such as C_v.

    Cached on PreparedData instance because main-table evaluation calls this
    once per model/seed.
    """
    cache_name = "_topic_eval_train_texts"
    cached = getattr(data, cache_name, None)
    if cached is not None:
        return cached

    vocab = data.vocab
    texts = [
        [tok for tok in data.loader.process_doc(text) if tok in vocab]
        for text in data.train_texts_filtered
    ]
    texts = [text for text in texts if len(text) > 0]
    setattr(data, cache_name, texts)
    return texts


def _clean_topic_words(
    topic: list[str],
    *,
    top_k: int,
    vocabulary: set[str] | None = None,
) -> list[str]:
    words = []
    seen = set()

    for word in topic:
        if vocabulary is not None and word not in vocabulary:
            continue
        if word in seen:
            continue

        words.append(word)
        seen.add(word)

        if len(words) >= top_k:
            break

    return words


def c_v_coherence_from_topic_words(
    topic_words: list[list[str]],
    texts: list[list[str]] | None,
    *,
    top_k: int = 10,
    processes: int = 1,
) -> float:
    """
    C_v coherence from top words using gensim.

    Returns NaN if gensim is missing or if no valid topics/texts are available.
    """
    if texts is None or len(texts) == 0 or top_k <= 1:
        return float("nan")

    try:
        from gensim.corpora import Dictionary
        from gensim.models.coherencemodel import CoherenceModel
    except ImportError:
        warnings.warn(
            "gensim is not installed; returning NaN for C_v coherence. "
            "Install with: pip install gensim",
            stacklevel=2,
        )
        return float("nan")

    dictionary = Dictionary(texts)
    vocabulary = set(dictionary.token2id.keys())

    topics = [
        words
        for topic in topic_words
        if len(
            words := _clean_topic_words(
                topic,
                top_k=top_k,
                vocabulary=vocabulary,
            )
        ) >= 2
    ]

    if len(topics) == 0:
        return float("nan")

    cm = CoherenceModel(
        topics=topics,
        texts=texts,
        dictionary=dictionary,
        coherence="c_v",
        topn=top_k,
        processes=processes,
    )
    return float(cm.get_coherence())


def bertscore_from_topic_words(
    topic_words: list[list[str]],
    *,
    top_k: int = 10,
    lang: str = "en",
    model_type: str | None = None,
    batch_size: int = 64,
    rescale_with_baseline: bool = False,
    device: str | None = None,
) -> float:
    """
    Semantic topic-word coherence via pairwise BERTScore F1.

    For each topic, compare all unordered pairs of top words in both directions,
    then average F1 over all pairs from all topics.
    """
    if top_k <= 1:
        return float("nan")

    try:
        from bert_score import score
    except ImportError:
        warnings.warn(
            "bert-score is not installed; returning NaN for BERTScore. "
            "Install with: pip install bert-score",
            stacklevel=2,
        )
        return float("nan")

    candidates: list[str] = []
    references: list[str] = []

    for topic in topic_words:
        words = _clean_topic_words(topic, top_k=top_k)
        if len(words) < 2:
            continue

        # Replace underscores just in case an external baseline emits phrases.
        words = [word.replace("_", " ") for word in words]

        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                candidates.append(words[i])
                references.append(words[j])
                candidates.append(words[j])
                references.append(words[i])

    if len(candidates) == 0:
        return float("nan")

    kwargs = {
        "batch_size": batch_size,
        "verbose": False,
        "rescale_with_baseline": rescale_with_baseline,
    }
    if model_type is not None:
        kwargs["model_type"] = model_type
    else:
        kwargs["lang"] = lang
    if device is not None:
        kwargs["device"] = device

    _, _, f1 = score(candidates, references, **kwargs)

    if hasattr(f1, "detach"):
        return float(f1.mean().detach().cpu().item())
    return float(np.mean(f1))


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
    cv_texts: list[list[str]] | None = None,
    bertscore_lang: str = "en",
    bertscore_model_type: str | None = None,
    bertscore_batch_size: int = 64,
    bertscore_rescale_with_baseline: bool = False,
    bertscore_device: str | None = None,
    seed: int,
) -> dict[str, float]:
    metrics = {
        "npmi_10": npmi_from_topic_words(topic_words, vocab, train_bow, top_k=10),
        "topic_diversity_25": topic_diversity_from_topic_words(topic_words, top_k=25),
        f"c_v_10": c_v_coherence_from_topic_words(topic_words, cv_texts, top_k=10),
        f"bertscore_f1_10": bertscore_from_topic_words(
            topic_words,
            top_k=10,
            lang=bertscore_lang,
            model_type=bertscore_model_type,
            batch_size=bertscore_batch_size,
            rescale_with_baseline=bertscore_rescale_with_baseline,
            device=bertscore_device,
        ),
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
