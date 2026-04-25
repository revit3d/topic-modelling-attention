from __future__ import annotations

from time import perf_counter
import numpy as np


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    denom = x.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return x / denom


def preprocessed_docs(loader, texts: list[str]) -> list[str]:
    return [" ".join(loader.process_doc(text)) for text in texts]


# ---------------------------
# BERTopic
# ---------------------------

def fit_bertopic(
    data,
    *,
    n_topics: int,
    seed: int,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    top_n_words: int = 25,
):
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP
    from hdbscan import HDBSCAN

    train_prep = preprocessed_docs(data.loader, data.train_texts_filtered)
    test_prep = preprocessed_docs(data.loader, data.test_texts_filtered)

    encoder = SentenceTransformer(embedding_model_name)
    train_embeddings = encoder.encode(
        data.train_texts_filtered,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    vectorizer = CountVectorizer(
        vocabulary=data.vocab,
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
    )

    model = BERTopic(
        embedding_model=encoder,
        vectorizer_model=vectorizer,
        umap_model=UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=seed,
        ),
        hdbscan_model=HDBSCAN(
            min_cluster_size=15,
            prediction_data=True,
        ),
        nr_topics=n_topics,
        top_n_words=top_n_words,
        calculate_probabilities=False,
        verbose=False,
    )

    t0 = perf_counter()
    model.fit_transform(train_prep, embeddings=train_embeddings)
    elapsed = perf_counter() - t0

    cache = {
        "train_docs": train_prep,
        "test_docs": test_prep,
    }
    return model, elapsed, cache


def bertopic_topic_words(model, top_k: int = 25) -> list[list[str]]:
    topic_ids = sorted(t for t in model.get_topics().keys() if t != -1)
    topics = []
    for tid in topic_ids:
        words = model.get_topic(tid) or []
        topics.append([w for w, _ in words[:top_k]])
    return topics


def bertopic_doc_topics(model, docs: list[str]) -> np.ndarray:
    distr, _ = model.approximate_distribution(docs)
    return normalize_rows(np.asarray(distr, dtype=np.float32))


# ---------------------------
# CombinedTM / CTM
# ---------------------------

def fit_combined_tm(
    data,
    *,
    n_topics: int,
    seed: int,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    num_epochs: int = 50,
):
    from sentence_transformers import SentenceTransformer
    from contextualized_topic_models.models.ctm import CombinedTM
    from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation

    # CTM best practice: raw text for contextual branch, preprocessed for bow branch
    train_bow_text = preprocessed_docs(data.loader, data.train_texts_filtered)
    test_bow_text = preprocessed_docs(data.loader, data.test_texts_filtered)

    encoder = SentenceTransformer(embedding_model_name)
    contextual_size = encoder.get_sentence_embedding_dimension()

    qt = TopicModelDataPreparation(embedding_model_name)
    training_dataset = qt.fit(
        text_for_contextual=data.train_texts_filtered,
        text_for_bow=train_bow_text,
    )
    testing_dataset = qt.transform(
        text_for_contextual=data.test_texts_filtered,
        text_for_bow=test_bow_text,
    )

    model = CombinedTM(
        bow_size=len(qt.vocab),
        contextual_size=contextual_size,
        n_components=n_topics,
        num_epochs=num_epochs,
    )

    t0 = perf_counter()
    model.fit(training_dataset)
    elapsed = perf_counter() - t0

    cache = {
        "train_dataset": training_dataset,
        "test_dataset": testing_dataset,
    }
    return model, elapsed, cache


def ctm_topic_words(model, top_k: int = 25) -> list[list[str]]:
    topics = model.get_topics(top_k)
    if isinstance(topics, dict):
        return [topics[k] for k in sorted(topics.keys())]
    if isinstance(topics, list):
        return topics
    raise TypeError(f"Unsupported CTM topics type: {type(topics)}")


def ctm_doc_topics(model, dataset) -> np.ndarray:
    theta = model.get_doc_topic_distribution(dataset, n_samples=20)
    return normalize_rows(np.asarray(theta, dtype=np.float32))
