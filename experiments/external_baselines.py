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


# -------------------- BERTopic --------------------
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


# -------------------- CTM --------------------
def fit_combined_tm(
    data,
    *,
    n_topics: int,
    seed: int,
    embedding_model_name: str = "all-MiniLM-L6-v2",
):
    from sentence_transformers import SentenceTransformer
    from contextualized_topic_models.models.ctm import CombinedTM
    from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
    import torch

    torch.manual_seed(seed)

    train_bow_text = preprocessed_docs(data.loader, data.train_texts_filtered)
    test_bow_text = preprocessed_docs(data.loader, data.test_texts_filtered)

    encoder = SentenceTransformer(embedding_model_name)
    contextual_size = encoder.get_embedding_dimension()

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
        num_epochs=10,
        batch_size=64,
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


# -------------------- BTM --------------------
def fit_btm(
    data,
    *,
    n_topics: int,
    seed: int,
    num_iterations: int = 100,
    window: int = 3,
):
    import bitermplus as btm
    import numpy as np

    train_prep = preprocessed_docs(data.loader, data.train_texts_filtered)
    test_prep  = preprocessed_docs(data.loader, data.test_texts_filtered)

    X, vocab, vocab_dict = btm.get_words_freqs(train_prep)
    docs_vec_train = btm.get_vectorized_docs(train_prep, vocab)
    docs_vec_test  = btm.get_vectorized_docs(test_prep,  vocab)
    biterms = btm.get_biterms(docs_vec_train, win=window)

    model = btm.BTM(X, vocab, T=n_topics, M=20, alpha=50.0/n_topics,
                    beta=0.01, seed=seed)
    t0 = perf_counter()
    model.fit(biterms, iterations=num_iterations, verbose=True)
    elapsed = perf_counter() - t0

    cache = {
        "vocab": vocab,
        "vocab_dict": vocab_dict,
        "docs_vec_train": docs_vec_train,
        "docs_vec_test":  docs_vec_test,
    }
    return model, elapsed, cache


def btm_topic_words(model, top_k: int = 25) -> list[list[str]]:
    import bitermplus as btm
    return btm.get_top_topic_words(model, words_num=top_k).T.values.tolist()


def btm_doc_topics(model, docs_vec) -> "np.ndarray":
    return normalize_rows(model.transform(docs_vec))


# -------------------- BigARTM --------------------
def fit_bigartm(
    data,
    *,
    n_topics: int,
    seed: int,
    max_iter: int = 50,
    decorrelation_tau: float = 0.0,
    sparsity_tau: float = 0.0,
):
    import artm
    import os, tempfile, numpy as np, scipy.sparse as sp
    from scipy.sparse import coo_matrix

    tmpdir = tempfile.mkdtemp(prefix="bigartm_")
    vw_path = os.path.join(tmpdir, "train.vw")
    id2word = data.id2word
    coo = data.train_bow.tocoo()
    docs: dict[int, list[str]] = {}
    for r, c, v in zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist()):
        docs.setdefault(r, []).append(f"{id2word[int(c)]}:{int(v)}")
    with open(vw_path, "w", encoding="utf-8") as f:
        for r in range(data.train_bow.shape[0]):
            tokens = docs.get(r, [])
            f.write(f"doc{r} |@default_class " + " ".join(tokens) + "\n")

    bv = artm.BatchVectorizer(data_path=vw_path, data_format="vowpal_wabbit",
                              target_folder=os.path.join(tmpdir, "batches"))

    dictionary = artm.Dictionary()
    dictionary.gather(data_path=bv.data_path)

    model = artm.ARTM(
        num_topics=n_topics,
        dictionary=dictionary,
        seed=seed,
        cache_theta=False,
    )
    model.scores.add(artm.PerplexityScore(name="perp", dictionary=dictionary))
    if decorrelation_tau > 0:
        model.regularizers.add(artm.DecorrelatorPhiRegularizer(
            name="decorr", tau=decorrelation_tau))
    if sparsity_tau > 0:
        model.regularizers.add(artm.SmoothSparsePhiRegularizer(
            name="sparse", tau=-abs(sparsity_tau)))

    t0 = perf_counter()
    model.fit_offline(batch_vectorizer=bv, num_collection_passes=max_iter)
    elapsed = perf_counter() - t0

    cache = {"batches_dir": os.path.join(tmpdir, "batches"),
             "tmpdir": tmpdir, "id2word": id2word}
    return model, elapsed, cache


def bigartm_topic_words(model, top_k: int = 25) -> list[list[str]]:
    phi = model.get_phi()
    topics = []
    for col in phi.columns:
        topics.append(phi[col].sort_values(ascending=False).head(top_k).index.tolist())
    return topics


def bigartm_doc_topics(model, data, split: str = "train") -> "np.ndarray":
    import artm, os, tempfile, numpy as np
    bow = data.train_bow if split == "train" else data.test_bow
    coo = bow.tocoo()
    tmpdir = tempfile.mkdtemp(prefix="bigartm_inf_")
    vw_path = os.path.join(tmpdir, f"{split}.vw")
    docs: dict[int, list[str]] = {}
    for r, c, v in zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist()):
        docs.setdefault(r, []).append(f"{data.id2word[int(c)]}:{int(v)}")
    with open(vw_path, "w", encoding="utf-8") as f:
        for r in range(bow.shape[0]):
            f.write(f"doc{r} |@default_class " + " ".join(docs.get(r, [])) + "\n")
    bv = artm.BatchVectorizer(data_path=vw_path, data_format="vowpal_wabbit",
                              target_folder=os.path.join(tmpdir, "batches"))
    theta = model.transform(batch_vectorizer=bv)  # (T, D)
    return normalize_rows(np.asarray(theta.T, dtype=np.float32))


# -------------------- Contextual Top2Vec --------------------
def fit_contextual_top2vec(
    data,
    *,
    n_topics: int,
    seed: int,
    embedding_model_name: str = "all-MiniLM-L6-v2",
):
    import numpy as np
    import random, os

    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    from top2vec import Top2Vec

    builtin = {
        "universal-sentence-encoder",
        "universal-sentence-encoder-multilingual",
        "distiluse-base-multilingual-cased",
    }
    if embedding_model_name in builtin:
        embedding_model = embedding_model_name
        embedding_callable = None
    else:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(embedding_model_name)
        embedding_model = "custom"
        def embedding_callable(texts):
            return encoder.encode(list(texts), show_progress_bar=False,
                                  normalize_embeddings=True)

    t0 = perf_counter()
    kwargs = dict(
        documents=data.train_texts_filtered,
        speed="learn",
        workers=4,
        min_count=10,
        embedding_model=embedding_model,
    )
    if embedding_callable is not None:
        kwargs["embedding_model"] = embedding_callable
    model = Top2Vec(**kwargs)
    if model.get_num_topics() > n_topics:
        model.hierarchical_topic_reduction(num_topics=n_topics)
    elapsed = perf_counter() - t0

    return model, elapsed, {}


def top2vec_topic_words(model, top_k: int = 25) -> list[list[str]]:
    reduced = getattr(model, "topic_words_reduced", None)
    words = reduced if reduced is not None else model.topic_words
    return [list(w[:top_k]) for w in words]


def top2vec_doc_topics(model, data, split: str = "train") -> "np.ndarray":
    """
    Top2Vec is centroid-based, so it has no per-doc topic distribution.
    We approximate it by cosine similarity between doc vectors and topic vectors.
    """
    import numpy as np
    texts = data.train_texts_filtered if split == "train" else data.test_texts_filtered
    use_reduced = hasattr(model, "topic_vectors_reduced") and model.topic_vectors_reduced is not None
    topic_vecs = model.topic_vectors_reduced if use_reduced else model.topic_vectors

    if split == "train":
        doc_vecs = model.document_vectors
    else:
        if hasattr(model, "embed"):
            doc_vecs = model.embed(texts)
        else:
            doc_vecs = model._embed_documents(texts)

    def _l2(x):
        n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1.0
        return x / n
    sim = _l2(doc_vecs) @ _l2(topic_vecs).T          # cosine in [-1, 1]
    sim = np.maximum(sim, 0)                          # clip negatives
    return normalize_rows(sim.astype(np.float32))
