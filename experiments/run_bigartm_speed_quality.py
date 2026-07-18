"""
Speed / quality benchmark: AttentiveTopicModel vs batched BigARTM.

Both models are trained in a batched regime for the same number of collection
passes; wall-clock time is recorded per pass and quality is evaluated at
checkpoints OUTSIDE the timed region, producing speed-quality trajectories.

Quality metrics:
  - NPMI@10, topic diversity@25 (topic-word quality, both models)
  - held-out DOC-LEVEL perplexity computed the same way for both models
    (p(w|d) = sum_t p(w|t) p(t|d)), so the numbers are directly comparable
  - AARTM contextual perplexity (reported for reference, NOT comparable
    to the doc-level one)
  - C_v coherence (final checkpoint only, optional: --compute_cv)
  - logistic-regression accuracy / macro-F1 on doc-topic vectors
    (labeled datasets only)

Datasets:
  wikitext103  - ~28k long articles, ~103M raw tokens, no labels
  dbpedia14    - 630k short docs, 14 classes
  ag_news      - 127k short docs, 4 classes
  20ng         - small, for sanity checks

Example:
  python -m experiments.run_bigartm_speed_quality \
      --dataset wikitext103 --n_topics 100 --max_iter 30 --eval_every 5 \
      --batch_size 200000 --artm_batch_docs 1000 --seeds 0,1

  python -m experiments.run_bigartm_speed_quality \
      --dataset dbpedia14 --n_topics 100 --max_docs 200000
"""
from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import nltk
import numpy as np
import pandas as pd
import scipy.sparse as sp

from cartm import AttentiveTopicModel
from cartm.preprocessing import CorpusDataLoader, build_bow_from_loader
from experiments.common import (
    TokenBatchLoader,
    _tokenize_and_filter_empty_docs,
    aartm_perplexity,
    aartm_phi_pwt,
    classification_scores,
    doc_spans,
    flatten_loader_to_arrays,
    infer_doc_topics_aartm,
    load_text_classification_dataset,
    npmi_score,
    parse_df_arg,
    topic_diversity,
)
from experiments.topic_eval import (
    c_v_coherence_from_topic_words,
    phi_to_topic_words,
    save_topic_words_list,
)


# ----------------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------------

_WIKI_TOP_HEADING = re.compile(r"^\s*=\s[^=].*=\s*$")


def _group_wikitext_articles(lines: list[str]) -> list[str]:
    """WikiText-103 is line-based; articles start with a top-level ' = Title = '."""
    docs: list[str] = []
    current: list[str] = []
    for line in lines:
        if _WIKI_TOP_HEADING.match(line) and line.count("=") == 2:
            if current:
                docs.append(" ".join(current))
            current = [line]
        elif line.strip():
            current.append(line)
    if current:
        docs.append(" ".join(current))
    return docs


def load_dataset(name: str):
    """Returns (train_texts, test_texts, y_train | None, y_test | None)."""
    if name == "wikitext103":
        from datasets import load_dataset as hf_load_dataset

        ds = hf_load_dataset("wikitext", "wikitext-103-raw-v1")
        train = _group_wikitext_articles(list(ds["train"]["text"]))
        # official test split is tiny (~60 articles) -> use validation + test
        test = _group_wikitext_articles(
            list(ds["validation"]["text"]) + list(ds["test"]["text"])
        )
        return train, test, None, None

    train_texts, test_texts, y_train, y_test = load_text_classification_dataset(name)
    return train_texts, test_texts, y_train, y_test


def prepare(args):
    train_texts, test_texts, y_train, y_test = load_dataset(args.dataset)
    labeled = y_train is not None

    rng = np.random.default_rng(0)
    if args.max_docs > 0 and len(train_texts) > args.max_docs:
        idx = rng.choice(len(train_texts), size=args.max_docs, replace=False)
        train_texts = [train_texts[i] for i in idx]
        if labeled:
            y_train = np.asarray(y_train)[idx]
    if args.max_test_docs > 0 and len(test_texts) > args.max_test_docs:
        idx = rng.choice(len(test_texts), size=args.max_test_docs, replace=False)
        test_texts = [test_texts[i] for i in idx]
        if labeled:
            y_test = np.asarray(y_test)[idx]

    loader = CorpusDataLoader(
        train_texts,
        lower=True,
        min_df=parse_df_arg(args.min_df),
        max_df=parse_df_arg(args.max_df),
        min_token_len=3,
        max_token_len=20,
        num_workers=args.num_workers,
        pad_token_id=0,
    )
    loader.fit()
    vocab = loader.vocabulary
    id2word = {v: k for k, v in vocab.items()}

    y_train_f = y_train if labeled else np.zeros(len(train_texts), dtype=np.int32)
    y_test_f = y_test if labeled else np.zeros(len(test_texts), dtype=np.int32)
    train_texts_f, _, y_train_f = _tokenize_and_filter_empty_docs(train_texts, y_train_f, loader)
    test_texts_f, _, y_test_f = _tokenize_and_filter_empty_docs(test_texts, y_test_f, loader)

    def split_loader(texts):
        return CorpusDataLoader(
            texts,
            lower=True,
            vocabulary=vocab,
            min_token_len=3,
            max_token_len=20,
            num_workers=args.num_workers,
            pad_token_id=0,
        )

    train_loader = split_loader(train_texts_f)
    test_loader = split_loader(test_texts_f)

    train_tokens, train_bounds = flatten_loader_to_arrays(train_loader)
    test_tokens, test_bounds = flatten_loader_to_arrays(test_loader)
    train_bow = build_bow_from_loader(train_loader)
    test_bow = build_bow_from_loader(test_loader)

    print("=== Prepared data ===")
    print(f"dataset={args.dataset} | labeled={labeled}")
    print(f"train docs: {len(train_texts_f)} | test docs: {len(test_texts_f)}")
    print(f"vocab: {len(vocab)} | train tokens: {len(train_tokens)} | test tokens: {len(test_tokens)}")

    return dict(
        labeled=labeled,
        vocab=vocab,
        id2word=id2word,
        loader=loader,
        train_loader=train_loader,
        y_train=y_train_f,
        y_test=y_test_f,
        train_tokens=train_tokens,
        train_bounds=train_bounds,
        test_tokens=test_tokens,
        test_bounds=test_bounds,
        train_bow=train_bow,
        test_bow=test_bow,
    )


# ----------------------------------------------------------------------------
# Shared quality metrics
# ----------------------------------------------------------------------------

def bow_perplexity(phi_wt: np.ndarray, theta: np.ndarray, bow: sp.csr_matrix) -> float:
    """Doc-level held-out perplexity, identical formula for both models."""
    bow = bow.tocsr()
    ll, n = 0.0, 0.0
    for d in range(bow.shape[0]):
        s, e = bow.indptr[d], bow.indptr[d + 1]
        if s == e:
            continue
        idx = bow.indices[s:e]
        cnt = bow.data[s:e].astype(np.float64)
        p = phi_wt[idx].astype(np.float64) @ theta[d].astype(np.float64)
        ll += float(cnt @ np.log(p + 1e-12))
        n += float(cnt.sum())
    return float(np.exp(-ll / n)) if n > 0 else float("nan")


def phi_quality(phi_wt: np.ndarray, train_bow: sp.csr_matrix) -> dict[str, float]:
    return {
        "npmi_10": npmi_score(phi_wt, train_bow, top_k=10),
        "topic_diversity_25": topic_diversity(phi_wt, top_k=25),
    }


def normalize_rows(x: np.ndarray) -> np.ndarray:
    denom = x.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return x / denom


def cv_texts_from_loader(train_loader, id2word) -> list[list[str]]:
    texts = []
    for doc in train_loader.iter_encoded_docs():
        if doc:
            texts.append([id2word[i] for i in doc])
    return texts


# ----------------------------------------------------------------------------
# AARTM: trajectory fit
# ----------------------------------------------------------------------------

def make_inference_batch_size(bounds, tokens, default: int) -> int:
    """Batch size large enough to hold the longest document (no splitting),
    so that doc-topic aggregation stays aligned with the BoW rows."""
    spans = doc_spans(bounds, len(tokens))
    max_len = max((e - s) for s, e in spans) if spans else 1
    return max(default, int(max_len))


def run_aartm(d, args, seed: int):
    model = AttentiveTopicModel(
        vocab_size=len(d["vocab"]),
        ctx_len=args.ctx_len,
        n_topics=args.n_topics,
        gamma=args.gamma,
        self_aware_context=args.self_aware_context,
    )

    train_batches = TokenBatchLoader(
        d["train_tokens"], d["train_bounds"],
        batch_size=args.batch_size,
        split_documents=True,
        pad_token_id=0,
    )

    infer_bs = make_inference_batch_size(d["test_bounds"], d["test_tokens"], args.batch_size)
    test_batches_ctx = TokenBatchLoader(
        d["test_tokens"], d["test_bounds"],
        batch_size=infer_bs, split_documents=False, pad_token_id=0,
    )

    # --- init (mirrors ModelBase.fit) ---
    n_w = jnp.zeros(model.vocab_size)
    for batch, _, mask in train_batches:
        n_w += jnp.bincount(batch, weights=mask.astype(jnp.float32), length=model.vocab_size)
    model.p_w = n_w / jnp.sum(n_w)
    model._init_state(seed=seed)
    grad_reg = model._compose_regularizations()

    n_train_tokens = int(len(d["train_tokens"]))
    rows, cum = [], 0.0

    for it in range(args.max_iter):
        t0 = perf_counter()
        phi_new, n_t_new = model._batched_step_wrapper(
            batches=train_batches,
            ctx_weights=model.context_weights,
            grad_reg=grad_reg,
            num_attn_passes=args.num_attn_passes,
            lr=args.lr,
            num_batches_before_update=args.num_batches_before_update,
        )
        phi_new.block_until_ready()
        dt = perf_counter() - t0
        cum += dt
        model.phi, model.n_t = phi_new, n_t_new

        last = it + 1 == args.max_iter
        if (it + 1) % args.eval_every == 0 or last:
            phi_wt = aartm_phi_pwt(model)
            rec = {
                "model": "AttentiveTopicModel",
                "seed": seed,
                "pass": it + 1,
                "pass_time_sec": dt,
                "cum_time_sec": cum,
                "tokens_per_sec": n_train_tokens / dt,
                **phi_quality(phi_wt, d["train_bow"]),
            }
            # held-out doc-level perplexity (comparable to BigARTM)
            theta_test = infer_doc_topics_aartm(
                model, test_batches_ctx, num_attn_passes=args.num_attn_passes
            )
            rec["heldout_doc_perplexity"] = bow_perplexity(phi_wt, theta_test, d["test_bow"])
            # contextual perplexity (AARTM-native, NOT comparable)
            rec["heldout_ctx_perplexity"] = aartm_perplexity(
                model, test_batches_ctx,
                num_attn_passes=args.num_attn_passes, phi_wt=phi_wt,
            )
            rows.append(rec)
            print(f"  [aartm] pass {it + 1}/{args.max_iter} "
                  f"t={cum:.1f}s npmi={rec['npmi_10']:.4f} "
                  f"ppl={rec['heldout_doc_perplexity']:.1f}")

    # --- final extras ---
    phi_wt = aartm_phi_pwt(model)
    final = rows[-1]
    topic_words = phi_to_topic_words(phi_wt, d["id2word"], top_k=25)
    save_topic_words_list(
        topic_words,
        Path(args.out_dir) / args.dataset / f"top_words_aartm_seed{seed}.txt",
    )
    if args.compute_cv:
        final["c_v_10"] = c_v_coherence_from_topic_words(
            d["cv_texts"], texts=d["cv_texts"], top_k=10,
        ) if False else c_v_coherence_from_topic_words(topic_words, d["cv_texts"], top_k=10)
    if d["labeled"]:
        infer_bs_tr = make_inference_batch_size(d["train_bounds"], d["train_tokens"], args.batch_size)
        X_train = infer_doc_topics_aartm(
            model,
            TokenBatchLoader(d["train_tokens"], d["train_bounds"],
                             batch_size=infer_bs_tr, pad_token_id=0),
            num_attn_passes=args.num_attn_passes,
        )
        X_test = infer_doc_topics_aartm(
            model, test_batches_ctx, num_attn_passes=args.num_attn_passes
        )
        final.update(classification_scores(X_train, d["y_train"], X_test, d["y_test"], seed=seed))
    return rows


# ----------------------------------------------------------------------------
# BigARTM: trajectory fit
# ----------------------------------------------------------------------------

def _write_vw(bow: sp.csr_matrix, id2word: dict[int, str], path: str):
    bow = bow.tocsr()
    with open(path, "w", encoding="utf-8") as f:
        for r in range(bow.shape[0]):
            s, e = bow.indptr[r], bow.indptr[r + 1]
            toks = " ".join(
                f"{id2word[int(c)]}:{int(v)}"
                for c, v in zip(bow.indices[s:e], bow.data[s:e])
            )
            f.write(f"doc{r} |@default_class {toks}\n")


def _make_bv(bow, id2word, tmp_root, name, batch_docs):
    import artm
    tmpdir = tempfile.mkdtemp(prefix=f"bigartm_{name}_", dir=tmp_root)
    vw = os.path.join(tmpdir, f"{name}.vw")
    _write_vw(bow, id2word, vw)
    return artm.BatchVectorizer(
        data_path=vw,
        data_format="vowpal_wabbit",
        batch_size=batch_docs,
        target_folder=os.path.join(tmpdir, "batches"),
    )


def _bigartm_phi_matrix(model, vocab) -> np.ndarray:
    phi_df = model.get_phi()  # index: token, columns: topic
    phi = np.zeros((len(vocab), phi_df.shape[1]), dtype=np.float32)
    vals = phi_df.values.astype(np.float32)
    for pos, tok in enumerate(phi_df.index):
        wid = vocab.get(str(tok))
        if wid is not None:
            phi[wid] = vals[pos]
    return phi  # (W, T), columns are p(w|t)


def _bigartm_theta(model, bv) -> np.ndarray:
    theta_df = model.transform(batch_vectorizer=bv)  # (T, D)
    order = np.argsort([int(str(c)[3:]) for c in theta_df.columns])
    return normalize_rows(theta_df.values.T[order].astype(np.float32))


def run_bigartm(d, args, seed: int, online: bool):
    import artm

    name = "BigARTM_online" if online else "BigARTM_offline"
    tmp_root = tempfile.mkdtemp(prefix="bigartm_bench_")
    train_bv = _make_bv(d["train_bow"], d["id2word"], tmp_root, "train", args.artm_batch_docs)
    test_bv = _make_bv(d["test_bow"], d["id2word"], tmp_root, "test", args.artm_batch_docs)

    dictionary = artm.Dictionary()
    dictionary.gather(data_path=train_bv.data_path)

    model = artm.ARTM(
        num_topics=args.n_topics,
        dictionary=dictionary,
        seed=seed,
        cache_theta=False,
        num_processors=args.artm_num_processors,
    )

    n_train_tokens = int(len(d["train_tokens"]))
    rows, cum = [], 0.0

    for it in range(args.max_iter):
        t0 = perf_counter()
        if online:
            model.fit_online(batch_vectorizer=train_bv,
                             update_every=args.artm_update_every)
        else:
            model.fit_offline(batch_vectorizer=train_bv, num_collection_passes=1)
        dt = perf_counter() - t0
        cum += dt

        last = it + 1 == args.max_iter
        if (it + 1) % args.eval_every == 0 or last:
            phi_wt = _bigartm_phi_matrix(model, d["vocab"])
            theta_test = _bigartm_theta(model, test_bv)
            rec = {
                "model": name,
                "seed": seed,
                "pass": it + 1,
                "pass_time_sec": dt,
                "cum_time_sec": cum,
                "tokens_per_sec": n_train_tokens / dt,
                **phi_quality(phi_wt, d["train_bow"]),
                "heldout_doc_perplexity": bow_perplexity(phi_wt, theta_test, d["test_bow"]),
                "heldout_ctx_perplexity": float("nan"),
            }
            rows.append(rec)
            print(f"  [{name}] pass {it + 1}/{args.max_iter} "
                  f"t={cum:.1f}s npmi={rec['npmi_10']:.4f} "
                  f"ppl={rec['heldout_doc_perplexity']:.1f}")

    # --- final extras ---
    phi_wt = _bigartm_phi_matrix(model, d["vocab"])
    final = rows[-1]
    topic_words = phi_to_topic_words(phi_wt, d["id2word"], top_k=25)
    save_topic_words_list(
        topic_words,
        Path(args.out_dir) / args.dataset / f"top_words_{name}_seed{seed}.txt",
    )
    if args.compute_cv:
        final["c_v_10"] = c_v_coherence_from_topic_words(topic_words, d["cv_texts"], top_k=10)
    if d["labeled"]:
        X_train = _bigartm_theta(model, train_bv)
        X_test = _bigartm_theta(model, test_bv)
        final.update(classification_scores(X_train, d["y_train"], X_test, d["y_test"], seed=seed))
    return rows


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="wikitext103",
                   choices=["wikitext103", "dbpedia14", "ag_news", "20ng"])
    p.add_argument("--out_dir", type=str, default="results/bigartm_speed_quality")
    p.add_argument("--models", type=str, default="aartm,bigartm,bigartm_online")
    p.add_argument("--n_topics", type=int, default=100)
    p.add_argument("--max_iter", type=int, default=30)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--seeds", type=str, default="0")
    p.add_argument("--min_df", type=str, default="20")
    p.add_argument("--max_df", type=str, default="0.5")
    p.add_argument("--max_docs", type=int, default=-1)
    p.add_argument("--max_test_docs", type=int, default=-1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--compute_cv", action="store_true")
    # AARTM
    p.add_argument("--ctx_len", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--self_aware_context", action="store_true")
    p.add_argument("--num_attn_passes", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=100_000,
                   help="AARTM batch size in TOKENS")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--num_batches_before_update", type=int, default=-1,
                   help=">0 enables mini-batch EMA updates (analog of fit_online)")
    # BigARTM
    p.add_argument("--artm_batch_docs", type=int, default=1000,
                   help="BigARTM batch size in DOCUMENTS")
    p.add_argument("--artm_update_every", type=int, default=8)
    p.add_argument("--artm_num_processors", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    nltk.download("stopwords")

    d = prepare(args)
    d["cv_texts"] = (
        cv_texts_from_loader(d["train_loader"], d["id2word"]) if args.compute_cv else None
    )

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    all_rows = []
    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        for m in models:
            print(f"--- {m} ---")
            if m == "aartm":
                all_rows += run_aartm(d, args, seed)
            elif m == "bigartm":
                all_rows += run_bigartm(d, args, seed, online=False)
            elif m == "bigartm_online":
                all_rows += run_bigartm(d, args, seed, online=True)
            else:
                raise ValueError(f"Unknown model: {m}")

        pd.DataFrame(all_rows).to_csv(out_dir / "trajectory.csv", index=False)  # checkpoint

    traj = pd.DataFrame(all_rows)
    traj.to_csv(out_dir / "trajectory.csv", index=False)

    final = traj.sort_values("pass").groupby(["model", "seed"]).tail(1)
    metric_cols = [c for c in final.columns if c not in {"model", "seed", "pass"}]
    summary = final.groupby("model")[metric_cols].agg(["mean", "std"])
    summary.to_csv(out_dir / "summary.csv")

    print("\n=== Final summary ===")
    print(summary)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
