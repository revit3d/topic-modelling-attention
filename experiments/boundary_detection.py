from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import nltk
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer

from cartm import AttentiveTopicModel, ContextTopicModel
from experiments.model_no_N_wt import AttentiveTopicModelNoNWT
from experiments.common import (
    prepare_data,
    fit_lda,
    fit_nmf,
    doc_spans,
    parse_df_arg,
    build_regularizers,
    fit_topic_model,
    infer_token_topics_aartm,
    infer_token_topics_cartm,
    make_synthetic_boundary_dataset,
    evaluate_boundary_detection,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="20ng", choices=["20ng", "ag_news", "dbpedia14"])
    parser.add_argument("--out_dir", type=str, default="results/boundary_detection")
    parser.add_argument("--n_topics", type=int, default=50)
    parser.add_argument("--ctx_len", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.6)
    parser.add_argument("--self_aware_context", action="store_true")
    parser.add_argument("--num_attn_passes", type=int, default=2)
    parser.add_argument("--max_iter", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--decorrelation_tau", type=float, default=0.0)
    parser.add_argument("--per_side_tokens", type=int, default=64)
    parser.add_argument("--n_pairs", type=int, default=500)
    parser.add_argument("--boundary_window", type=int, default=16)
    parser.add_argument("--min_df", type=str, default="5")
    parser.add_argument("--max_df", type=str, default="0.5")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    return parser.parse_args()


def normalize_rows(x: np.ndarray) -> np.ndarray:
    denom = x.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return x / denom


def fit_context_model(model_cls, data, args, seed):
    regs = build_regularizers(args.decorrelation_tau)
    model = model_cls(
        vocab_size=len(data.vocab),
        ctx_len=args.ctx_len,
        n_topics=args.n_topics,
        gamma=args.gamma,
        self_aware_context=args.self_aware_context,
        regularizers=regs,
    )
    elapsed = fit_topic_model(
        model,
        data.train_tokens,
        data.train_bounds,
        num_attn_passes=args.num_attn_passes,
        max_iter=args.max_iter,
        tol=args.tol,
        seed=seed,
        batch_size=args.batch_size,
    )
    return model, elapsed


def flat_docs_from_bounds(tokens, bounds):
    tokens_np = np.asarray(tokens, dtype=np.int32)
    spans = doc_spans(bounds, len(tokens_np))
    return [tokens_np[s:e] for s, e in spans]


def make_window_bow(doc_tokens: np.ndarray, vocab_size: int, half_window: int) -> sp.csr_matrix:
    rows, cols, vals = [], [], []
    n = len(doc_tokens)
    for i in range(n):
        lo = max(0, i - half_window)
        hi = min(n, i + half_window + 1)
        cnt = np.bincount(doc_tokens[lo:hi], minlength=vocab_size)
        nz = np.flatnonzero(cnt)
        rows.extend([i] * len(nz))
        cols.extend(nz.tolist())
        vals.extend(cnt[nz].astype(np.float32).tolist())
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, vocab_size), dtype=np.float32)


def infer_token_topics_sklearn_windows(model, docs, vocab_size, half_window, tfidf=None):
    all_theta = []
    for doc in docs:
        X = make_window_bow(doc, vocab_size=vocab_size, half_window=half_window)
        if tfidf is not None:
            X = tfidf.transform(X)
        theta = model.transform(X)
        theta = normalize_rows(np.asarray(theta, dtype=np.float32))
        all_theta.append(theta)
    return np.vstack(all_theta)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    nltk.download("stopwords")

    data = prepare_data(
        args.dataset,
        min_df=parse_df_arg(args.min_df),
        max_df=parse_df_arg(args.max_df),
        min_token_len=3,
        max_token_len=20,
    )

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    rows = []

    for seed in seeds:
        print(f"\n=== Seed {seed} ===")

        aartm, _ = fit_context_model(AttentiveTopicModel, data, args, seed)
        aartm_no_nwt, _ = fit_context_model(AttentiveTopicModelNoNWT, data, args, seed)
        cartm, _ = fit_context_model(ContextTopicModel, data, args, seed)

        lda, _ = fit_lda(
            data,
            n_topics=args.n_topics,
            max_iter=args.max_iter,
            seed=seed,
        )
        nmf, _ = fit_nmf(
            data,
            n_topics=args.n_topics,
            max_iter=args.max_iter,
            seed=seed,
        )

        mixed_docs, mix_tokens, mix_bounds, true_boundaries = make_synthetic_boundary_dataset(
            data.test_texts_filtered,
            data.y_test,
            data.loader,
            per_side_tokens=args.per_side_tokens,
            n_pairs=args.n_pairs,
            seed=seed,
        )

        token_topics = infer_token_topics_aartm(
            aartm,
            mix_tokens,
            mix_bounds,
            num_attn_passes=args.num_attn_passes,
        )
        metrics = evaluate_boundary_detection(
            token_topics,
            mix_bounds,
            true_boundaries,
            window=args.boundary_window,
        )
        metrics.update({"dataset": args.dataset, "model": "AttentiveTopicModel", "seed": seed})
        rows.append(metrics)

        token_topics = infer_token_topics_aartm(
            aartm_no_nwt,
            mix_tokens,
            mix_bounds,
            num_attn_passes=args.num_attn_passes,
        )
        metrics = evaluate_boundary_detection(
            token_topics,
            mix_bounds,
            true_boundaries,
            window=args.boundary_window,
        )
        metrics.update({"dataset": args.dataset, "model": "AttentiveTopicModelNoNWT", "seed": seed})
        rows.append(metrics)

        token_topics = infer_token_topics_cartm(
            cartm,
            mix_tokens,
            mix_bounds,
            num_attn_passes=args.num_attn_passes,
        )
        metrics = evaluate_boundary_detection(
            token_topics,
            mix_bounds,
            true_boundaries,
            window=args.boundary_window,
        )
        metrics.update({"dataset": args.dataset, "model": "ContextTopicModel", "seed": seed})
        rows.append(metrics)

        docs = flat_docs_from_bounds(mix_tokens, mix_bounds)
        token_topics = infer_token_topics_sklearn_windows(
            lda,
            docs,
            vocab_size=len(data.vocab),
            half_window=args.boundary_window,
            tfidf=None,
        )
        metrics = evaluate_boundary_detection(
            token_topics,
            mix_bounds,
            true_boundaries,
            window=args.boundary_window,
        )
        metrics.update({"dataset": args.dataset, "model": "LDA", "seed": seed})
        rows.append(metrics)

        tfidf = TfidfTransformer(norm="l2")
        tfidf.fit(data.train_bow)
        token_topics = infer_token_topics_sklearn_windows(
            nmf,
            docs,
            vocab_size=len(data.vocab),
            half_window=args.boundary_window,
            tfidf=tfidf,
        )
        metrics = evaluate_boundary_detection(
            token_topics,
            mix_bounds,
            true_boundaries,
            window=args.boundary_window,
        )
        metrics.update({"dataset": args.dataset, "model": "NMF", "seed": seed})
        rows.append(metrics)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "boundary_detection_raw.csv", index=False)

    summary = df.groupby(["dataset", "model"]).agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    summary.to_csv(out_dir / "boundary_detection_summary.csv", index=False)

    print(df.head())
    print(summary.head())
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
