from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfTransformer

from cartm import AttentiveTopicModel, ContextTopicModel
from experiments.model_no_N_wt import AttentiveTopicModelNoNWT
from experiments.common import (
    prepare_data,
    infer_doc_topics_aartm,
    infer_doc_topics_cartm,
    classification_scores,
    fit_lda,
    fit_nmf,
    parse_df_arg,
    build_regularizers,
    fit_topic_model,
    truncate_corpus,
)
from cartm.preprocessing import build_bow


def normalize_rows(x: np.ndarray) -> np.ndarray:
    denom = x.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return x / denom


def parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="20ng", choices=["20ng", "ag_news", "dbpedia14"])
    parser.add_argument("--out_dir", type=str, default="results/length_robustness")
    parser.add_argument("--n_topics", type=int, default=50)
    parser.add_argument("--ctx_len", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.6)
    parser.add_argument("--self_aware_context", action="store_true")
    parser.add_argument("--num_attn_passes", type=int, default=2)
    parser.add_argument("--max_iter", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--decorrelation_tau", type=float, default=0.0)
    parser.add_argument("--trunc_lengths", type=str, default="8,16,32,64,128")
    parser.add_argument("--min_df", type=str, default="5")
    parser.add_argument("--max_df", type=str, default="0.5")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    return parser.parse_args()


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

    trunc_lengths = parse_int_list(args.trunc_lengths)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    rows = []

    for seed in seeds:
        print(f"\n=== Seed {seed} ===")

        regs = build_regularizers(args.decorrelation_tau)

        aartm = AttentiveTopicModel(
            vocab_size=len(data.vocab),
            ctx_len=args.ctx_len,
            n_topics=args.n_topics,
            gamma=args.gamma,
            self_aware_context=args.self_aware_context,
            regularizers=regs,
        )
        fit_topic_model(
            aartm,
            data.train_tokens,
            data.train_bounds,
            num_attn_passes=args.num_attn_passes,
            max_iter=args.max_iter,
            tol=args.tol,
            seed=seed,
            batch_size=args.batch_size,
        )

        aartm_no_nwt = AttentiveTopicModelNoNWT(
            vocab_size=len(data.vocab),
            ctx_len=args.ctx_len,
            n_topics=args.n_topics,
            gamma=args.gamma,
            self_aware_context=args.self_aware_context,
            regularizers=regs,
        )
        fit_topic_model(
            aartm_no_nwt,
            data.train_tokens,
            data.train_bounds,
            num_attn_passes=args.num_attn_passes,
            max_iter=args.max_iter,
            tol=args.tol,
            seed=seed,
            batch_size=args.batch_size,
        )

        cartm = ContextTopicModel(
            vocab_size=len(data.vocab),
            ctx_len=args.ctx_len,
            n_topics=args.n_topics,
            gamma=args.gamma,
            self_aware_context=args.self_aware_context,
            regularizers=regs,
        )
        fit_topic_model(
            cartm,
            data.train_tokens,
            data.train_bounds,
            num_attn_passes=args.num_attn_passes,
            max_iter=args.max_iter,
            tol=args.tol,
            seed=seed,
            batch_size=args.batch_size,
        )

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

        for max_len in trunc_lengths:
            print(f"  truncation: {max_len} tokens/doc")

            train_tokens_t, train_bounds_t = truncate_corpus(
                data.train_tokens,
                data.train_bounds,
                max_tokens_per_doc=max_len,
            )
            test_tokens_t, test_bounds_t = truncate_corpus(
                data.test_tokens,
                data.test_bounds,
                max_tokens_per_doc=max_len,
            )

            X_train = infer_doc_topics_aartm(
                aartm,
                train_tokens_t,
                train_bounds_t,
                num_attn_passes=args.num_attn_passes,
            )
            X_test = infer_doc_topics_aartm(
                aartm,
                test_tokens_t,
                test_bounds_t,
                num_attn_passes=args.num_attn_passes,
            )
            metrics = classification_scores(X_train, data.y_train, X_test, data.y_test, seed=seed)
            metrics.update(
                {
                    "dataset": args.dataset,
                    "model": "AttentiveTopicModel",
                    "seed": seed,
                    "max_tokens_per_doc": max_len,
                }
            )
            rows.append(metrics)

            X_train = infer_doc_topics_aartm(
                aartm_no_nwt,
                train_tokens_t,
                train_bounds_t,
                num_attn_passes=args.num_attn_passes,
            )
            X_test = infer_doc_topics_aartm(
                aartm_no_nwt,
                test_tokens_t,
                test_bounds_t,
                num_attn_passes=args.num_attn_passes,
            )
            metrics = classification_scores(X_train, data.y_train, X_test, data.y_test, seed=seed)
            metrics.update(
                {
                    "dataset": args.dataset,
                    "model": "AttentiveTopicModelNoNWT",
                    "seed": seed,
                    "max_tokens_per_doc": max_len,
                }
            )
            rows.append(metrics)

            X_train = infer_doc_topics_cartm(
                cartm,
                train_tokens_t,
                train_bounds_t,
                num_attn_passes=args.num_attn_passes,
            )
            X_test = infer_doc_topics_cartm(
                cartm,
                test_tokens_t,
                test_bounds_t,
                num_attn_passes=args.num_attn_passes,
            )
            metrics = classification_scores(X_train, data.y_train, X_test, data.y_test, seed=seed)
            metrics.update(
                {
                    "dataset": args.dataset,
                    "model": "ContextTopicModel",
                    "seed": seed,
                    "max_tokens_per_doc": max_len,
                }
            )
            rows.append(metrics)

            train_bow_t = build_bow(train_tokens_t, train_bounds_t, len(data.vocab))
            test_bow_t = build_bow(test_tokens_t, test_bounds_t, len(data.vocab))

            X_train = normalize_rows(lda.transform(train_bow_t))
            X_test = normalize_rows(lda.transform(test_bow_t))
            metrics = classification_scores(X_train, data.y_train, X_test, data.y_test, seed=seed)
            metrics.update(
                {
                    "dataset": args.dataset,
                    "model": "LDA",
                    "seed": seed,
                    "max_tokens_per_doc": max_len,
                }
            )
            rows.append(metrics)

            tfidf = TfidfTransformer(norm="l2")
            train_tfidf_t = tfidf.fit_transform(train_bow_t)
            test_tfidf_t = tfidf.transform(test_bow_t)

            X_train = normalize_rows(nmf.transform(train_tfidf_t))
            X_test = normalize_rows(nmf.transform(test_tfidf_t))
            metrics = classification_scores(X_train, data.y_train, X_test, data.y_test, seed=seed)
            metrics.update(
                {
                    "dataset": args.dataset,
                    "model": "NMF",
                    "seed": seed,
                    "max_tokens_per_doc": max_len,
                }
            )
            rows.append(metrics)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "length_robustness.csv", index=False)

    summary = df.groupby(["dataset", "model", "max_tokens_per_doc"]).agg(["mean", "std"])
    summary.to_csv(out_dir / "length_robustness_summary.csv")

    print(df.head())
    print(f"\nSaved to:\n  {out_dir / 'length_robustness.csv'}\n  {out_dir / 'length_robustness_summary.csv'}")


if __name__ == "__main__":
    main()
