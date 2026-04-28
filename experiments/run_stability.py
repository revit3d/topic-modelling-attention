from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from experiments.common import prepare_data, aggregate_results, parse_df_arg
from experiments.run_main_table import parse_csv_list, build_specs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="20ng", choices=["20ng", "ag_news", "dbpedia14"])
    parser.add_argument("--out_dir", type=str, default="results/stability")
    parser.add_argument("--models", type=str, default="aartm,aartm_no_nwt,lda,nmf,bertopic")
    parser.add_argument("--n_topics", type=int, default=100)
    parser.add_argument("--ctx_len", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--self_aware_context", action="store_true")
    parser.add_argument("--num_attn_passes", type=int, default=1)
    parser.add_argument("--max_iter", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--decorrelation_tau", type=float, default=0.0)
    parser.add_argument("--min_df", type=str, default="5")
    parser.add_argument("--max_df", type=str, default="0.5")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2")
    return parser.parse_args()


def matched_topic_jaccard(
    topic_words_a: list[list[str]],
    topic_words_b: list[list[str]],
    top_k: int = 10,
) -> float:
    A = [set(words[:top_k]) for words in topic_words_a]
    B = [set(words[:top_k]) for words in topic_words_b]

    if len(A) == 0 or len(B) == 0:
        return float("nan")

    sim = np.zeros((len(A), len(B)), dtype=np.float32)
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            union = len(a | b)
            sim[i, j] = 0.0 if union == 0 else len(a & b) / union

    row_ind, col_ind = linear_sum_assignment(1.0 - sim)
    return float(sim[row_ind, col_ind].mean())


if __name__ == "__main__":
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

    specs = build_specs(args)
    selected = parse_csv_list(args.models)
    seeds = [int(x) for x in parse_csv_list(args.seeds)]

    rows = []

    for key in selected:
        spec = specs[key]
        print(f"\n=== {spec.name} ===")

        per_seed_topics = {}
        for seed in seeds:
            try:
                model, elapsed, cache = spec.fit_fn(data, seed)
            except ImportError as e:
                print(f"Skipping {spec.name}: missing dependency: {e}")
                per_seed_topics = {}
                break
            per_seed_topics[seed] = spec.topic_words_fn(model, data, cache, 15)

        for seed_a, seed_b in combinations(seeds, 2):
            if seed_a not in per_seed_topics or seed_b not in per_seed_topics:
                continue
            score = matched_topic_jaccard(
                per_seed_topics[seed_a],
                per_seed_topics[seed_b],
                top_k=10,
            )
            rows.append({
                "dataset": args.dataset,
                "model": spec.name,
                "seed_a": seed_a,
                "seed_b": seed_b,
                "matched_jaccard_top10": score,
            })

    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(out_dir / "stability_raw.csv", index=False)

    summary_df = aggregate_results(raw_df, group_cols=["dataset", "model"])
    summary_df.to_csv(out_dir / "stability_summary.csv", index=False)

    print(raw_df.head())
    print(summary_df.head())
    print(f"\nSaved to {out_dir}")
