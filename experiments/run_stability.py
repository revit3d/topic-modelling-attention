from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from experiments.common import prepare_data, aggregate_results
from experiments.paper_utils import parse_df_arg
from experiments.run_main_table_v2 import parse_csv_list, build_specs, parse_args as parse_main_args


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
    args = parse_main_args()
    out_dir = Path(args.out_dir) / args.dataset / "stability"
    out_dir.mkdir(parents=True, exist_ok=True)

    nltk.download("stopwords")

    data = prepare_data(
        args.dataset,
        min_df=parse_df_arg(args.min_df),
        max_df=parse_df_arg(args.max_df),
        min_token_len=3,
        max_token_len=20,
    )

    specs = build_specs()
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
