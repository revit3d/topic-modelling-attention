from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import pandas as pd
import nltk

from cartm import AttentiveTopicModel
from experiments.model_no_N_wt import AttentiveTopicModelNoNWT
from experiments.common import (
    prepare_data,
    aggregate_results,
    evaluate_aartm,
    parse_df_arg,
    build_regularizers,
    fit_topic_model,
)


def parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def parse_bool_list(s: str) -> list[bool]:
    mapping = {"true": True, "false": False, "1": True, "0": False}
    return [mapping[x.strip().lower()] for x in s.split(",") if x.strip()]


def parse_str_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="20ng", choices=["20ng", "ag_news", "dbpedia14"])
    parser.add_argument("--out_dir", type=str, default="results/ablation")
    parser.add_argument("--n_topics", type=int, default=50)
    parser.add_argument("--ctx_lens", type=str, default="1,2,4,8,16")
    parser.add_argument("--gammas", type=str, default="0.2,0.4,0.6,0.8")
    parser.add_argument("--self_aware_values", type=str, default="false,true")
    parser.add_argument("--num_attn_passes_values", type=str, default="1,2,4")
    parser.add_argument("--decorrelation_taus", type=str, default="0.0,0.1")
    parser.add_argument("--model_variants", type=str, default="full,no_nwt")
    parser.add_argument("--max_iter", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=10000)
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

    seeds = parse_int_list(args.seeds)
    ctx_lens = parse_int_list(args.ctx_lens)
    gammas = parse_float_list(args.gammas)
    self_aware_values = parse_bool_list(args.self_aware_values)
    num_attn_passes_values = parse_int_list(args.num_attn_passes_values)
    decorrelation_taus = parse_float_list(args.decorrelation_taus)
    model_variants = parse_str_list(args.model_variants)

    rows = []

    variant_to_cls = {
        "full": AttentiveTopicModel,
        "no_nwt": AttentiveTopicModelNoNWT,
    }

    for variant, seed, ctx_len, gamma, self_aware, num_attn_passes, tau in product(
        model_variants,
        seeds,
        ctx_lens,
        gammas,
        self_aware_values,
        num_attn_passes_values,
        decorrelation_taus,
    ):
        model_cls = variant_to_cls[variant]
        print(
            f"variant={variant} | seed={seed} | ctx_len={ctx_len} | gamma={gamma} | "
            f"self_aware={self_aware} | passes={num_attn_passes} | tau={tau}"
        )

        regs = build_regularizers(tau)
        model = model_cls(
            vocab_size=len(data.vocab),
            ctx_len=ctx_len,
            n_topics=args.n_topics,
            gamma=gamma,
            self_aware_context=self_aware,
            regularizers=regs,
        )

        train_time = fit_topic_model(
            model,
            data.train_tokens,
            data.train_bounds,
            num_attn_passes=num_attn_passes,
            max_iter=args.max_iter,
            tol=args.tol,
            seed=seed,
            batch_size=args.batch_size,
        )

        metrics = evaluate_aartm(
            model,
            data,
            num_attn_passes=num_attn_passes,
            batch_size=args.batch_size,
            seed=seed,
        )
        metrics.update({
            "dataset": args.dataset,
            "model_variant": variant,
            "seed": seed,
            "ctx_len": ctx_len,
            "gamma": gamma,
            "self_aware_context": self_aware,
            "num_attn_passes": num_attn_passes,
            "decorrelation_tau": tau,
            "train_time_sec": train_time,
        })
        rows.append(metrics)

    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(out_dir / "ablation_raw.csv", index=False)

    summary_df = aggregate_results(
        raw_df,
        group_cols=[
            "dataset",
            "model_variant",
            "ctx_len",
            "gamma",
            "self_aware_context",
            "num_attn_passes",
            "decorrelation_tau",
        ],
    )
    summary_df.to_csv(out_dir / "ablation_summary.csv", index=False)

    print(raw_df.head())
    print(summary_df.head())
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
