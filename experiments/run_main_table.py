from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any

import nltk
import numpy as np
import pandas as pd

from cartm import AttentiveTopicModel, ContextTopicModel
from experiments.model_no_N_wt import AttentiveTopicModelNoNWT

from experiments.common import (
    prepare_data,
    aggregate_results,
    fit_lda,
    fit_nmf,
    infer_doc_topics_aartm,
    infer_doc_topics_cartm,
    aartm_phi_pwt,
    cartm_phi_pwt,
    normalize_cols,
)
from experiments.paper_utils import parse_df_arg, build_regularizers, fit_topic_model
from experiments.topic_eval import (
    phi_to_topic_words,
    save_topic_words_list,
    evaluate_topic_words_and_doc_topics,
)
from experiments.external_baselines import (
    fit_bertopic,
    fit_combined_tm,
    bertopic_doc_topics,
    bertopic_topic_words,
    ctm_doc_topics,
    ctm_topic_words,
)


@dataclass
class ModelSpec:
    name: str
    fit_fn: Callable[[Any, int], tuple[Any, float, dict]]
    eval_fn: Callable[[Any, Any, dict, int], dict[str, float]]
    topic_words_fn: Callable[[Any, Any, dict, int], list[list[str]]]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="20ng", choices=["20ng", "ag_news", "dbpedia14"])
    parser.add_argument("--out_dir", type=str, default="results/main_table_v2")
    parser.add_argument("--models", type=str, default="aartm,aartm_no_nwt,cartm,lda,nmf,bertopic,ctm")
    parser.add_argument("--n_topics", type=int, default=50)
    parser.add_argument("--ctx_len", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.6)
    parser.add_argument("--self_aware_context", action="store_true")
    parser.add_argument("--num_attn_passes", type=int, default=2)
    parser.add_argument("--max_iter", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--decorrelation_tau", type=float, default=0.0)
    parser.add_argument("--min_df", type=str, default="5")
    parser.add_argument("--max_df", type=str, default="0.5")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2")
    return parser.parse_args()


def parse_csv_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def fit_local_model(model_cls, data, args, seed):
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
    return model, elapsed, {}


def evaluate_aartm_like(model, data, _, seed):
    phi_wt = aartm_phi_pwt(model, data.train_tokens)
    topic_words = phi_to_topic_words(phi_wt, data.id2word, top_k=25)

    X_train = infer_doc_topics_aartm(
        model,
        data.train_tokens,
        data.train_bounds,
        num_attn_passes=args.num_attn_passes,
    )
    X_test = infer_doc_topics_aartm(
        model,
        data.test_tokens,
        data.test_bounds,
        num_attn_passes=args.num_attn_passes,
    )

    return evaluate_topic_words_and_doc_topics(
        topic_words=topic_words,
        X_train=X_train,
        X_test=X_test,
        y_train=data.y_train,
        y_test=data.y_test,
        train_bow=data.train_bow,
        vocab=data.vocab,
        seed=seed,
    )


def evaluate_cartm_like(model, data, _, seed):
    phi_wt = cartm_phi_pwt(model)
    topic_words = phi_to_topic_words(phi_wt, data.id2word, top_k=25)

    X_train = infer_doc_topics_cartm(
        model,
        data.train_tokens,
        data.train_bounds,
        num_attn_passes=args.num_attn_passes,
    )
    X_test = infer_doc_topics_cartm(
        model,
        data.test_tokens,
        data.test_bounds,
        num_attn_passes=args.num_attn_passes,
    )

    return evaluate_topic_words_and_doc_topics(
        topic_words=topic_words,
        X_train=X_train,
        X_test=X_test,
        y_train=data.y_train,
        y_test=data.y_test,
        train_bow=data.train_bow,
        vocab=data.vocab,
        seed=seed,
    )


def aartm_topic_words(model, data, _, top_k):
    phi_wt = aartm_phi_pwt(model, data.train_tokens)
    return phi_to_topic_words(phi_wt, data.id2word, top_k=top_k)


def cartm_topic_words(model, data, _, top_k):
    phi_wt = cartm_phi_pwt(model)
    return phi_to_topic_words(phi_wt, data.id2word, top_k=top_k)


def fit_lda_spec(data, seed):
    model, elapsed = fit_lda(
        data,
        n_topics=args.n_topics,
        max_iter=args.max_iter,
        seed=seed,
    )
    return model, elapsed, {}


def fit_nmf_spec(data, seed):
    model, elapsed = fit_nmf(
        data,
        n_topics=args.n_topics,
        max_iter=args.max_iter,
        seed=seed,
    )
    return model, elapsed, {}


def evaluate_lda_spec(model, data, _, seed):
    phi_wt = normalize_cols(model.components_.T)
    topic_words = phi_to_topic_words(phi_wt, data.id2word, top_k=25)

    X_train = model.transform(data.train_bow)
    X_test = model.transform(data.test_bow)
    X_train = X_train / np.maximum(X_train.sum(axis=1, keepdims=True), 1e-12)
    X_test = X_test / np.maximum(X_test.sum(axis=1, keepdims=True), 1e-12)

    return evaluate_topic_words_and_doc_topics(
        topic_words=topic_words,
        X_train=X_train,
        X_test=X_test,
        y_train=data.y_train,
        y_test=data.y_test,
        train_bow=data.train_bow,
        vocab=data.vocab,
        seed=seed,
    )


def evaluate_nmf_spec(model, data, _, seed):
    phi_wt = normalize_cols(model.components_.T)
    topic_words = phi_to_topic_words(phi_wt, data.id2word, top_k=25)

    X_train = model.transform(data.train_tfidf)
    X_test = model.transform(data.test_tfidf)
    X_train = X_train / np.maximum(X_train.sum(axis=1, keepdims=True), 1e-12)
    X_test = X_test / np.maximum(X_test.sum(axis=1, keepdims=True), 1e-12)

    return evaluate_topic_words_and_doc_topics(
        topic_words=topic_words,
        X_train=X_train,
        X_test=X_test,
        y_train=data.y_train,
        y_test=data.y_test,
        train_bow=data.train_bow,
        vocab=data.vocab,
        seed=seed,
    )


def lda_topic_words(model, data, _, top_k):
    phi_wt = normalize_cols(model.components_.T)
    return phi_to_topic_words(phi_wt, data.id2word, top_k=top_k)


def nmf_topic_words(model, data, _, top_k):
    phi_wt = normalize_cols(model.components_.T)
    return phi_to_topic_words(phi_wt, data.id2word, top_k=top_k)


def fit_bertopic_spec(data, seed):
    return fit_bertopic(
        data,
        n_topics=args.n_topics,
        seed=seed,
        embedding_model_name=args.embedding_model,
    )


def evaluate_bertopic_spec(model, data, cache, seed):
    topic_words = bertopic_topic_words(model, top_k=25)
    X_train = bertopic_doc_topics(model, cache["train_docs"])
    X_test = bertopic_doc_topics(model, cache["test_docs"])

    return evaluate_topic_words_and_doc_topics(
        topic_words=topic_words,
        X_train=X_train,
        X_test=X_test,
        y_train=data.y_train,
        y_test=data.y_test,
        train_bow=data.train_bow,
        vocab=data.vocab,
        seed=seed,
    )


def bertopic_words_spec(model, data, cache, top_k):
    return bertopic_topic_words(model, top_k=top_k)


def fit_ctm_spec(data, seed):
    return fit_combined_tm(
        data,
        n_topics=args.n_topics,
        seed=seed,
        embedding_model_name=args.embedding_model,
        num_epochs=args.max_iter,
    )


def evaluate_ctm_spec(model, data, cache, seed):
    topic_words = ctm_topic_words(model, top_k=25)
    X_train = ctm_doc_topics(model, cache["train_dataset"])
    X_test = ctm_doc_topics(model, cache["test_dataset"])

    return evaluate_topic_words_and_doc_topics(
        topic_words=topic_words,
        X_train=X_train,
        X_test=X_test,
        y_train=data.y_train,
        y_test=data.y_test,
        train_bow=data.train_bow,
        vocab=data.vocab,
        seed=seed,
    )


def ctm_words_spec(model, data, cache, top_k):
    return ctm_topic_words(model, top_k=top_k)


def build_specs():
    return {
        "aartm": ModelSpec(
            name="AttentiveTopicModel",
            fit_fn=lambda data, seed: fit_local_model(AttentiveTopicModel, data, args, seed),
            eval_fn=evaluate_aartm_like,
            topic_words_fn=aartm_topic_words,
        ),
        "aartm_no_nwt": ModelSpec(
            name="AttentiveTopicModelNoNWT",
            fit_fn=lambda data, seed: fit_local_model(AttentiveTopicModelNoNWT, data, args, seed),
            eval_fn=evaluate_aartm_like,
            topic_words_fn=aartm_topic_words,
        ),
        "cartm": ModelSpec(
            name="ContextTopicModel",
            fit_fn=lambda data, seed: fit_local_model(ContextTopicModel, data, args, seed),
            eval_fn=evaluate_cartm_like,
            topic_words_fn=cartm_topic_words,
        ),
        "lda": ModelSpec(
            name="LDA",
            fit_fn=fit_lda_spec,
            eval_fn=evaluate_lda_spec,
            topic_words_fn=lda_topic_words,
        ),
        "nmf": ModelSpec(
            name="NMF",
            fit_fn=fit_nmf_spec,
            eval_fn=evaluate_nmf_spec,
            topic_words_fn=nmf_topic_words,
        ),
        "bertopic": ModelSpec(
            name="BERTopic",
            fit_fn=fit_bertopic_spec,
            eval_fn=evaluate_bertopic_spec,
            topic_words_fn=bertopic_words_spec,
        ),
        "ctm": ModelSpec(
            name="CombinedTM",
            fit_fn=fit_ctm_spec,
            eval_fn=evaluate_ctm_spec,
            topic_words_fn=ctm_words_spec,
        ),
    }


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

    selected = parse_csv_list(args.models)
    specs = build_specs()
    seeds = [int(x) for x in parse_csv_list(args.seeds)]

    rows = []

    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        for key in selected:
            spec = specs[key]
            print(f"Training {spec.name} ...")

            try:
                model, elapsed, cache = spec.fit_fn(data, seed)
            except ImportError as e:
                print(f"Skipping {spec.name}: missing dependency: {e}")
                continue

            metrics = spec.eval_fn(model, data, cache, seed)
            metrics.update({
                "dataset": args.dataset,
                "model": spec.name,
                "seed": seed,
                "train_time_sec": elapsed,
            })
            rows.append(metrics)

            topic_words = spec.topic_words_fn(model, data, cache, 15)
            save_topic_words_list(
                topic_words,
                out_dir / f"top_words_{spec.name}_seed{seed}.txt",
            )

    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(out_dir / "main_table_raw.csv", index=False)

    summary_df = aggregate_results(raw_df, group_cols=["dataset", "model"])
    summary_df.to_csv(out_dir / "main_table_summary.csv", index=False)

    print(raw_df.head())
    print(summary_df.head())
    print(f"\nSaved to {out_dir}")
