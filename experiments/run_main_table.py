from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any
from functools import partial

import nltk
import numpy as np
import pandas as pd

from cartm import AttentiveTopicModel
from experiments.model_no_N_wt import AttentiveTopicModelNoNWT

from experiments.common import (
    prepare_data,
    aggregate_results,
    fit_lda,
    fit_nmf,
    evaluate_aartm,
    aartm_phi_pwt,
    normalize_cols,
    parse_df_arg,
    build_regularizers,
    fit_topic_model,
)
from experiments.topic_eval import (
    phi_to_topic_words,
    save_topic_words_list,
    evaluate_topic_words_and_doc_topics,
    topic_eval_texts_from_data,
    c_v_coherence_from_topic_words,
    bertscore_from_topic_words,
)
from experiments.external_baselines import (
    fit_bertopic, bertopic_topic_words, bertopic_doc_topics,
    fit_combined_tm, ctm_topic_words, ctm_doc_topics,
    fit_btm, btm_topic_words, btm_doc_topics,
    fit_bigartm, bigartm_topic_words, bigartm_doc_topics,
    fit_contextual_top2vec, top2vec_topic_words, top2vec_doc_topics,
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
    parser.add_argument("--out_dir", type=str, default="results/main_table")
    parser.add_argument(
        "--models",
        type=str,
        default="aartm,aartm_no_nwt,lda,nmf,bertopic,ctm,btm,bigartm,top2vec",
    )
    parser.add_argument("--n_topics", type=int, default=100)
    parser.add_argument("--ctx_len", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--self_aware_context", action="store_true")
    parser.add_argument("--num_attn_passes", type=int, default=1)
    parser.add_argument("--max_iter", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--decorrelation_tau", type=float, default=0.0)
    parser.add_argument("--min_df", type=str, default="10")
    parser.add_argument("--max_df", type=str, default="0.1")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2")

    parser.add_argument("--bertscore_lang", type=str, default="en")
    parser.add_argument("--bertscore_model", type=str, default=None)
    parser.add_argument("--bertscore_batch_size", type=int, default=64)
    parser.add_argument("--bertscore_rescale_with_baseline", action="store_true")
    parser.add_argument("--bertscore_device", type=str, default=None)
    return parser.parse_args()


def parse_csv_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _topic_eval_kwargs(args, data) -> dict[str, Any]:
    return {
        "cv_texts": topic_eval_texts_from_data(data),
        "bertscore_lang": args.bertscore_lang,
        "bertscore_model_type": args.bertscore_model,
        "bertscore_batch_size": args.bertscore_batch_size,
        "bertscore_rescale_with_baseline": args.bertscore_rescale_with_baseline,
        "bertscore_device": args.bertscore_device,
    }


def _topic_word_metric_values(
    topic_words: list[list[str]],
    data,
    args,
) -> dict[str, float]:
    cv_texts = topic_eval_texts_from_data(data)
    return {
        "c_v_10": c_v_coherence_from_topic_words(
            topic_words,
            cv_texts,
            top_k=10,
        ),
        "bertscore_f1_10": bertscore_from_topic_words(
            topic_words,
            top_k=10,
            lang=args.bertscore_lang,
            model_type=args.bertscore_model,
            batch_size=args.bertscore_batch_size,
            rescale_with_baseline=args.bertscore_rescale_with_baseline,
            device=args.bertscore_device,
        ),
    }


def fit_local_model(model_cls, data, args, seed):
    regs = build_regularizers(args.decorrelation_tau, "tw")
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


def aartm_topic_words(model, data, _, top_k):
    phi_wt = aartm_phi_pwt(model, data.train_tokens)
    return phi_to_topic_words(phi_wt, data.id2word, top_k=top_k)


def fit_lda_spec(data, seed, n_topics, max_iter):
    model, elapsed = fit_lda(
        data,
        n_topics=n_topics,
        max_iter=max_iter,
        seed=seed,
    )
    return model, elapsed, {}


def fit_nmf_spec(data, seed, n_topics, max_iter):
    model, elapsed = fit_nmf(
        data,
        n_topics=n_topics,
        max_iter=max_iter,
        seed=seed,
    )
    return model, elapsed, {}


def evaluate_aartm_spec(model, data, cache, seed, *, args):
    metrics = evaluate_aartm(
        model,
        data,
        batch_size=args.batch_size,
        num_attn_passes=args.num_attn_passes,
        seed=seed,
    )
    topic_words = aartm_topic_words(
        model,
        data,
        cache,
        top_k=25,
    )
    metrics.update(_topic_word_metric_values(topic_words, data, args))
    return metrics


def evaluate_lda_spec(model, data, cache, seed, *, args):
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
        **_topic_eval_kwargs(args, data),
    )


def evaluate_nmf_spec(model, data, cache, seed, *, args):
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
        **_topic_eval_kwargs(args, data),
    )


def lda_topic_words(model, data, _, top_k):
    phi_wt = normalize_cols(model.components_.T)
    return phi_to_topic_words(phi_wt, data.id2word, top_k=top_k)


def nmf_topic_words(model, data, _, top_k):
    phi_wt = normalize_cols(model.components_.T)
    return phi_to_topic_words(phi_wt, data.id2word, top_k=top_k)


def fit_bertopic_spec(data, seed, n_topics, embedding_model):
    return fit_bertopic(
        data,
        n_topics=n_topics,
        seed=seed,
        embedding_model_name=embedding_model,
    )


def evaluate_bertopic_spec(model, data, cache, seed, *, args):
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
        **_topic_eval_kwargs(args, data),
    )


def bertopic_words_spec(model, data, cache, top_k):
    return bertopic_topic_words(model, top_k=top_k)


def fit_ctm_spec(data, seed, *, args):
    return fit_combined_tm(
        data,
        n_topics=args.n_topics,
        seed=seed,
        embedding_model_name=args.embedding_model,
    )


def evaluate_ctm_spec(model, data, cache, seed, *, args):
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
        **_topic_eval_kwargs(args, data),
    )


def ctm_words_spec(model, data, cache, top_k):
    return ctm_topic_words(model, top_k=top_k)


def fit_btm_spec(data, seed, *, args):
    return fit_btm(data, n_topics=args.n_topics, seed=seed,
                   num_iterations=args.max_iter)


def evaluate_btm_spec(model, data, cache, seed, *, args):
    topic_words = btm_topic_words(model, top_k=25)
    X_train = btm_doc_topics(model, cache["docs_vec_train"])
    X_test  = btm_doc_topics(model, cache["docs_vec_test"])
    return evaluate_topic_words_and_doc_topics(
        topic_words=topic_words,
        X_train=X_train, X_test=X_test,
        y_train=data.y_train, y_test=data.y_test,
        train_bow=data.train_bow, vocab=data.vocab,
        seed=seed,
        **_topic_eval_kwargs(args, data),
    )


def btm_words_spec(model, data, cache, top_k):
    return btm_topic_words(model, top_k=top_k)


def fit_bigartm_spec(data, seed, *, args):
    return fit_bigartm(data, n_topics=args.n_topics, seed=seed,
                       max_iter=args.max_iter,
                       decorrelation_tau=args.decorrelation_tau)


def evaluate_bigartm_spec(model, data, cache, seed, *, args):
    topic_words = bigartm_topic_words(model, top_k=25)
    X_train = bigartm_doc_topics(model, data, split="train")
    X_test  = bigartm_doc_topics(model, data, split="test")
    return evaluate_topic_words_and_doc_topics(
        topic_words=topic_words,
        X_train=X_train, X_test=X_test,
        y_train=data.y_train, y_test=data.y_test,
        train_bow=data.train_bow, vocab=data.vocab,
        seed=seed,
        **_topic_eval_kwargs(args, data),
    )


def bigartm_words_spec(model, data, cache, top_k):
    return bigartm_topic_words(model, top_k=top_k)


def fit_top2vec_spec(data, seed, *, args):
    return fit_contextual_top2vec(
        data, n_topics=args.n_topics, seed=seed,
        embedding_model_name=args.embedding_model,
    )


def evaluate_top2vec_spec(model, data, cache, seed, *, args):
    topic_words = top2vec_topic_words(model, top_k=25)
    X_train = top2vec_doc_topics(model, data, split="train")
    X_test  = top2vec_doc_topics(model, data, split="test")
    return evaluate_topic_words_and_doc_topics(
        topic_words=topic_words,
        X_train=X_train, X_test=X_test,
        y_train=data.y_train, y_test=data.y_test,
        train_bow=data.train_bow, vocab=data.vocab,
        seed=seed,
        **_topic_eval_kwargs(args, data),
    )


def top2vec_words_spec(model, data, cache, top_k):
    return top2vec_topic_words(model, top_k=top_k)


def build_specs(args):
    return {
        "aartm": ModelSpec(
            name="AttentiveTopicModel",
            fit_fn=lambda data, seed: fit_local_model(AttentiveTopicModel, data, args, seed),
            eval_fn=partial(evaluate_aartm_spec, args=args),
            topic_words_fn=aartm_topic_words,
        ),
        "aartm_no_nwt": ModelSpec(
            name="AttentiveTopicModelNoNWT",
            fit_fn=lambda data, seed: fit_local_model(AttentiveTopicModelNoNWT, data, args, seed),
            eval_fn=partial(evaluate_aartm_spec, args=args),
            topic_words_fn=aartm_topic_words,
        ),
        "lda": ModelSpec(
            name="LDA",
            fit_fn=partial(fit_lda_spec, n_topics=args.n_topics, max_iter=args.max_iter),
            eval_fn=partial(evaluate_lda_spec, args=args),
            topic_words_fn=lda_topic_words,
        ),
        "nmf": ModelSpec(
            name="NMF",
            fit_fn=partial(fit_nmf_spec, n_topics=args.n_topics, max_iter=args.max_iter),
            eval_fn=partial(evaluate_nmf_spec, args=args),
            topic_words_fn=nmf_topic_words,
        ),
        "bertopic": ModelSpec(
            name="BERTopic",
            fit_fn=partial(fit_bertopic_spec, n_topics=args.n_topics, embedding_model=args.embedding_model),
            eval_fn=partial(evaluate_bertopic_spec, args=args),
            topic_words_fn=bertopic_words_spec,
        ),
        "ctm": ModelSpec(
            name="CombinedTM",
            fit_fn=partial(fit_ctm_spec, args=args),
            eval_fn=partial(evaluate_ctm_spec, args=args),
            topic_words_fn=ctm_words_spec,
        ),
        "btm": ModelSpec(
            name="BTM",
            fit_fn=partial(fit_btm_spec, args=args),
            eval_fn=partial(evaluate_btm_spec, args=args),
            topic_words_fn=btm_words_spec,
        ),
        "bigartm": ModelSpec(
            name="BigARTM",
            fit_fn=partial(fit_bigartm_spec, args=args),
            eval_fn=partial(evaluate_bigartm_spec, args=args),
            topic_words_fn=bigartm_words_spec,
        ),
        "top2vec": ModelSpec(
            name="ContextualTop2Vec",
            fit_fn=partial(fit_top2vec_spec, args=args),
            eval_fn=partial(evaluate_top2vec_spec, args=args),
            topic_words_fn=top2vec_words_spec,
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
    specs = build_specs(args)
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

            metrics = spec.eval_fn(model, data, cache=cache, seed=seed)
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
