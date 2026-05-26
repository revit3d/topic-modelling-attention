#!/bin/bash

python3 -m experiments.run_length_robustness \
    --dataset 20ng \
    --out_dir results/length_robustness \
    --n_topics 100 \
    --ctx_len 100 \
    --gamma 0.01 \
    --trunc_lengths 1,2,4,6,8,10,16,32 \
    --min_df 10 \
    --max_df 0.1 \
    --seeds 182,12392,8237


python3 -m experiments.run_length_robustness \
    --dataset ag_news \
    --out_dir results/length_robustness \
    --n_topics 100 \
    --ctx_len 100 \
    --gamma 0.01 \
    --trunc_lengths 1,2,4,6,8,10,16,32 \
    --min_df 25 \
    --max_df 0.1 \
    --seeds 182,12392,8237


python3 -m experiments.run_length_robustness \
    --dataset dbpedia14 \
    --out_dir results/length_robustness \
    --n_topics 100 \
    --ctx_len 100 \
    --gamma 0.01 \
    --trunc_lengths 1,2,4,6,8,10,16,32 \
    --min_df 100 \
    --max_df 0.1 \
    --seeds 182,12392,8237
