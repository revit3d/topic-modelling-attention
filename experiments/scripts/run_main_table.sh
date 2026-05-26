#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3


python3 -m experiments.run_main_table \
    --dataset 20ng \
    --out_dir results/main_table \
    --n_topics 100 \
    --ctx_len 100 \
    --gamma 0.01 \
    --num_attn_passes 1 \
    --min_df 10 \
    --max_df 0.1 \
    --batch_size 10000 \
    --seeds 182,12392,8237


python3 -m experiments.run_main_table \
    --dataset ag_news \
    --out_dir results/main_table \
    --n_topics 100 \
    --ctx_len 100 \
    --gamma 0.01 \
    --num_attn_passes 1 \
    --min_df 25 \
    --max_df 0.1 \
    --batch_size 10000 \
    --seeds 182,12392,8237


python3 -m experiments.run_main_table \
    --dataset dbpedia14 \
    --out_dir results/main_table \
    --n_topics 100 \
    --ctx_len 100 \
    --gamma 0.01 \
    --num_attn_passes 1 \
    --min_df 100 \
    --max_df 0.1 \
    --batch_size 10000 \
    --seeds 182,12392,8237
