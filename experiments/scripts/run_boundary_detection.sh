python3 -m experiments.boundary_detection \
    --dataset 20ng \
    --out_dir results/boundary_detection \
    --n_topics 100 \
    --ctx_len 100 \
    --gamma 0.01 \
    --per_side_tokens 32 \
    --boundary_window 16 \
    --n_pairs 500 \
    --min_df 10 \
    --max_df 0.1 \
    --seeds 182,12392,8237


python3 -m experiments.boundary_detection \
    --dataset ag_news \
    --out_dir results/boundary_detection \
    --n_topics 100 \
    --ctx_len 100 \
    --gamma 0.01 \
    --per_side_tokens 32 \
    --boundary_window 16 \
    --n_pairs 500 \
    --min_df 25 \
    --max_df 0.1 \
    --seeds 182,12392,8237


python3 -m experiments.boundary_detection \
    --dataset dbpedia14 \
    --out_dir results/boundary_detection \
    --n_topics 100 \
    --ctx_len 100 \
    --gamma 0.01 \
    --per_side_tokens 32 \
    --boundary_window 16 \
    --n_pairs 500 \
    --min_df 100 \
    --max_df 0.1 \
    --seeds 182,12392,8237
