python3 -m experiments.run_main_table \
    --dataset 20ng \
    --out_dir results/main_table/100topics \
    --n_topics 100 \
    --ctx_len 100 \
    --gamma 0.01 \
    --num_attn_passes 1 \
    --min_df 10 \
    --max_df 0.1 \
    --batch_size 10000

python3 -m experiments.run_main_table \
    --dataset ag_news \
    --out_dir results/main_table/100topics \
    --n_topics 100 \
    --ctx_len 100 \
    --gamma 0.01 \
    --num_attn_passes 1 \
    --min_df 25 \
    --max_df 0.1 \
    --batch_size 10000

python3 -m experiments.run_main_table \
    --dataset dbpedia14 \
    --out_dir results/main_table/100topics \
    --n_topics 100 \
    --ctx_len 100 \
    --gamma 0.01 \
    --num_attn_passes 1 \
    --min_df 100 \
    --max_df 0.1 \
    --batch_size 10000 \
    --seeds 1


python3 -m experiments.run_length_robustness \
    --dataset dbpedia14 \
    --out_dir results/length_robustness/100topics \
    --n_topics 100 \
    --ctx_len 10 \
    --gamma 0.1 \
    --trunc_lengths "8,16,32,64,128" \
    --seeds "1"


python3 -m experiments.boundary_detection \
    --dataset ag_news \
    --out_dir results/boundary_detection/100topics \
    --n_topics 100 \
    --ctx_len 10 \
    --gamma 0.1 \
    --per_side_tokens 32 \
    --boundary_window 16 \
    --n_pairs 500 \
    --seeds "0,1,2"
