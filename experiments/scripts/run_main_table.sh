python3 -m experiments.run_main_table \
    --dataset dbpedia14 \
    --out_dir results/main_table/100topics \
    --n_topics 100 \
    --ctx_len 100 \
    --gamma 0.01 \
    --num_attn_passes 1 \
    --batch_size 10000 \
    --decorrelation_tau 0.0


python3 -m experiments.run_ablation \
    --dataset 20ng \
    --out_dir results/ablation/100topics \
    --n_topics 100 \
    --ctx_lens "100" \
    --gammas "0.01" \
    --num_attn_passes_values "1" \
    --decorrelation_taus "0.0, 0.1, 0.4, 0.7, 1.0, 1.3" \
    --batch_size 10000 \
    --self_aware_values "false" \
    --model_variants "full" \
    --seeds "0,1,2"


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

python3 -m experiments.run_stability \
    --dataset 20ng \
    --out_dir results/stablity/100topics \
    --n_topics 100 \
    --ctx_len 100 \
    --gamma 0.01
