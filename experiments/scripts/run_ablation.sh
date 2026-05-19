########################################################
##################### Context length ###################
########################################################
python3 -m experiments.run_ablation \
    --dataset 20ng \
    --out_dir results/ablation/context_length \
    --n_topics 100 \
    --ctx_lens 1000 \
    --gammas 0.9,0.5,0.2,0.1,0.05,0.01,0.005,0.001 \
    --num_attn_passes_values 1 \
    --decorrelation_taus 0 \
    --batch_size 10000 \
    --self_aware_values false \
    --model_variants full \
    --min_df 10 \
    --max_df 0.1 \
    --seeds 182,12392


python3 -m experiments.run_ablation \
    --dataset ag_news \
    --out_dir results/ablation/context_length \
    --n_topics 100 \
    --ctx_lens 1000 \
    --gammas 0.9,0.5,0.2,0.1,0.05,0.01,0.005,0.001 \
    --num_attn_passes_values 1 \
    --decorrelation_taus 0 \
    --batch_size 10000 \
    --self_aware_values false \
    --model_variants full \
    --min_df 25 \
    --max_df 0.1 \
    --seeds 182,12392


python3 -m experiments.run_ablation \
    --dataset dbpedia14 \
    --out_dir results/ablation/context_length \
    --n_topics 100 \
    --ctx_lens 1000 \
    --gammas 0.9,0.5,0.2,0.1,0.05,0.01,0.005,0.001 \
    --num_attn_passes_values 1 \
    --decorrelation_taus 0 \
    --batch_size 10000 \
    --self_aware_values false \
    --model_variants full \
    --min_df 100 \
    --max_df 0.1 \
    --seeds 182,12392


########################################################
################### Self-aware context #################
########################################################
python3 -m experiments.run_ablation \
    --dataset 20ng \
    --out_dir results/ablation/self_aware \
    --n_topics 100 \
    --ctx_lens 100 \
    --gammas 0.01 \
    --num_attn_passes_values 1 \
    --decorrelation_taus 0 \
    --batch_size 10000 \
    --self_aware_values true,false \
    --model_variants full \
    --min_df 10 \
    --max_df 0.1 \
    --seeds 182,12392,8237


python3 -m experiments.run_ablation \
    --dataset ag_news \
    --out_dir results/ablation/self_aware \
    --n_topics 100 \
    --ctx_lens 100 \
    --gammas 0.01 \
    --num_attn_passes_values 1 \
    --decorrelation_taus 0 \
    --batch_size 10000 \
    --self_aware_values true,false \
    --model_variants full \
    --min_df 25 \
    --max_df 0.1 \
    --seeds 182,12392,8237


python3 -m experiments.run_ablation \
    --dataset dbpedia14 \
    --out_dir results/ablation/self_aware \
    --n_topics 100 \
    --ctx_lens 100 \
    --gammas 0.01 \
    --num_attn_passes_values 1 \
    --decorrelation_taus 0 \
    --batch_size 10000 \
    --self_aware_values true,false \
    --model_variants full \
    --min_df 100 \
    --max_df 0.1 \
    --seeds 182,12392,8237


########################################################
#################### Attention passes ##################
########################################################
python3 -m experiments.run_ablation \
    --dataset 20ng \
    --out_dir results/ablation/attn_passes \
    --n_topics 100 \
    --ctx_lens 1,2,3,4,5 \
    --gammas 0.1 \
    --num_attn_passes_values 1,2,3,4,5 \
    --decorrelation_taus 0 \
    --batch_size 10000 \
    --self_aware_values false \
    --model_variants full \
    --min_df 10 \
    --max_df 0.1 \
    --seeds 182,12392,8237,38293,18237,12,904,7326,9948,104


python3 -m experiments.run_ablation \
    --dataset ag_news \
    --out_dir results/ablation/attn_passes \
    --n_topics 100 \
    --ctx_lens 1,2,3,4,5 \
    --gammas 0.1 \
    --num_attn_passes_values 1,2,3,4,5 \
    --decorrelation_taus 0 \
    --batch_size 10000 \
    --self_aware_values false \
    --model_variants full \
    --min_df 25 \
    --max_df 0.1 \
    --seeds 182,12392,8237


python3 -m experiments.run_ablation \
    --dataset dbpedia14 \
    --out_dir results/ablation/attn_passes \
    --n_topics 100 \
    --ctx_lens 1,2,3,4,5 \
    --gammas 0.1 \
    --num_attn_passes_values 1,2,3,4,5 \
    --decorrelation_taus 0 \
    --batch_size 10000 \
    --self_aware_values false \
    --model_variants full \
    --min_df 100 \
    --max_df 0.1 \
    --seeds 182,12392,8237


########################################################
###################### no-N_tw model ###################
########################################################
python3 -m experiments.run_ablation \
    --dataset 20ng \
    --out_dir results/ablation/no_ntw \
    --n_topics 100 \
    --ctx_lens 100 \
    --gammas 0.01 \
    --num_attn_passes_values 1 \
    --decorrelation_taus 0 \
    --batch_size 10000 \
    --self_aware_values false \
    --model_variants no_nwt \
    --min_df 10 \
    --max_df 0.1 \
    --seeds 182,12392,8237


python3 -m experiments.run_ablation \
    --dataset ag_news \
    --out_dir results/ablation/no_ntw \
    --n_topics 100 \
    --ctx_lens 100 \
    --gammas 0.01 \
    --num_attn_passes_values 1 \
    --decorrelation_taus 0 \
    --batch_size 10000 \
    --self_aware_values false \
    --model_variants no_nwt \
    --min_df 25 \
    --max_df 0.1 \
    --seeds 182,12392,8237


python3 -m experiments.run_ablation \
    --dataset dbpedia14 \
    --out_dir results/ablation/no_ntw \
    --n_topics 100 \
    --ctx_lens 100 \
    --gammas 0.01 \
    --num_attn_passes_values 1 \
    --decorrelation_taus 0 \
    --batch_size 10000 \
    --self_aware_values false \
    --model_variants no_nwt \
    --min_df 100 \
    --max_df 0.1 \
    --seeds 182,12392,8237
