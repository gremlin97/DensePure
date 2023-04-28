#!/usr/bin/env bash
cd ..

sigma=$1
steps=1
reverse_seed=$2

python eval_certified_densepure.py \
--exp exp/imagenet \
--config imagenet.yml \
-i imagenet-densepure-sample_num_10000-noise_$sigma-$steps-$reverse_seed \
--domain imagenet \
--seed 0 \
--diffusion_type cm \
--lp_norm L2 \
--outfile imagenet-densepure-sample_num_10000-noise_$sigma-$steps-$reverse_seed \
--sigma $sigma \
--N 1600 \
--N0 128 \
--certified_batch 16 \
--sample_id $(seq -s ' ' 0 500 49500) \
--use_id \
--certify_mode purify \
--advanced_classifier beit \
--use_one_step \
--reverse_seed $reverse_seed 
--save_predictions \
--predictions_path exp/imagenet/$sigma- 