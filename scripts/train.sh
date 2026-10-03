#!/bin/bash
# train.sh — launch one DiffuSeq training run (run from scripts/; run_train.py switches to the repo root).
#
# GPU Sharing Rule (.agents/AGENTS.md): the GPU is shared with other repositories. Check
# `nvidia-smi` and get the user's approval before launching; long runs go through systemd-run
# (Long-Running Job Rule).
#
# Batch accounting: --bsz is the optimizer batch; --microbatch is how many samples are on the GPU
# at once. Gradients of ceil(bsz / microbatch) microbatches are accumulated, each weighted by its
# share of the batch, so the update equals the mean over the full batch for any split.
#
# Dataset: built by prepare_docamr_datasets.py (datasets/docAMR/<DATASET>/{train,valid,test}.jsonl).
# --seq_len must equal the --max_seq_len used when the dataset was built (meta.json records it).
# Graph runs (text_amr_* / amr_* datasets): add --graph_encoder gatv2 --graph_mode edge_attr|levi.
#
# Resume: add --resume_checkpoint <run folder>/ema_0.9999_<step>.pt (the step counter and LR schedule
# restart from 0, see TrainLoop._load_and_sync_parameters).

DATASET=${DATASET:-v2_plain_en_vi_chunk_1}

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12231 run_train.py \
--diff_steps 2000 \
--lr 0.0001 \
--learning_steps 50000 \
--save_interval 5000 \
--seed 102 \
--noise_schedule sqrt \
--hidden_dim 768 \
--hidden_t_dim 768 \
--bsz 384 \
--microbatch 32 \
--dataset docAMR/${DATASET} \
--data_dir ./datasets/docAMR/${DATASET} \
--learned_mean_embed True \
--denoise True \
--vocab bert \
--seq_len 128 \
--use_fp16 \
--config_name bert-base-multilingual-cased \
--denoise_rate 0.5 \
--schedule_sampler lossaware \
--notes learned_mask_fp16_denoise_0.5 \
--amr_vocab relations \
--mask_docamr_rel False \
--graph_encoder none
