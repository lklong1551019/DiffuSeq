#!/bin/bash
# train.sh — Launch DiffuSeq training.
#
# FIX: Changed from the deprecated `python -m torch.distributed.launch` to `torchrun`.
# FIX: Changed --nproc_per_node=2 → 1 because this machine has only 1 GPU (RTX 3060 Ti).
#      --nproc_per_node must equal the number of GPUs available under CUDA_VISIBLE_DEVICES.
#      Setting it higher than the actual GPU count causes:
#          RuntimeError: CUDA error: invalid device ordinal
#      because the extra rank tries to access a non-existent cuda:N device.
#
# To use multiple GPUs in the future (e.g., 2 GPUs):
#     CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 ...


# OOM FIX: --microbatch controls how many samples are processed on GPU at one time.
# With batch_size=425 and --microbatch=425, NO gradient accumulation happened — the full
# 425-sample batch hit the GPU at once, causing OOM on the RTX 3060 Ti (8GB).
#
# How --microbatching works in train_util.py:
#   for i in range(0, batch_size, microbatch):   ← loop over chunks
#       micro = batch[i:i+microbatch].to(GPU)    ← only microbatch samples on GPU
#       loss.backward()                           ← accumulate gradients
#   optimizer.step()                              ← one update per full batch
#

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12231 run_train.py \
--diff_steps 2000 \
--lr 0.0001 \
--learning_steps 50000 \
--save_interval 5000 \
--seed 102 \
--noise_schedule sqrt \
--hidden_dim 128 \
--bsz 425 \
--microbatch 16 \
--dataset qqp \
--data_dir ./datasets/qqp \
--learned_mean_embed True \
--denoise True \
--vocab bert \
--seq_len 128 \
--use_fp16 \
--denoise_rate 0.5 \
--schedule_sampler lossaware \
--notes learned_mask_fp16_denoise_0.5_reproduce
