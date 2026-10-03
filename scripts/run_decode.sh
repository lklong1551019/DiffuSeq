# Decode all EMA checkpoints of one run with the DDPM/DDIM sampler (GPU job: check nvidia-smi first).
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR=diffusion_models/<run folder>}

python -u run_decode.py \
--model_dir ${MODEL_DIR} \
--seed 123 \
--split test
