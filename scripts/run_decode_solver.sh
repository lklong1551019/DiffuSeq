# Decode all EMA checkpoints of one run with DPM-Solver++ (GPU job: check nvidia-smi and get approval first).
# MODEL_DIR: a run folder under diffusion_models/. Score the outputs with scripts/eval_bleu.py.
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR=diffusion_models/<run folder>}

CUDA_VISIBLE_DEVICES=0 python -u run_decode_solver.py \
--model_dir ${MODEL_DIR} \
--seed 110 \
--bsz 50 \
--step 10 \
--split test
