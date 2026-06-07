CUDA_VISIBLE_DEVICES=0 python -u run_decode_solver.py \
--model_dir diffusion_models/diffuseq_docAMR/plain_text_vi_en_chunk_1_h256_lr0.0001_t2000_sqrt_lossaware_seed102_learned_mask_fp16_denoise_0.5_reproduce20260518-08:17:28 \
--seed 110 \
--bsz 10 \
--step 10 \
--split test \
--use_simple_amr False \
--enable_gcn False \
--use_relational_gcn False \
#--filter_direction TEXT_TO_AMR
