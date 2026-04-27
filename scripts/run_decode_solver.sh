CUDA_VISIBLE_DEVICES=0 python -u run_decode_solver.py \
--model_dir diffusion_models/diffuseq_docAMR/src_docamr_en_trg_doc_vi_chunk_5_simple_h256_lr0.0001_t2000_sqrt_lossaware_seed102_learned_mask_fp16_denoise_0.5_reproduce20260426-21:37:30 \
--seed 110 \
--bsz 10 \
--step 10 \
--split test \
--use_simple_amr True
