CUDA_VISIBLE_DEVICES=0 python -u run_decode_solver.py \
--model_dir diffusion_models/diffuseq_docAMR/src_doc_vi_trg_docamr_en_chunk_5_h256_lr0.0001_t2000_sqrt_lossaware_seed102_learned_mask_fp16_denoise_0.5_reproduce20260421-08:08:11 \
--seed 110 \
--bsz 24 \
--step 10 \
--split test
