"""
sample_seq2seq_dpmSolver.py — DPM-Solver accelerated inference for DiffuSeq.

This script generates text from a trained DiffuSeq model using the DPM-Solver++
ODE solver, which dramatically reduces the number of function evaluations (NFE)
needed compared to the standard DDPM sampler (e.g., 10-20 steps instead of 2000).

Pipeline:
  1. Load model checkpoint + hyperparameters.
  2. Load tokenizer and tie the model's word embeddings to the embedding matrix.
  3. Wrap the model with DPM-Solver's model_wrapper (converts to continuous-time formulation).
  4. For each batch in the test/validation set:
       a. Embed source tokens → x_source in embedding space.
       b. Add Gaussian noise to target positions only (source positions left clean).
       c. Run DPM-Solver++ to denoise x_noised → x_sample over SOLVER_STEP steps.
       d. Project x_sample → vocab logits via lm_head → greedy decode (argmax).
       e. Separate source and recovered target tokens using the input_mask.
  5. Write {"recover": ..., "reference": ..., "source": ...} JSON lines to an output file.

Key design:
  - "Partial noising": only target (mask=1) positions are noised; source (mask=0)
    positions stay as their clean embeddings. This conditions generation on the source.
  - DPM-Solver++: works with the model's x_0 prediction type ("x_start").
  - Output is saved per-batch to a .json file inside a structured output directory.
"""

import argparse
import os, json
from tracemalloc import start

import numpy as np
import torch as th
import torch.distributed as dist
from transformers import set_seed
from diffuseq.rounding import denoised_fn_round, get_weights  # nearest-neighbour rounding utilities
from diffuseq.text_datasets import load_data_text
from torch.cuda.amp import autocast

import time
from diffuseq.utils import dist_util, logger
from functools import partial
from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    load_model_emb,
    load_tokenizer,
    AMR_TO_TEXT_LABEL,
    TEXT_TO_AMR_LABEL,
)


def create_argparser():
    """
    Build the argument parser for the inference script.

    Adds inference-specific defaults on top of the training config:
        model_path:      path to the EMA checkpoint .pt file.
        step (SOLVER_STEP): number of DPM-Solver ODE steps (10–20 is recommended).
        out_dir:         directory where output .json files are written.
        top_p:           nucleus sampling threshold (0 = greedy argmax).
        rejection_rate:  unused at inference (kept for config compatibility).
        note:            suffix appended to the output filename for identification.
        split:           'valid' or 'test' (which data split to decode).
        clamp_step:      number of initial steps to apply clamping (per DPM-Solver).
        seed2:           secondary random seed for sampling reproducibility.
        clip_denoised:   whether to clip predicted x_0 to [-1, 1] during denoising.
        start_n:         skip the first N batches (allows resuming partial decoding runs).

    Returns:
        argparse.ArgumentParser
    """
    defaults = dict(model_path='', step=0, out_dir='', top_p=0, rejection_rate=0.0, note='none')
    decode_defaults = dict(split='valid', clamp_step=0, seed2=105, clip_denoised=False, start_n=0,
                           filter_direction='AMR_TO_TEXT')  # 'AMR_TO_TEXT' or 'TEXT_TO_AMR'
    defaults.update(load_defaults_config())   # pull in all training defaults
    defaults.update(decode_defaults)          # override/add inference-specific defaults
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def main():
    args = create_argparser().parse_args()

    # Initialize distributed environment (needed even for single-GPU inference
    # because DistributedSampler and dist.all_gather are used for output collection)
    dist_util.setup_dist()
    logger.configure()

    # -------------------------------------------------------------------------
    # 1. Load Training Configuration
    # -------------------------------------------------------------------------
    # The model was saved alongside training_args.json which records all hyperparameters.
    # We reload these to reconstruct the exact same model architecture at inference.
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    print(config_path)
    with open(config_path, 'rb') as f:
        training_args = json.load(f)
    # Allow overriding batch size from the CLI (useful to fit inference on less GPU memory)
    training_args['batch_size'] = args.batch_size
    args.__dict__.update(training_args)
    
    # Step: Disable DocAMR relation masking logic during evaluation.
    # This ensures full-target denoising and evaluation consistency.
    args.mask_docamr_rel = False

    # -------------------------------------------------------------------------
    # 2. Reconstruct Model + Diffusion
    # -------------------------------------------------------------------------
    logger.log("### Creating model and diffusion...")
    args.device = dist_util.dev()
    print('#' * 10, args.clamp_step)
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config().keys())
    )

    # Load checkpoint weights. The EMA checkpoint is loaded for inference
    # since EMA weights typically generalize better than the raw model weights.
    # map_location="cpu" avoids GPU OOM during loading; model is moved to GPU next.
    # dist_util.load_state_dict(path, **kwargs) → reads the file and calls th.load(**kwargs)
    # map_location="cpu": load weights to CPU first to avoid GPU OOM; model is moved to GPU later
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'### The parameter count is {pytorch_total_params}')

    model.to(dist_util.dev())
    model.eval()   # disable dropout for deterministic inference

    # -------------------------------------------------------------------------
    # 3. Load Tokenizer and Embedding Matrix
    # -------------------------------------------------------------------------
    tokenizer = load_tokenizer(args)
    model_emb, tokenizer = load_model_emb(args, tokenizer)

    # Tie the standalone embedding matrix to the model's learned word embeddings.
    # During training, model.word_embedding.weight was updated jointly with the model,
    # so the standalone model_emb must reflect those learned weights at inference.
    model_emb.weight = th.nn.Parameter(model.word_embedding.weight.clone().cpu())
    # get_weights: ensures model_emb is a plain nn.Embedding with frozen weights
    model_emb_copy = get_weights(model_emb, args)

    # Set secondary seed for sampling reproducibility (separate from training seed)
    set_seed(args.seed2)

    print("### Sampling...on", args.split)

    # -------------------------------------------------------------------------
    # 4. Load Inference Data (single pass, not looped)
    # -------------------------------------------------------------------------
    # loop=False: returns a single-pass iterator that raises StopIteration at end
    data_valid = load_data_text(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        deterministic=True,  # no shuffling → reproducible output order
        data_args=args,
        split=args.split,
        loaded_vocab=tokenizer,
        model_emb=model_emb.cpu(),  # use CPU embedding; samples moved to GPU below
        loop=False,
        filter_direction=args.filter_direction,  # CLI-configurable: 'AMR_TO_TEXT' or 'TEXT_TO_AMR'
    )

    start_t = time.time()

    SOLVER_STEP = args.step   # number of DPM-Solver ODE integration steps

    # -------------------------------------------------------------------------
    # 5. Build Output Path
    # -------------------------------------------------------------------------
    # Output structure:
    #   out_dir/
    #     {model_folder}.{checkpoint_file}/            ← model identifier
    #       ema{...}.samples/                          ← EMA rate + step suffix
    #         seed{seed2}_solverstep{N}_{note}.json    ← one JSON-lines file per run
    model_base_name = (
        os.path.basename(os.path.split(args.model_path)[0])
        + f'.{os.path.split(args.model_path)[1]}'
    )
    out_dir = os.path.join(args.out_dir, f"{model_base_name.split('.ema')[0]}")
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    out_path = os.path.join(out_dir, f"ema{model_base_name.split('.ema')[1]}.samples")
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    out_path = os.path.join(
        out_path, f"seed{args.seed2}_solverstep{SOLVER_STEP}_{args.filter_direction}_{args.note}.json"
    )

    # -------------------------------------------------------------------------
    # 6. Collect All Test Batches
    # -------------------------------------------------------------------------
    # Pre-load the entire dataset into memory to allow start_n skipping.
    # Each element of all_test_data is a cond dict: {'input_ids': ..., 'input_mask': ...}
    all_test_data = []
    try:
        while True:
            batch, cond = next(data_valid)
            all_test_data.append(cond)
    except StopIteration:
        print('### End of reading iteration...')

    from tqdm import tqdm
    print('Start from ...', args.start_n)
    all_test_data = all_test_data[args.start_n:]   # skip first start_n batches (resume support)

    # -------------------------------------------------------------------------
    # 7. Set Up DPM-Solver
    # -------------------------------------------------------------------------
    from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver

    # NoiseScheduleVP: wraps the discrete β schedule into a continuous VPvariance schedule.
    # DPM-Solver works in continuous time; this bridging object converts the discrete betas.
    noise_schedule = NoiseScheduleVP(
        schedule='discrete',
        betas=th.from_numpy(diffusion.betas)
    )

    # model_wrapper: converts the DiffuSeq model (which outputs x̂_0) to the format
    # expected by DPM-Solver. model_type="x_start" means the model predicts the clean x_0,
    # not noise epsilon or the score.
    model_kwargs = {}
    model_fn = model_wrapper(
        model,
        noise_schedule,
        model_type="x_start",     # DiffuSeq predicts clean x_0 (not noise ε)
        model_kwargs=model_kwargs,
        guidance_type="uncond",   # unconditional generation (no classifier guidance)
    )

    # DPM-Solver++: higher-order ODE solver for diffusion models.
    # "dpmsolver++" is faster and more stable than vanilla "dpmsolver".
    dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")

    # -------------------------------------------------------------------------
    # 8. Decoding Loop
    # -------------------------------------------------------------------------
    for cond in tqdm(all_test_data):

        # Extract and remove conditioning tensors from the dict (pop mutates in-place)
        input_ids_x = cond.pop('input_ids').to(dist_util.dev())   # [B, seq_len] token IDs
        x_start = model.get_embeds(input_ids_x)                   # [B, seq_len, hidden_dim] clean embeddings

        input_ids_mask = cond.pop('input_mask')      # [B, seq_len] mask: 0=source, 1=target
        input_ids_mask_ori = input_ids_mask           # keep original for decoding step

        model_kwargs['edge_index'] = cond.pop('edge_index').to(dist_util.dev()) if 'edge_index' in cond else None
        model_kwargs['edge_type'] = cond.pop('edge_type').to(dist_util.dev()) if 'edge_type' in cond else None

        # Sample Gaussian noise for target positions
        noise = th.randn_like(x_start)

        # Broadcast scalar mask [B, seq_len] → [B, seq_len, hidden_dim] for torch.where
        input_ids_mask = th.broadcast_to(
            input_ids_mask.unsqueeze(dim=-1), x_start.shape
        ).to(dist_util.dev())

        # Partial noising: source positions (mask=0) keep clean embeddings x_start;
        # target positions (mask=1) are replaced with pure Gaussian noise.
        # The model must reverse this noise to recover the target text.
        x_noised = th.where(input_ids_mask == 0, x_start, noise)

        # Run DPM-Solver++ ODE to denoise x_noised → x_sample in SOLVER_STEP iterations.
        # order=2: second-order solver (good balance of speed and quality).
        # skip_type="time_uniform": evenly spaces the ODE steps in time.
        # method="multistep": uses previous function evaluations to improve accuracy.
        with autocast():
            x_sample = dpm_solver.sample(
                x_noised,
                steps=SOLVER_STEP,
                order=2,
                skip_type="time_uniform",
                method="multistep",
                input_ids_mask=input_ids_mask,   # passed through to model to re-anchor source positions
                x_start=x_start,                 # passed through for source anchoring in the model
            )

        # -----------------------------------------------------------------------
        # 9. Gather Outputs Across GPUs
        # -----------------------------------------------------------------------
        sample = x_sample
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        # all_gather: collect denoised samples from all GPUs into a list
        dist.all_gather(gathered_samples, sample)
        all_sentence = [sample.cpu().numpy() for sample in gathered_samples]

        # -----------------------------------------------------------------------
        # 10. Decode: Embedding → Token IDs → Text
        # -----------------------------------------------------------------------
        word_lst_recover = []   # model's generated target text
        word_lst_ref = []       # ground-truth target text
        word_lst_source = []    # input source text

        # Concatenate gathered samples from all GPUs along the batch dimension
        arr = np.concatenate(all_sentence, axis=0)     # [total_B, seq_len, hidden_dim]
        x_t = th.tensor(arr).cuda()

        reshaped_x_t = x_t
        # Project denoised embeddings to vocabulary: logits [B, seq_len, vocab_size]
        logits = model.get_logits(reshaped_x_t)

        # Greedy decoding: take the highest-probability token at each position
        cands = th.topk(logits, k=1, dim=-1)
        sample = cands.indices   # [B, seq_len, 1] predicted token IDs

        # Recover generated target: skip source positions (first len_x positions)
        for seq, input_mask in zip(cands.indices, input_ids_mask_ori):
            # len_x = number of source tokens; target starts at position len_x
            len_x = args.seq_len - sum(input_mask).tolist()
            tokens = tokenizer.decode_token(seq[len_x:])
            word_lst_recover.append(tokens)

        # Decode source and reference target from the original input
        for seq, input_mask in zip(input_ids_x, input_ids_mask_ori):
            len_x = args.seq_len - sum(input_mask).tolist()
            word_lst_source.append(tokenizer.decode_token(seq[:len_x]))    # source portion
            word_lst_ref.append(tokenizer.decode_token(seq[len_x:]))       # reference target

        # -----------------------------------------------------------------------
        # 11. Write Results to Output File
        # -----------------------------------------------------------------------
        # Append to the output file in JSON-lines format (one dict per line)
        fout = open(out_path, 'a')
        for (recov, ref, src) in zip(word_lst_recover, word_lst_ref, word_lst_source):
            print(json.dumps({"recover": recov, "reference": ref, "source": src}, ensure_ascii=False), file=fout)
        fout.close()

    print('### Total takes {:.2f}s .....'.format(time.time() - start_t))
    print(f'### Written the decoded output to {out_path}')


if __name__ == "__main__":
    main()
