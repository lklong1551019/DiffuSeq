"""
train.py — Main entry point for training the DiffuSeq diffusion model.

Pipeline:
  1. Parse command-line arguments (hyperparameters, paths, etc.)
  2. Set random seed for reproducibility
  3. Initialize distributed training (multi-GPU via torch.distributed)
  4. Load the tokenizer and the data-loader embedding table
  5. Build training and validation data loaders
  6. Construct the Transformer model + diffusion noise schedule
  7. Initialize Weights & Biases (W&B) logging on the master process only
  8. Hand off to TrainLoop to run the actual training
"""

import argparse
import json, torch, os
import numpy as np

# dist_util: handles distributed training setup (NCCL backend, device selection)
# logger:    lightweight logging utility (writes to stdout / log files)
from diffuseq.utils import dist_util, logger

# load_data_text: builds a PyTorch DataLoader for tokenized text sequences
from diffuseq.text_datasets import load_data_text

# create_named_schedule_sampler: builds a timestep sampler for the diffusion process
# (e.g., uniform sampling or loss-aware importance sampling)
from diffuseq.step_sample import create_named_schedule_sampler

from basic_utils import (
    load_defaults_config,       # loads default hyperparameters from a config file
    create_model_and_diffusion, # instantiates the Transformer model + GaussianDiffusion
    args_to_dict,               # converts argparse.Namespace → dict (for model construction)
    add_dict_to_argparser,      # registers config keys as CLI flags
    load_model_emb,             # embedding table used by the data loader (not by the loss)
    load_tokenizer              # loads the tokenizer (vocab, special tokens, etc.)
)

# TrainLoop: encapsulates the full training loop, EMA updates, checkpointing, and eval
from train_util import TrainLoop

# set_seed: sets Python / NumPy / PyTorch random seeds for full reproducibility
from transformers import set_seed
import wandb

# W&B runs offline; logs land in ./wandb (sync later with `wandb sync`).
os.environ["WANDB_MODE"] = "offline"


def create_argparser():
    """
    Build the argument parser by merging default config values with CLI flags.

    Steps:
      1. Load default hyperparameters from diffuseq/config.json.
      2. Register each default key as a CLI argument so users can override via command line.

    Returns:
        argparse.ArgumentParser: ready-to-parse argument parser.
    """
    defaults = dict()
    defaults.update(load_defaults_config())  # pull in defaults (lr, seq_len, batch_size, etc.)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)  # register defaults as --flag arguments
    return parser


def main():
    # -------------------------------------------------------------------------
    # 1. Parse Arguments & Set Seed
    # -------------------------------------------------------------------------
    args = create_argparser().parse_args()

    # Fix all random seeds (Python, NumPy, PyTorch) for reproducible training
    set_seed(args.seed)

    # -------------------------------------------------------------------------
    # 2. Distributed Training Setup
    # -------------------------------------------------------------------------
    # Initializes torch.distributed (NCCL backend for GPU communication).
    # Each GPU process gets a unique rank. dist_util.dev() returns the correct
    # torch.device for this process (e.g., cuda:0, cuda:1, ...).
    dist_util.setup_dist()

    # Configure the logger (sets up log file path, formatting, etc.)
    logger.configure()
    logger.log("### Creating data loader...")

    # -------------------------------------------------------------------------
    # 3. Load Tokenizer and Word Embeddings
    # -------------------------------------------------------------------------
    # Base tokenizer (args.config_name) + AMR tokens of args.amr_vocab + direction tokens; saved to
    # args.checkpoint_path so decoding reloads the exact vocabulary.
    tokenizer = load_tokenizer(args)

    # Embedding table for TextDataset (N(0,1), or mBERT rows when use_plm_init=bert), saved as
    # random_emb.torch. It does not initialize the model: the loss embeds input_ids with the
    # model's own word_embedding.
    model_weight, tokenizer = load_model_emb(args, tokenizer)

    # -------------------------------------------------------------------------
    # 4. Build Training Data Loader
    # -------------------------------------------------------------------------
    # load_data_text returns an infinite generator of (batch, extra_info) tuples.
    # Each batch contains tokenized + padded source-target sequence pairs.
    data = load_data_text(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        data_args=args,
        loaded_vocab=tokenizer,
        model_emb=model_weight
    )

    # Advance the generator once to trigger dataset loading/caching eagerly,
    # so the first training step doesn't pay a cold-start penalty.
    next(data)

    # -------------------------------------------------------------------------
    # 5. Build Validation Data Loader
    # -------------------------------------------------------------------------
    # The valid split goes through a DistributedSampler (shuffled with its fixed default seed);
    # `deterministic` only affects the test split. TrainLoop evaluates one batch per eval_interval.
    data_valid = load_data_text(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        data_args=args,
        split='valid',
        deterministic=True,
        loaded_vocab=tokenizer,
        model_emb=model_weight
    )

    print('#' * 30, 'size of vocab', args.vocab_size)

    # -------------------------------------------------------------------------
    # 6. Build Model and Diffusion Process
    # -------------------------------------------------------------------------
    logger.log("### Creating model and diffusion...")

    # Store the target device on args so downstream utilities can access it
    args.device = dist_util.dev()

    # create_model_and_diffusion:
    #   - model:     a Transformer that predicts x_0 (the original embedding)
    #                from a noisy embedding x_t and timestep t.
    #   - diffusion: a GaussianDiffusion object that defines the forward noise
    #                schedule (beta values) and the reverse sampling procedure.
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config().keys())
    )

    # Move the model to the appropriate GPU for this process
    model.to(dist_util.dev())

    # Log total parameter count for model size awareness
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'### The parameter count is {pytorch_total_params}')

    # create_named_schedule_sampler: determines which diffusion timesteps to
    # sample during training (e.g., uniform or loss-second-moment reweighted).
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    # -------------------------------------------------------------------------
    # 7. Save Hyperparameters to Disk
    # -------------------------------------------------------------------------
    logger.log(f'### Saving the hyperparameters to {args.checkpoint_path}/training_args.json')

    # Clear device from args before serializing (torch.device is not JSON-serializable)
    args.device = ""
    with open(f'{args.checkpoint_path}/training_args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # -------------------------------------------------------------------------
    # 8. Initialize W&B Logging (Master Process Only)
    # -------------------------------------------------------------------------
    # LOCAL_RANK is set by torchrun/torch.distributed.launch — it identifies
    # which GPU this process controls within the current node.
    #
    # W&B is initialized on LOCAL_RANK 0 only (one run per job, not one per GPU); without
    # LOCAL_RANK the process is single-GPU and always logs.
    if ('LOCAL_RANK' not in os.environ) or (int(os.environ['LOCAL_RANK']) == 0):
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "DiffuSeq"),  # W&B project name
            name=args.checkpoint_path,                        # run name = checkpoint folder
        )
        # Sync all hyperparameters to W&B for experiment tracking
        wandb.config.update(args.__dict__, allow_val_change=True)

    # -------------------------------------------------------------------------
    # 9. Run Training Loop
    # -------------------------------------------------------------------------
    logger.log("### Training...")

    # TrainLoop handles:
    #   - Forward pass: add noise to x_0, predict x_0 with model, compute loss
    #   - Backward pass: gradient accumulation via microbatches, gradient clipping
    #   - EMA: maintains exponential moving average of model weights for stable eval
    #   - Checkpointing: saves the EMA weights (ema_<rate>_<step>.pt) every save_interval steps
    #   - Evaluation: runs on data_valid every eval_interval steps
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,           # process in smaller chunks to save GPU memory
        lr=args.lr,
        ema_rate=args.ema_rate,               # smoothing factor for EMA weights (e.g., 0.9999)
        log_interval=args.log_interval,       # log loss every N steps
        save_interval=args.save_interval,     # save checkpoint every N steps
        resume_checkpoint=args.resume_checkpoint,  # .pt path to resume from; 'none' = train fresh
        use_fp16=args.use_fp16,               # mixed-precision training for speed/memory
        fp16_scale_growth=args.fp16_scale_growth,  # legacy fp16 path only; GradScaler ignores it
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        learning_steps=args.learning_steps,   # total number of gradient update steps
        checkpoint_path=args.checkpoint_path,
        gradient_clipping=args.gradient_clipping,  # max grad norm to prevent exploding gradients
        eval_data=data_valid,
        eval_interval=args.eval_interval      # run validation every N steps
    ).run_loop()


if __name__ == "__main__":
    main()
