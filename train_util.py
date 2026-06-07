"""
train_util.py — Core training loop for DiffuSeq.

Contains the TrainLoop class which orchestrates:
  - Gradient accumulation via microbatches
  - Mixed-precision (FP16) training with GradScaler
  - Exponential Moving Average (EMA) of model weights
  - Linear learning rate annealing
  - Periodic checkpointing (EMA weights only)
  - Periodic validation loss evaluation
  - Distributed training synchronization (DDP)

Also contains helper functions for checkpoint management and loss logging.
"""

import copy
import functools
import os

import blobfile as bf           # cloud-agnostic file I/O (works local or GCS/S3)
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import io
import time

from diffuseq.utils import dist_util, logger
from diffuseq.utils.fp16_util import (
    make_master_params,              # convert model FP16 params → FP32 master copies
    master_params_to_model_params,   # copy FP32 masters back → FP16 model params
    model_grads_to_master_grads,     # accumulate FP16 gradients into FP32 master grads
    unflatten_master_params,         # re-associate flat master params with model param names
    zero_grad,                       # zero gradients on a list of parameters
)
from diffuseq.utils.nn import update_ema      # exponential moving average update step
from diffuseq.step_sample import LossAwareSampler, UniformSampler

# PyTorch native AMP (Automatic Mixed Precision)
from torch.cuda.amp import GradScaler   # dynamic loss scaling to avoid FP16 underflow
from torch.cuda.amp import autocast     # context manager: runs ops in FP16 where safe

# Initial log₂ loss scale for FP16 training.
# The GradScaler will adjust this dynamically. Starting at 2^20 ≈ 1M is a good
# heuristic — it typically climbs to ~2^20–2^21 within the first ~1K steps.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    """
    Encapsulates the full DiffuSeq training procedure.

    Responsibilities:
      1. Load (or initialize fresh) model + EMA parameter checkpoints.
      2. Wrap the model in DistributedDataParallel (DDP) for multi-GPU training.
      3. Run the main loop: forward → backward → optimize → EMA update → checkpoint.
      4. Periodically evaluate on validation data (forward only, no gradient).
      5. Save EMA checkpoints at a configurable interval.
    """

    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        learning_steps=0,
        checkpoint_path='',
        gradient_clipping=-1.,
        eval_data=None,
        eval_interval=-1,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data               # infinite training data generator
        self.eval_data = eval_data     # validation data generator (optional)
        self.batch_size = batch_size
        # microbatch: process the full batch in smaller chunks to fit GPU memory.
        # If microbatch <= 0, treat the entire batch as one microbatch.
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        # ema_rate can be a single float (e.g., 0.9999) or a comma-separated string
        # (e.g., "0.9999,0.9995") for tracking multiple EMA rates simultaneously.
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval     # log training loss every N steps
        self.eval_interval = eval_interval   # run validation every N steps
        self.save_interval = save_interval   # save checkpoint every N steps
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        # fall back to uniform timestep sampling if no sampler is provided
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.learning_steps = learning_steps   # total gradient steps (0 = run forever)
        self.gradient_clipping = gradient_clipping  # max gradient norm (-1 = disabled)

        self.step = 0         # steps taken in this training session
        self.resume_step = 0  # steps already taken (loaded from checkpoint)

        # global_batch = effective batch size across all GPUs
        # Used for logging: "how many samples have we processed in total?"
        self.global_batch = self.batch_size * dist.get_world_size()

        # master_params: the FP32 reference parameters used for optimizer updates.
        # In standard (non-FP16) training, model_params == master_params.
        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()

        self.checkpoint_path = checkpoint_path

        # Load weights from checkpoint if provided; then broadcast weights
        # from rank 0 to all other ranks to ensure consistent initialization.
        self._load_and_sync_parameters()

        # Build the AdamW optimizer on the master (FP32) parameters.
        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)

        if self.resume_step:
            # When resuming, reconstruct the LR at the point we left off using
            # linear annealing: lr * (1 - fraction_completed).
            frac_done = (self.step + self.resume_step) / self.learning_steps
            lr = self.lr * (1 - frac_done)
            self.opt = AdamW(self.master_params, lr=lr, weight_decay=self.weight_decay)
            # Load the corresponding EMA parameter snapshots from checkpoint files.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            # Fresh training: EMA starts as an exact copy of the initial model weights.
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            # DistributedDataParallel (DDP): wraps the model so that:
            #   - Forward passes run independently on each GPU.
            #   - Backward passes all-reduce gradients across GPUs automatically.
            # bucket_cap_mb=128: gradients are bucketed in 128MB chunks for efficient communication.
            # find_unused_parameters=False: assume all parameters receive gradients (faster).
            self.use_ddp = True
            print(dist_util.dev())
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,   # don't sync buffers (e.g., BatchNorm stats) at each step
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model  # no-op wrapper on CPU

        # GradScaler: manages the dynamic loss scaling needed for FP16 training.
        # It scales the loss up before backward() to prevent FP16 gradient underflow,
        # then unscales before the optimizer step.
        self.scaler = GradScaler()

    def _load_and_sync_parameters(self):
        """
        Load model weights from a checkpoint (if one exists) and broadcast them
        from rank 0 to all other ranks so every GPU starts with identical weights.
        """
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint[-3:] == '.pt':
            # Extract the step number embedded in the filename (e.g., model000100.pt → 100)
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            self.resume_step = 0  # NOTE: currently forces resume_step to 0 (fresh LR schedule)
            if dist.get_rank() == 0:
                # Only rank 0 reads from disk; sync_params below broadcasts to others.
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        actual_model_path(resume_checkpoint), map_location=dist_util.dev()
                    )
                )

        # Broadcast model parameters from rank 0 to all other processes.
        # This ensures all GPUs start with identical weights even if only rank 0 loaded from disk.
        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        """
        Load EMA parameter snapshot matching the given rate from checkpoint.

        Args:
            rate (float): EMA decay rate (e.g., 0.9999). Used to locate the EMA file.

        Returns:
            list[Tensor]: EMA parameter list (one tensor per model parameter).
        """
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    actual_model_path(ema_checkpoint), map_location=dist_util.dev()
                )
                # Convert the state dict (name → tensor) back to a flat parameter list
                ema_params = self._state_dict_to_master_params(state_dict)

        # Broadcast EMA params from rank 0 to all other ranks
        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        """
        Load optimizer state dict from checkpoint (currently unused / commented out).
        Restoring optimizer state allows exact continuation of momentum statistics.
        """
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if bf.exists(main_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {main_checkpoint}")
            state_dict = dist_util.load_state_dict(
                actual_model_path(main_checkpoint), map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def _setup_fp16(self):
        """
        (Currently unused) Set up legacy FP16 training by converting model weights
        to FP16 and creating FP32 master parameter copies for the optimizer.
        The current implementation uses torch.cuda.amp instead.
        """
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    # -------------------------------------------------------------------------
    # Main Training Loop
    # -------------------------------------------------------------------------

    def run_loop(self):
        """
        Main training loop. Runs until learning_steps is reached (or forever if 0).

        Each iteration:
          1. Fetch next batch from the data generator.
          2. Run a full training step (forward + backward + optimize + EMA update).
          3. Log metrics at log_interval.
          4. Evaluate on validation data at eval_interval.
          5. Save checkpoint at save_interval.
        """
        while (
            not self.learning_steps                                # run forever if 0
            or self.step + self.resume_step < self.learning_steps  # or until budget used
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)

            # Dump accumulated log metrics to stdout
            if self.step % self.log_interval == 0:
                logger.dumpkvs()

            # Validation: compute loss on one batch without updating parameters
            if self.eval_data is not None and self.step % self.eval_interval == 0:
                batch_eval, cond_eval = next(self.eval_data)
                self.forward_only(batch_eval, cond_eval)
                print('eval on validation set')
                logger.dumpkvs()

            # Save EMA checkpoints
            if self.step > 0 and self.step % self.save_interval == 0:
                self.save()
                # Early exit hook for integration tests
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return

            self.step += 1

        # Always save the final step if it wasn't caught by the save_interval condition
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        """
        Execute one full training step:
          1. forward_backward: compute loss and accumulate gradients.
          2. optimize_fp16 or optimize_normal: clip gradients, step optimizer, update EMA.
          3. log_step: record step count and sample count.
        """
        self.forward_backward(batch, cond)
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        self.log_step()

    # -------------------------------------------------------------------------
    # Forward Passes
    # -------------------------------------------------------------------------

    def forward_only(self, batch, cond):
        """
        Validation pass: compute and log losses without updating any parameters.

        Mirrors forward_backward but wraps everything in th.no_grad() and
        skips the optimizer and EMA steps.

        Args:
            batch (Tensor): [B, seq_len, hidden_dim] input embeddings (x_0).
            cond (dict):    conditioning dict with 'input_ids' and 'input_mask'.
        """
        with th.no_grad():
            zero_grad(self.model_params)
            for i in range(0, batch.shape[0], self.microbatch):
                # Slice the batch into microbatches and move to GPU
                micro = batch[i: i + self.microbatch].to(dist_util.dev())
                micro_cond = {
                    k: v[i: i + self.microbatch].to(dist_util.dev())
                    for k, v in cond.items()
                }
                last_batch = (i + self.microbatch) >= batch.shape[0]

                # Sample random diffusion timesteps for this microbatch
                t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

                # Partially apply diffusion.training_losses so it can be called later
                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.ddp_model,
                    micro,
                    t,
                    model_kwargs=micro_cond,
                )

                # For all microbatches except the last, suppress DDP gradient sync
                # (no_sync avoids the all-reduce communication until the last chunk)
                if last_batch or not self.use_ddp:
                    losses = compute_losses()
                else:
                    with self.ddp_model.no_sync():
                        losses = compute_losses()

                # Log validation losses prefixed with "eval_"
                log_loss_dict(
                    self.diffusion, t, {f"eval_{k}": v * weights for k, v in losses.items()}
                )

    def forward_backward(self, batch, cond):
        """
        Training pass: compute loss over all microbatches and accumulate gradients.

        Microbatching strategy:
          - Split the batch into chunks of size `microbatch`.
          - For all chunks except the last, call ddp_model.no_sync() to suppress
            the all-reduce until the final chunk. This batches the gradient
            communication into one round-trip per step.
          - If use_fp16, wrap the forward pass in autocast() and use scaler.scale()
            to backward through the scaled loss.

        Args:
            batch (Tensor): [B, seq_len, hidden_dim] noisy embeddings.
            cond (dict):    'input_ids' and 'input_mask' tensors.
        """
        zero_grad(self.model_params)  # clear gradients from previous step

        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]

            # Sample diffusion timesteps t ∈ [0, T-1] for each sample in the microbatch.
            # The sampler returns importance-sampling weights to correct for non-uniform sampling.
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            if self.use_fp16:
                # autocast: automatically casts eligible ops to FP16 for speed,
                # while keeping numerically sensitive ops (softmax, norms) in FP32.
                with autocast():
                    compute_losses = functools.partial(
                        self.diffusion.training_losses,
                        self.ddp_model,
                        micro,
                        t,
                        model_kwargs=micro_cond,
                    )

                    if last_batch or not self.use_ddp:
                        losses = compute_losses()
                    else:
                        with self.ddp_model.no_sync():
                            losses = compute_losses()

                    # --- NAN TRAP ---
                    if th.isnan(losses["loss"]).any() or th.isinf(losses["loss"]).any():
                        try:
                            from transformers import AutoTokenizer
                            tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
                            # Fix: model_kwargs.pop() deletes 'input_ids' from micro_cond. 
                            # We must read it from the original 'cond' dict.
                            token_ids = cond['input_ids'][i : i + self.microbatch].tolist()
                            text_seqs = [tokenizer.decode(seq, skip_special_tokens=False) for seq in token_ids]
                            text_out = "\n".join(text_seqs)
                        except Exception as e:
                            text_out = f"Failed to decode: {e}"

                        with open("NaNLossInput.txt", "a", encoding="utf-8") as f:
                            f.write("="*50 + "\n")
                            f.write(f"NaN Loss Detected at Timestep(s): {t.cpu().tolist()}\n")
                            f.write(f"Decoded Sequences:\n{text_out}\n")
                            f.write(f"Raw IDs:\n{cond['input_ids'][i : i + self.microbatch].tolist()}\n")
                            f.write("="*50 + "\n")
                        print("!!! NAN LOSS DETECTED !!! Skipped microbatch and logged to NaNLossInput.txt")
                        
                        # Memory cleanup: free the computation graph before raising error
                        del losses
                        th.cuda.empty_cache()
                        raise ValueError("NaN Loss Detected! Stopping training.")
                    # ----------------

                    # Update the timestep sampler's loss history for importance reweighting
                    if isinstance(self.schedule_sampler, LossAwareSampler):
                        self.schedule_sampler.update_with_local_losses(
                            t, losses["loss"].detach()
                        )

                    # Weighted mean loss: importance weights correct for the non-uniform
                    # timestep sampling distribution
                    loss = (losses["loss"] * weights).mean()
                    log_loss_dict(
                        self.diffusion, t, {k: v * weights for k, v in losses.items()}
                    )

                # Scale the loss before backward() to prevent FP16 gradient underflow,
                # then accumulate scaled gradients
                self.scaler.scale(loss).backward()

            else:
                # Standard FP32 training path
                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.ddp_model,
                    micro,
                    t,
                    model_kwargs=micro_cond,
                )

                if last_batch or not self.use_ddp:
                    losses = compute_losses()
                else:
                    with self.ddp_model.no_sync():
                        losses = compute_losses()

                # --- NAN TRAP ---
                if th.isnan(losses["loss"]).any() or th.isinf(losses["loss"]).any():
                    try:
                        from transformers import AutoTokenizer
                        tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
                        # Fix: model_kwargs.pop() deletes 'input_ids' from micro_cond. 
                        # We must read it from the original 'cond' dict.
                        token_ids = cond['input_ids'][i : i + self.microbatch].tolist()
                        text_seqs = [tokenizer.decode(seq, skip_special_tokens=False) for seq in token_ids]
                        text_out = "\n".join(text_seqs)
                    except Exception as e:
                        text_out = f"Failed to decode: {e}"

                    with open("NaNLossInput.txt", "a", encoding="utf-8") as f:
                        f.write("="*50 + "\n")
                        f.write(f"NaN Loss Detected at Timestep(s): {t.cpu().tolist()}\n")
                        f.write(f"Decoded Sequences:\n{text_out}\n")
                        f.write(f"Raw IDs:\n{cond['input_ids'][i : i + self.microbatch].tolist()}\n")
                        f.write("="*50 + "\n")
                    print("!!! NAN LOSS DETECTED !!! Skipped microbatch and logged to NaNLossInput.txt")
                    
                    # Memory cleanup: free the computation graph before raising error
                    del losses
                    th.cuda.empty_cache()
                    raise ValueError("NaN Loss Detected! Stopping training.")
                # ----------------

                if isinstance(self.schedule_sampler, LossAwareSampler):
                    self.schedule_sampler.update_with_local_losses(
                        t, losses["loss"].detach()
                    )

                loss = (losses["loss"] * weights).mean()
                log_loss_dict(
                    self.diffusion, t, {k: v * weights for k, v in losses.items()}
                )
                loss.backward()

    # -------------------------------------------------------------------------
    # Optimization Steps
    # -------------------------------------------------------------------------

    def optimize_fp16(self):
        """
        Optimizer step for FP16 (AMP) training:
          1. Unscale gradients (crucial before clipping!).
          2. (Optional) Clip gradients.
          3. Anneal learning rate linearly.
          4. scaler.step(opt): check for inf/NaN, then call opt.step().
          5. scaler.update(): adjust the scale factor for the next iteration.
          6. Log gradient norm.
          7. Update all EMA parameter copies.
        """
        # CRITICAL: Must unscale gradients BEFORE clipping, otherwise the scale factor
        # (e.g., 65536) causes the clipped gradients to vanish when scaler.step() unscales them!
        self.scaler.unscale_(self.opt)

        if self.gradient_clipping > 0:
            self.grad_clip()
        
        self._anneal_lr()
        # scaler.step handles unscaling + optimizer step atomically.
        # Since we already unscaled, it just checks for inf/NaN and calls opt.step().
        self.scaler.step(self.opt)
        self.scaler.update()   # double or halve the scale based on whether this step worked
        self._log_grad_norm()
        # After the optimizer step, update each EMA copy with the new master params
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def grad_clip(self):
        """
        Clip gradients to prevent exploding gradients.

        Uses the optimizer's own clip_grad_norm if available (e.g., sharded optimizers),
        otherwise falls back to torch's standard gradient clipping.
        """
        max_grad_norm = self.gradient_clipping
        if hasattr(self.opt, "clip_grad_norm"):
            # Some sharded optimizers expose this method directly
            self.opt.clip_grad_norm(max_grad_norm)
        else:
            # Standard PyTorch gradient clipping: scales all gradients so their
            # global L2 norm doesn't exceed max_grad_norm
            th.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_grad_norm,
            )

    def optimize_normal(self):
        """
        Optimizer step for standard FP32 training:
          1. (Optional) Clip gradients.
          2. Log gradient norm.
          3. Anneal learning rate linearly.
          4. Optimizer step.
          5. Update all EMA copies.
        """
        if self.gradient_clipping > 0:
            self.grad_clip()
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    # -------------------------------------------------------------------------
    # Utilities: Logging, LR Scheduling, Checkpointing
    # -------------------------------------------------------------------------

    def _log_grad_norm(self):
        """
        Compute and log the L2 norm of all gradients (across all parameters).
        Useful for detecting gradient explosion or vanishing during training.
        """
        sqsum = 0.0
        for p in self.master_params:
            if p.grad is not None:
                sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        """
        Apply linear learning rate decay.

        Schedule: lr(t) = lr_initial * (1 - t / T)
        where t = current step, T = total learning_steps.

        The LR reaches 0 at the end of training, acting as a linear warmdown.
        If learning_steps == 0 (infinite training), this is a no-op.
        """
        if not self.learning_steps:
            return
        frac_done = (self.step + self.resume_step) / self.learning_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        """Log the global step count and total number of samples processed so far."""
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        """
        Save EMA parameter snapshots to disk (rank 0 only).

        For each EMA rate, writes a file named:
            ema_{rate}_{step:06d}.pt
        to self.checkpoint_path.

        Note: the plain model weights (rate=0) are commented out — only EMA
        weights are saved since they typically generalize better at inference.

        All ranks call dist.barrier() at the end to synchronize before training resumes.
        """
        def save_checkpoint(rate, params):
            # Convert the flat parameter list back to a named state dict
            state_dict = self._master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step + self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step + self.resume_step):06d}.pt"
                print('writing to', bf.join(get_blob_logdir(), filename))
                print('writing to', bf.join(self.checkpoint_path, filename))
                # Save to the local checkpoint directory
                with bf.BlobFile(bf.join(self.checkpoint_path, filename), "wb") as f:
                    th.save(state_dict, f)

        # Only save EMA checkpoints (not the raw model weights)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        # Synchronize all ranks before training resumes — ensures the checkpoint
        # is fully written before any process continues
        dist.barrier()

    def _master_params_to_state_dict(self, master_params):
        """
        Convert a flat list of master (FP32) tensors back to a named state dict
        compatible with model.load_state_dict().

        Args:
            master_params (list[Tensor]): flat parameter list aligned with model.named_parameters().

        Returns:
            dict: {param_name: tensor} state dict.
        """
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        """
        Convert a named state dict back to a flat parameter list in the same order
        as model.named_parameters(). This is the inverse of _master_params_to_state_dict.

        Args:
            state_dict (dict): {param_name: tensor}

        Returns:
            list[Tensor]: flat master parameter list.
        """
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        return params


# =============================================================================
# Helper Functions
# =============================================================================

def parse_resume_step_from_filename(filename):
    """
    Extract the step count from a checkpoint filename.

    Expected format: path/to/modelNNNNNN.pt
    Example: 'diffusion_models/ema_0.9999_000100.pt' → 100

    Args:
        filename (str): path to the checkpoint file.

    Returns:
        int: the step number, or 0 if the file doesn't match the expected format.
    """
    if filename[-3:] == '.pt':
        return int(filename[-9:-3])   # last 6 chars before ".pt"
    else:
        return 0


def get_blob_logdir():
    """
    Return the directory to use for blob (remote) storage logging.
    Falls back to the local logger directory if the env var isn't set.
    """
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():
    """
    Auto-discover the latest checkpoint on blob storage.
    Currently returns None (no auto-discovery); checkpoint path is passed explicitly.

    Override this on cloud infrastructure to scan a GCS/S3 bucket automatically.
    """
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    """
    Given a main checkpoint path, construct and verify the corresponding EMA checkpoint.

    EMA filename convention: ema_{rate}_{step:06d}.pt
    The EMA file is expected to be in the same directory as the main checkpoint.

    Args:
        main_checkpoint (str): path to the main model checkpoint.
        step (int): the training step number.
        rate (float): the EMA decay rate.

    Returns:
        str | None: path to the EMA checkpoint if it exists, else None.
    """
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    """
    Log per-key losses and their per-quartile breakdowns.

    For each loss key (e.g., 'loss', 'mse', 'vb'):
      - Logs the mean value across the batch.
      - Splits timesteps into 4 quartiles [0,T/4), [T/4,T/2), [T/2,3T/4), [3T/4,T)
        and logs the mean loss in each quartile separately.
        This reveals whether the model struggles more at low or high noise levels.

    Args:
        diffusion: the GaussianDiffusion object (provides num_timesteps).
        ts (Tensor): 1D tensor of sampled timestep indices.
        losses (dict): {key: Tensor of per-sample losses}.
    """
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log quartile-stratified loss: q0 = easiest (low noise), q3 = hardest (high noise)
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


def actual_model_path(model_path):
    """
    Resolve the actual path to the model file.
    Currently a pass-through; can be overridden on cloud setups
    where paths may need to be remapped to local cache locations.
    """
    return model_path