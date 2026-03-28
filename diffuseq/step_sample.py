"""
step_sample.py — Diffusion timestep samplers for training.

During training, the diffusion model is conditioned on a random timestep t ∈ [0, T-1].
Naively, t could be sampled uniformly. However, some timesteps contribute more to the
loss variance than others. This module provides samplers that can reweight timestep
sampling to reduce gradient variance and speed up convergence.

Classes:
  ScheduleSampler (ABC):            base class defining the sampling interface.
  UniformSampler:                   samples t uniformly → equal weight to all timesteps.
  FixSampler:                       fixed custom weights (first half heavier than second).
  LossAwareSampler (ABC):           base class for samplers that track loss history.
  LossSecondMomentResampler:        reweights by √E[loss²] per timestep (importance sampling).

Factory:
  create_named_schedule_sampler:    builds a sampler by name ('uniform', 'lossaware', 'fixstep').
"""

from abc import ABC, abstractmethod

import numpy as np
import torch as th
import torch.distributed as dist


def create_named_schedule_sampler(name, diffusion):
    """
    Factory function: build a ScheduleSampler by name.

    Args:
        name (str): one of 'uniform', 'lossaware', 'fixstep'.
        diffusion: the GaussianDiffusion / SpacedDiffusion object
                   (provides num_timesteps used to set up the weight array).

    Returns:
        ScheduleSampler subclass instance.

    Raises:
        NotImplementedError: if name is not recognized.
    """
    if name == "uniform":
        return UniformSampler(diffusion)
    elif name == "lossaware":
        return LossSecondMomentResampler(diffusion)
    elif name == "fixstep":
        return FixSampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(ABC):
    """
    Abstract base class for diffusion timestep samplers.

    A ScheduleSampler defines a probability distribution over diffusion timesteps
    [0, T-1]. At each training step, a batch of timesteps is sampled from this
    distribution. To keep the training objective unbiased (importance sampling),
    each sample's loss is scaled by 1 / (T * p(t)) — the inverse of its sampling
    probability.

    Subclasses must implement:
        weights() → np.ndarray of shape [T]: unnormalized positive weights per timestep.
    """

    @abstractmethod
    def weights(self):
        """
        Return unnormalized positive weights for each diffusion timestep.

        Returns:
            np.ndarray of shape [num_timesteps]: weights[t] is proportional to p(t).
        """

    def sample(self, batch_size, device):
        """
        Sample a batch of timesteps using importance sampling.

        Steps:
          1. Normalize weights → probability distribution p.
          2. Draw batch_size timestep indices from p.
          3. Compute importance weights w_i = 1 / (T * p(t_i)) to correct for
             the non-uniform distribution, keeping E[loss] unbiased.

        Args:
            batch_size (int): number of timesteps to sample (= number of samples in microbatch).
            device (torch.device): device for the output tensors.

        Returns:
            (Tensor, Tensor):
              - indices: [batch_size] long tensor of sampled timestep indices.
              - weights: [batch_size] float tensor of importance-sampling correction weights.
        """
        w = self.weights()
        p = w / np.sum(w)                                               # normalize to probabilities
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p) # sample from the distribution
        indices = th.from_numpy(indices_np).long().to(device)
        # Importance weight: accounts for oversampling high-loss timesteps
        # w_i = 1 / (T * p(t_i)). When p is uniform, all weights = 1.
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    """
    Samples timesteps uniformly at random from [0, T-1].

    All importance weights equal 1.0, so no correction is applied to the loss.
    Simple and effective; use as a baseline.
    """

    def __init__(self, diffusion):
        self.diffusion = diffusion
        # All timesteps get the same weight = 1.0
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        return self._weights


class FixSampler(ScheduleSampler):
    """
    Applies a fixed, manually-defined weight schedule across timesteps.

    Default behaviour: first half of timesteps get weight 1.0, second half get 0.5.
    This oversamples early timesteps (low noise), which can be useful if the model
    needs more training signal at clean inputs.

    Customize the weights array to experiment with different curricula.
    """

    def __init__(self, diffusion):
        self.diffusion = diffusion

        ###############################################################
        ### You can customize your own sampling weight of steps here. ###
        ###############################################################
        # First half: weight 1.0, second half: weight 0.5
        self._weights = np.concatenate([
            np.ones([diffusion.num_timesteps // 2]),
            np.zeros([diffusion.num_timesteps // 2]) + 0.5
        ])

    def weights(self):
        return self._weights


class LossAwareSampler(ScheduleSampler):
    """
    Abstract base for samplers that adapt weights based on observed training losses.

    In importance sampling, the optimal proposal distribution is proportional to
    the standard deviation of the loss at each timestep. Higher-variance timesteps
    should be sampled more frequently to reduce overall gradient variance.

    This class handles the distributed loss collection: each GPU process calls
    update_with_local_losses() with its local timestep/loss pairs, and this
    method all-gathers them across all ranks before calling update_with_all_losses().

    Subclasses must implement:
        update_with_all_losses(ts, losses): update internal state from all-gathered losses.
    """

    def update_with_local_losses(self, local_ts, local_losses):
        """
        Collect local (per-GPU) losses, synchronize across all ranks via all_gather,
        then call update_with_all_losses() with the combined data.

        This ensures that the weight update is identical on all ranks (deterministic),
        keeping the samplers in sync without extra communication.

        Args:
            local_ts (Tensor):     1D integer tensor of sampled timestep indices (this GPU).
            local_losses (Tensor): 1D float tensor of corresponding losses (this GPU).
        """
        # Step 1: exchange batch sizes across ranks (they may differ near dataset boundaries)
        batch_sizes = [
            th.tensor([0], dtype=th.int32, device=local_ts.device)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(
            batch_sizes,
            th.tensor([len(local_ts)], dtype=th.int32, device=local_ts.device),
        )

        # Step 2: pad all tensors to the maximum batch size and all_gather
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)

        timestep_batches = [th.zeros(max_bs).to(local_ts) for bs in batch_sizes]
        loss_batches = [th.zeros(max_bs).to(local_losses) for bs in batch_sizes]
        dist.all_gather(timestep_batches, local_ts)
        dist.all_gather(loss_batches, local_losses)

        # Step 3: flatten and trim padding, then call the update hook
        timesteps = [
            x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ]
        losses = [x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """
        Update the timestep weight distribution using the all-gathered loss data.

        Called on every rank with identical arguments, so the update must be
        deterministic to keep all ranks synchronized.

        Args:
            ts (list[int]):     timestep indices from all GPUs.
            losses (list[float]): corresponding loss values.
        """


class LossSecondMomentResampler(LossAwareSampler):
    """
    Resampler that weights timesteps by the square root of their mean squared loss.

    Theory (from "Improved DDPM", Nichol & Dhariwal 2021):
        The optimal importance sampling distribution is p*(t) ∝ std(loss(t)).
        Approximating std with √E[loss²] (the second moment) gives a practical
        on-line importance sampler that concentrates samples on high-variance timesteps.

    Implementation:
        - Maintains a sliding history buffer of size `history_per_term` losses per timestep.
        - During warm-up (buffer not full), falls back to uniform sampling.
        - A small uniform_prob (0.001) is mixed in to prevent any timestep from
          being sampled with probability 0.

    Args:
        diffusion:         the diffusion object (provides num_timesteps).
        history_per_term:  number of recent loss values to keep per timestep (default 10).
        uniform_prob:      fraction of the distribution reserved for uniform exploration (default 0.001).
    """

    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        self.diffusion = diffusion
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        # _loss_history[t, :]: ring buffer of the last `history_per_term` losses at timestep t
        self._loss_history = np.zeros(
            [diffusion.num_timesteps, history_per_term], dtype=np.float64
        )
        # _loss_counts[t]: number of losses recorded so far at timestep t (capped at history_per_term)
        self._loss_counts = np.zeros([diffusion.num_timesteps], dtype=int)

    def weights(self):
        """
        Compute the current sampling weights.

        Before warm-up: returns uniform weights (all 1.0).
        After warm-up:
            weights[t] = √(mean(loss_history[t]²))   ← RMS loss at each timestep
            normalized, then mixed with a small uniform component.
        """
        if not self._warmed_up():
            # Not enough data yet: sample uniformly
            return np.ones([self.diffusion.num_timesteps], dtype=np.float64)
        # RMS loss per timestep: higher = more variance = sample more often
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        weights /= np.sum(weights)           # normalize to a probability distribution
        weights *= 1 - self.uniform_prob     # scale down by (1 - ε)
        weights += self.uniform_prob / len(weights)  # add uniform floor ε
        return weights

    def update_with_all_losses(self, ts, losses):
        """
        Update the loss history buffer with newly observed (timestep, loss) pairs.

        For each timestep t:
          - If buffer not full: append the new loss.
          - If buffer full: shift left and append (sliding window / FIFO).

        Args:
            ts (list[int]):     global list of timestep indices (all GPUs).
            losses (list[float]): global list of corresponding losses.
        """
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # Shift the ring buffer left by 1 and write the new loss at the end
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                # Buffer not yet full: fill sequentially
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        """
        Returns True if every timestep has accumulated at least `history_per_term` loss values.
        Until then, the sampler uses uniform weights.
        """
        return (self._loss_counts == self.history_per_term).all()