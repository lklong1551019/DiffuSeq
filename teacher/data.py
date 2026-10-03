"""
Teacher data pipeline: JSONL pairs -> token ids -> length-bucketed batches -> padded tensors.

Walkthrough (docs/plans/teacher-model.md §5), pairs ("I sing .", "Tôi hát ."), ("Thank you .", "Cảm ơn các bạn ."):
  1. encode_pairs: source ids  [▁I ▁sing ▁. </s>], [▁Thank ▁you ▁. </s>]                 lengths 4, 4
                   target ids  [▁Tôi ▁hát ▁. </s>], [▁Cảm ▁ơn ▁các ▁bạn ▁. </s>]          lengths 4, 6
  2. TokenBucketBatcher: index batches with rows * longest(source, target) <= max_tokens
  3. collate: input_ids [2, 4] (pad 0), attention_mask [2, 4],
              labels [2, 6] with target padding = -100 (ignored by the loss);
              the model builds decoder_input_ids = shift_tokens_right(labels) itself.
"""

import json
import random

import torch

from .tokenizer import EOS_ID, PAD_ID

IGNORE_INDEX = -100


def read_pairs(path, max_rows=None):
    """(src, trg) pairs of a {train,valid,test}.jsonl file."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            pairs.append((row["src"], row["trg"]))
            if max_rows is not None and len(pairs) >= max_rows:
                break
    return pairs


def encode_pairs(pairs, tokenizer, max_len=256):
    """Token ids of every pair; sequences longer than max_len are cut and re-terminated with </s>."""
    src = tokenizer([s for s, _ in pairs])["input_ids"]
    trg = tokenizer([t for _, t in pairs])["input_ids"]

    def cut(ids):
        return ids if len(ids) <= max_len else ids[:max_len - 1] + [EOS_ID]

    return [(cut(s), cut(t)) for s, t in zip(src, trg)]


class TokenBucketBatcher:
    """
    Batches of indices whose padded size stays within a token budget.

    Step by step, per epoch:
      1. order indices by (max(len(src), len(trg)), random tie-break)       -> similar lengths together
      2. walk in that order; start a new batch when (rows + 1) * longest > max_tokens
      3. shuffle the batch order with seed + epoch                           -> no length curriculum
    Every index appears in exactly one batch per epoch; a single row longer than max_tokens forms its own batch.
    """

    def __init__(self, lengths, max_tokens, seed=1, shuffle=True):
        self.lengths = list(lengths)
        self.max_tokens = max_tokens
        self.seed = seed
        self.shuffle = shuffle

    def batches(self, epoch=0):
        rng = random.Random(self.seed + epoch)
        order = sorted(range(len(self.lengths)), key=lambda i: (self.lengths[i], rng.random()))
        batches, current, longest = [], [], 0
        for i in order:
            new_longest = max(longest, self.lengths[i])
            if current and (len(current) + 1) * new_longest > self.max_tokens:
                batches.append(current)
                current, new_longest = [], self.lengths[i]
            current.append(i)
            longest = new_longest
        if current:
            batches.append(current)
        if self.shuffle:
            rng.shuffle(batches)
        return batches


def pair_lengths(encoded):
    return [max(len(s), len(t)) for s, t in encoded]


def collate(encoded, indices):
    """
    Padded tensors for one batch.

    input_ids      [B, S]  source ids, padded with PAD_ID
    attention_mask [B, S]  1 on real source tokens
    labels         [B, T]  target ids, padded with IGNORE_INDEX (-100)
    """
    srcs = [encoded[i][0] for i in indices]
    trgs = [encoded[i][1] for i in indices]
    s_len, t_len = max(map(len, srcs)), max(map(len, trgs))
    input_ids = torch.full((len(indices), s_len), PAD_ID, dtype=torch.long)
    attention_mask = torch.zeros((len(indices), s_len), dtype=torch.long)
    labels = torch.full((len(indices), t_len), IGNORE_INDEX, dtype=torch.long)
    for row, (s, t) in enumerate(zip(srcs, trgs)):
        input_ids[row, :len(s)] = torch.tensor(s)
        attention_mask[row, :len(s)] = 1
        labels[row, :len(t)] = torch.tensor(t)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
