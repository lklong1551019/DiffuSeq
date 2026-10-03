"""
text_datasets.py — data loading for DiffuSeq seq2seq training and decoding.

Pipeline:
  1. get_corpus            reads {train,valid,test}.jsonl: {"src", "trg", ["direction"], ["graph_src"]}.
  2. helper_tokenize       tokenizes, merges source and target into one sequence, pads, builds masks
                           (pure per-row logic in merge_pair / shift_source_graph / pad_to).
  3. TextDataset           returns (embeddings, {input_ids, input_mask, rel_mask[, edge_index, edge_type]}).
  4. collate_with_adj      stacks a batch; concatenates per-sample graphs into one disjoint graph.
  5. load_data_text        DataLoader (+ DistributedSampler for train/valid), infinite when loop=True.

Sequence layout (real output, seq_len 16, source "Thank you .", target "Cảm ơn các bạn ."):

    pos    0     1     2   3    4     5     6   7   8   9  10  11  12  13   14    15
    token  [CLS] Thank you .   [SEP] [SEP] [CLS] C ##ảm ơ ##n các bạn .  [SEP] [PAD]
    mask   0     0     0   0    0     0     1   1   1   1   1   1   1   1    1     1

    - source = [CLS] src [SEP]; one extra [SEP] separates source and target;
    - mask 0 = source (kept clean during diffusion), mask 1 = target region (noised and generated),
      [PAD] included: the model generates padding, which is how output length is decided;
    - decode: len_x = seq_len - sum(mask) is the source length (6 above).

Graph input (AMR source only): "graph_src" entries [head_pos, dep_pos, label, label_pos] are token
positions inside the tokenized source string ([CLS] = 0), computed by prepare_docamr_datasets.py.
A target-side graph is never read: at decoding time it would hand the model the structure of the
answer (review bug B11).
"""

import glob
import json
import os

import datasets
import numpy as np
import psutil
import torch
from datasets import Dataset as Dataset2
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from itertools import chain

from basic_utils import (AMR_TO_TEXT_LABEL, AMR_TO_TEXT_TOKEN, TEXT_TO_AMR_LABEL, TEXT_TO_AMR_TOKEN,
                         load_relation_vocab)

TOKENIZE_NUM_PROC = 4   # parallel workers for tokenization (tests set 1)


def load_data_text(
    batch_size,
    seq_len,
    deterministic=False,
    data_args=None,
    model_emb=None,
    split='train',
    loaded_vocab=None,
    loop=True,
    filter_direction=None,
):
    """
    Build the data generator for one split.

    Yields (batch, cond):
        batch: [B, seq_len, hidden_dim] float embeddings (not used by the loss, see load_model_emb).
        cond:  input_ids [B, L], input_mask [B, L], rel_mask [B, L]
               (+ edge_index [2, E], edge_type [E] when data_args.graph_encoder != 'none').
    filter_direction: keep only rows of this direction (rows without a direction are always kept).
    """
    print('#' * 30, '\nLoading text data...')

    if "pretrain" in data_args.notes:
        print("#### Load Pretrain Data, fold=", data_args.data_split_num)
        training_data = get_corpus_pretrain(
            data_args, seq_len, split=split, loaded_vocab=loaded_vocab, split_num=data_args.data_split_num
        )
    else:
        training_data = get_corpus(
            data_args, seq_len, split=split, loaded_vocab=loaded_vocab, filter_direction=filter_direction
        )

    dataset = TextDataset(training_data, data_args, model_emb=model_emb)

    if split != 'test':
        data_loader = DataLoader(
            dataset, batch_size=batch_size, sampler=DistributedSampler(dataset),
            num_workers=4, collate_fn=collate_with_adj,
        )
    else:
        data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=not deterministic,
            num_workers=4, collate_fn=collate_with_adj,
        )

    return infinite_loader(data_loader) if loop else iter(data_loader)


def infinite_loader(data_loader):
    while True:
        yield from data_loader


# --------------------------------------------------------------------------------------------------
# Pure per-row helpers (unit-tested in tests/test_text_datasets.py)
# --------------------------------------------------------------------------------------------------

def build_rel_mask(target_tokens, direction):
    """
    0/1 mask over the target encoding marking AMR relation tokens; used by `mask_docamr_rel`.

    Only TEXT_TO_AMR rows (target = AMR) get 1s; every other row (plain text, AMR_TO_TEXT) is all 0.
    A relation is a token starting with ':' (e.g. ':ARG0' as one added token); its '##'
    continuation pieces are marked as well.

    Example: tokens [CLS] ( sing :ARG0 i ) [SEP], direction TEXT_TO_AMR -> [0,0,0,1,0,0,0]
    """
    mask = [0] * len(target_tokens)
    if direction != TEXT_TO_AMR_LABEL:
        return mask
    in_rel = False
    for j, tok in enumerate(target_tokens):
        if tok.startswith(':'):
            in_rel = True
        elif not tok.startswith('##'):
            in_rel = False
        mask[j] = 1 if in_rel else 0
    return mask


def merge_pair(src_ids, trg_ids, rel_trg, sep_id, seq_len):
    """
    Merge one source/target pair into the DiffuSeq layout (before padding).

    Inputs: src_ids = [CLS] s1 .. sn [SEP], trg_ids = [CLS] t1 .. tm [SEP], rel_trg aligned with trg_ids.
    Steps:
      1. drop the final [SEP] of both:  src = [CLS] s1..sn,  trg = [CLS] t1..tm
      2. trim while len(src) + len(trg) > seq_len - 3, popping from the end of the longer one
         (both when equal); the 3 reserved slots are the two re-appended [SEP] and the separator
      3. re-append [SEP] to both
      4. input_ids = src + [SEP] + trg
      5. input_mask = 0 for src and the separator (len(src) + 1 zeros); pad_to adds the 1s
      6. rel_mask = 0 for src and separator, then rel_trg (trimmed alike), 0 for the final [SEP]
    Returns dict: input_ids, input_mask, rel_mask, src_len (= len(src) incl. its [SEP]),
                  trg_start (= src_len + 1, first target position).
    """
    end_token = src_ids[-1]
    src = list(src_ids[:-1])
    trg = list(trg_ids[:-1])
    rel = list(rel_trg[:-1])
    assert len(rel) == len(trg), "rel_mask must align with the target ids"

    while len(src) + len(trg) > seq_len - 3:
        if len(src) > len(trg):
            src.pop()
        elif len(src) < len(trg):
            trg.pop()
            rel.pop()
        else:
            src.pop()
            trg.pop()
            rel.pop()

    src.append(end_token)
    trg.append(end_token)
    input_ids = src + [sep_id] + trg
    input_mask = [0] * (len(src) + 1)
    rel_mask = [0] * (len(src) + 1) + rel + [0]

    assert len(input_ids) == len(rel_mask) <= seq_len
    return {
        'input_ids': input_ids,
        'input_mask': input_mask,
        'rel_mask': rel_mask,
        'src_len': len(src),
        'trg_start': len(src) + 1,
    }


def shift_source_graph(graph, has_direction, src_len):
    """
    Move source-graph positions into the merged sequence and drop entries lost to trimming.

    graph:         [[head_pos, dep_pos, label, label_pos], ...], positions in the source encoding
                   without a direction token ([CLS] = 0).
    has_direction: a direction token was inserted at position 1, so every position >= 1 moves +1.
    src_len:       length of the trimmed source incl. its final [SEP] (merge_pair['src_len']).
                   Real source tokens occupy positions 1 .. src_len - 2.
    Positions of the source part are identical in the merged sequence (the source starts at 0).

    Example: graph [[2, 4, ':ARG0', 3]], has_direction True, src_len 8 -> [[3, 5, ':ARG0', 4]]
             same graph, src_len 5 (trimmed to [CLS] [DIR] ( sing [SEP]) -> []
    """
    shift = 1 if has_direction else 0
    out = []
    for head, dep, label, label_pos in graph:
        moved = [p + shift if p >= 1 else p for p in (head, dep, label_pos)]
        if all(1 <= p <= src_len - 2 for p in moved):
            out.append([moved[0], moved[1], label, moved[2]])
    return out


def pad_to(seq, value, seq_len):
    """Right-pad `seq` with `value` to exactly seq_len (sequences are never longer here)."""
    assert len(seq) <= seq_len, f"sequence of length {len(seq)} exceeds seq_len {seq_len}"
    return list(seq) + [value] * (seq_len - len(seq))


def split_source_target(seq, input_mask):
    """
    Decode-time split: len_x = seq_len - sum(mask) source positions, the rest is the target region.

    Valid because input_mask is a 0-prefix followed by a 1-suffix (asserted).
    Example: mask [0,0,0,0,0,0,1,1,1,1] -> source = seq[:6], target = seq[6:]
    """
    mask = [int(m) for m in input_mask]
    len_x = len(mask) - sum(mask)
    assert all(m == 0 for m in mask[:len_x]) and all(m == 1 for m in mask[len_x:]), \
        "input_mask must be a 0-prefix followed by a 1-suffix"
    return seq[:len_x], seq[len_x:]


def build_edges(graph, graph_mode, rel_to_id=None):
    """
    Turn merged-sequence graph entries into (edge_index [2, E], edge_type [E]) for one sample.

    graph_mode 'edge_attr': concept -> concept edges; type = 2 * rel_id (+1 for the reverse edge),
                            rel_id from rel_to_id (0 = unknown relation).
    graph_mode 'levi':      relation tokens become nodes: head -> label (0), label -> dep (1),
                            label -> head (2), dep -> label (3); no relation vocabulary needed.
    Example ('levi'): [[6, 8, ':ARG0', 7]] -> edges 6->7, 7->8, 7->6, 8->7 with types 0, 1, 2, 3
    """
    src, dst, typ = [], [], []
    for head, dep, label, label_pos in graph:
        if graph_mode == 'edge_attr':
            r = (rel_to_id or {}).get(label, 0)
            src += [head, dep]
            dst += [dep, head]
            typ += [2 * r, 2 * r + 1]
        elif graph_mode == 'levi':
            src += [head, label_pos, label_pos, dep]
            dst += [label_pos, dep, head, label_pos]
            typ += [0, 1, 2, 3]
        else:
            raise ValueError(f"graph_mode must be 'edge_attr' or 'levi', got {graph_mode!r}")
    if not src:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long), torch.tensor(typ, dtype=torch.long)


def format_position_table(ids, mask, id_to_token, limit=40):
    """Debug table pos / token / mask for the first `limit` positions (Indexing / Masking Rule)."""
    toks = [str(id_to_token(int(i)))[:7] for i in ids[:limit]]
    rows = [
        'pos   ' + ' '.join(f'{i:>7}' for i in range(len(toks))),
        'token ' + ' '.join(f'{t:>7}' for t in toks),
        'mask  ' + ' '.join(f'{int(m):>7}' for m in mask[:limit]),
    ]
    return '\n'.join(rows)


# --------------------------------------------------------------------------------------------------
# Dataset construction
# --------------------------------------------------------------------------------------------------

def helper_tokenize(sentence_lst, vocab_dict, seq_len):
    """
    Tokenize, merge, and pad a seq2seq corpus.

    sentence_lst: {'src': [...], 'trg': [...], 'direction': [...|None], 'graph_src': [JSON str]}
    Returns a DatasetDict {'train': Dataset} with input_ids, input_mask, rel_mask (all length
    seq_len) and graph (JSON string, positions in the merged sequence).
    """
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    raw_datasets = Dataset2.from_dict(sentence_lst)

    def ids_to_tokens(ids):
        if isinstance(vocab_dict.tokenizer, dict):
            return [vocab_dict.rev_tokenizer.get(int(i), '[UNK]') for i in ids]
        return vocab_dict.tokenizer.convert_ids_to_tokens(ids)

    def tokenize_function(examples):
        # The direction token goes right after [CLS]: "[AMR_TO_TEXT] <src>" -> [CLS] [AMR_TO_TEXT] ...
        src_texts = []
        for src, direction in zip(examples['src'], examples['direction']):
            if direction:
                token = TEXT_TO_AMR_TOKEN if direction == TEXT_TO_AMR_LABEL else AMR_TO_TEXT_TOKEN
                src_texts.append(f"{token} {src}")
            else:
                src_texts.append(src)
        input_id_x = vocab_dict.encode_token(src_texts)
        input_id_y = vocab_dict.encode_token(examples['trg'])
        rel_mask_y = [build_rel_mask(ids_to_tokens(y), d) for y, d in zip(input_id_y, examples['direction'])]
        return {'input_id_x': input_id_x, 'input_id_y': input_id_y, 'rel_mask_y': rel_mask_y}

    tokenized = raw_datasets.map(
        tokenize_function, batched=True, num_proc=TOKENIZE_NUM_PROC,
        remove_columns=['src', 'trg'], desc="Running tokenizer on dataset",
    )

    def merge_and_pad(group):
        out = {'input_ids': [], 'input_mask': [], 'rel_mask': [], 'graph': []}
        for x, y, rel, direction, graph_json in zip(group['input_id_x'], group['input_id_y'],
                                                    group['rel_mask_y'], group['direction'],
                                                    group['graph_src']):
            merged = merge_pair(x, y, rel, vocab_dict.sep_token_id, seq_len)
            graph = shift_source_graph(json.loads(graph_json), bool(direction), merged['src_len'])
            out['input_ids'].append(pad_to(merged['input_ids'], vocab_dict.pad_token_id, seq_len))
            out['input_mask'].append(pad_to(merged['input_mask'], 1, seq_len))
            out['rel_mask'].append(pad_to(merged['rel_mask'], 0, seq_len))
            out['graph'].append(json.dumps(graph))
        return out

    lm_datasets = tokenized.map(
        merge_and_pad, batched=True, num_proc=1, desc="merge and pad",
        remove_columns=['input_id_x', 'input_id_y', 'rel_mask_y', 'direction', 'graph_src'],
    )

    first = lm_datasets[0]
    print('### First example after merge/pad:')
    print(format_position_table(first['input_ids'], first['input_mask'], lambda i: ids_to_tokens([i])[0]))
    print(lm_datasets)
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    raw = datasets.DatasetDict()
    raw['train'] = lm_datasets
    return raw


def helper_tokenize_pretrain(sentence_lst, vocab_dict, seq_len, mask_ratio=0.5):
    """
    Pre-training data (upstream DiffuSeq): concatenate all texts, cut into seq_len blocks, and give
    each block a random source prefix of length in [0, mask_ratio * seq_len).
    """
    raw_datasets = Dataset2.from_dict(sentence_lst)

    def tokenize_function(examples):
        return {'input_ids': vocab_dict.encode_token(examples['text'])}

    tokenized_datasets = raw_datasets.map(
        tokenize_function, batched=True, num_proc=4, remove_columns=['text'],
        keep_in_memory=True, desc="Running tokenizer on dataset",
    )
    block_size = seq_len

    def group_texts(examples):
        concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
                  for k, t in concatenated.items()}
        random_mask_len = [np.random.randint(mask_ratio * block_size) for _ in range(len(result["input_ids"]))]
        result["input_mask"] = [[0] * l + [1] * (block_size - l) for l in random_mask_len]
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts, batched=True, num_proc=4, keep_in_memory=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )
    raw = datasets.DatasetDict()
    raw['train'] = lm_datasets
    return raw


def get_corpus(data_args, seq_len, split='train', loaded_vocab=None, filter_direction=None):
    """Read {split}.jsonl from data_args.data_dir and tokenize it (see helper_tokenize)."""
    print('#' * 30, '\nLoading dataset {} from {}...'.format(data_args.dataset, data_args.data_dir))
    if split not in ('train', 'valid', 'test'):
        raise ValueError(f"invalid split {split!r}")
    path = f'{data_args.data_dir}/{split}.jsonl'

    sentence_lst = {'src': [], 'trg': [], 'direction': [], 'graph_src': []}
    with open(path, 'r', encoding='utf-8') as f_reader:
        for row in f_reader:
            line = json.loads(row)
            direction = line.get('direction')
            if filter_direction is not None and direction is not None and direction != filter_direction:
                continue
            sentence_lst['src'].append(line['src'].strip())
            sentence_lst['trg'].append(line['trg'].strip())
            sentence_lst['direction'].append(direction)
            sentence_lst['graph_src'].append(json.dumps(line.get('graph_src', [])))

    print(f'### {len(sentence_lst["src"])} rows; samples:', sentence_lst['src'][:2], sentence_lst['trg'][:2])
    return helper_tokenize(sentence_lst, loaded_vocab, seq_len)


def get_corpus_pretrain(data_args, seq_len, split='train', loaded_vocab=None, split_num=0):
    """Upstream pre-training corpus: second half of shard `split_num`; last 5,000 texts = valid."""
    sentence_lst = {'text': []}
    path = sorted(glob.glob(f"{data_args.data_dir}/*jsonl"))[split_num]
    with open(path, 'r') as f_reader:
        for row in f_reader:
            sentence_lst['text'].append(json.loads(row)['text'].strip())
    sentence_lst['text'] = sentence_lst['text'][len(sentence_lst['text']) // 2:]
    if split == 'train':
        sentence_lst['text'] = sentence_lst['text'][:-5000]
    elif split == 'valid':
        sentence_lst['text'] = sentence_lst['text'][-5000:]
    return helper_tokenize_pretrain(sentence_lst, loaded_vocab, seq_len)


class TextDataset(Dataset):
    """
    Per sample: embeddings of input_ids plus the condition dict.

    Graph tensors are built only when data_args.graph_encoder != 'none'
    (build_edges with data_args.graph_mode; relation ids from paths.AMR_VOCAB_FILE).
    """

    def __init__(self, text_datasets, data_args, model_emb=None):
        super().__init__()
        self.text_datasets = text_datasets
        self.length = len(self.text_datasets['train'])
        self.data_args = data_args
        self.model_emb = model_emb
        self.graph_mode = None
        self.rel_to_id = None
        if getattr(data_args, 'graph_encoder', 'none') != 'none':
            self.graph_mode = getattr(data_args, 'graph_mode', 'edge_attr')
            if self.graph_mode == 'edge_attr':
                self.rel_to_id = load_relation_vocab()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        row = self.text_datasets['train'][idx]
        with torch.no_grad():
            hidden_state = self.model_emb(torch.tensor(row['input_ids']))
            arr = hidden_state.numpy().astype(np.float32)

        out_kwargs = {
            'input_ids': np.array(row['input_ids']),
            'input_mask': np.array(row['input_mask']),
        }
        if 'rel_mask' in row:
            out_kwargs['rel_mask'] = np.array(row['rel_mask'])
        if self.graph_mode is not None:
            graph = json.loads(row.get('graph', '[]') or '[]')
            out_kwargs['edge_index'], out_kwargs['edge_type'] = build_edges(graph, self.graph_mode, self.rel_to_id)
        return arr, out_kwargs


def _collate_batch_helper(examples, pad_token_id, max_length, return_mask=False):
    """Pad variable-length id lists to max_length (upstream helper, kept for the pretrain path)."""
    result = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    mask_ = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    for i, example in enumerate(examples):
        curr_len = min(len(example), max_length)
        result[i][:curr_len] = example[:curr_len]
        mask_[i][:curr_len] = [1] * curr_len
    if return_mask:
        return result, mask_
    return result


def collate_with_adj(batch):
    """
    Stack a batch; merge per-sample graphs into one disjoint graph.

    Sample b's node ids are shifted by b * seq_len, so node id = b * seq_len + position
    (the flattened [B * L] index used by GraphEncoder). Example (seq_len 20): sample 3, position 6
    -> node 66. Microbatch slicing (train_util.slice_microbatch) relies on this numbering.
    """
    from torch.utils.data._utils.collate import default_collate

    has_graph = 'edge_index' in batch[0][1]
    edge_indices, edge_types = [], []
    if has_graph:
        seq_len = batch[0][0].shape[0]
        for b, (_, out_kwargs) in enumerate(batch):
            ei = out_kwargs.pop('edge_index')
            et = out_kwargs.pop('edge_type')
            if ei.shape[1] > 0:
                edge_indices.append(ei + b * seq_len)
                edge_types.append(et)

    collated = default_collate(batch)
    if has_graph:
        if edge_indices:
            collated[1]['edge_index'] = torch.cat(edge_indices, dim=1)
            collated[1]['edge_type'] = torch.cat(edge_types, dim=0)
        else:
            collated[1]['edge_index'] = torch.empty((2, 0), dtype=torch.long)
            collated[1]['edge_type'] = torch.empty((0,), dtype=torch.long)
    return collated
