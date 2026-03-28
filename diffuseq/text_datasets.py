"""
text_datasets.py — Data loading pipeline for DiffuSeq seq2seq training.

Pipeline overview (for a seq2seq dataset like CommonsenseConversation):

  1. get_corpus / get_corpus_pretrain
       → reads raw .jsonl files ({"src": ..., "trg": ...}) into lists.

  2. helper_tokenize / helper_tokenize_pretrain
       → tokenizes src and trg using a myTokenizer,
       → concatenates them as [src_tokens | SEP | trg_tokens] (padded to seq_len),
       → builds an input_mask: 0 = source position, 1 = target position.
         (The model only needs to generate the target; source is conditioned on.)

  3. TextDataset (torch.utils.data.Dataset)
       → converts token IDs to continuous embedding vectors using model_emb,
       → returns (embedding_tensor, {input_ids, input_mask}) pairs.

  4. load_data_text
       → wraps TextDataset in a DataLoader with DistributedSampler,
       → returns an infinite generator (or a single-pass iterator for inference).

Key design choice — embedding as diffusion target:
  Token IDs (discrete) are mapped to continuous embedding vectors before the
  diffusion forward process. The model learns to denoise in embedding space and
  the nearest-neighbour rounding step (rounding.py) maps the denoised vectors
  back to discrete tokens at inference time.
"""

import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler  # evenly splits data across GPUs
from itertools import chain   # used in pretrain: concatenate all token sequences before chunking
import glob
import torch
import json
import psutil         # for RAM usage monitoring
import datasets
from datasets import Dataset as Dataset2  # HuggingFace datasets Arrow-backed Dataset


def load_data_text(
    batch_size,
    seq_len,
    deterministic=False,
    data_args=None,
    model_emb=None,
    split='train',
    loaded_vocab=None,
    loop=True,
):
    """
    Build a data generator (or iterator) for a seq2seq text dataset.

    Args:
        batch_size (int):    number of samples per batch.
        seq_len (int):       maximum total sequence length (src + SEP + trg, padded).
        deterministic (bool):if True, do not shuffle (useful for validation/test).
        data_args:           argparse.Namespace with dataset, data_dir, notes, etc.
        model_emb:           nn.Embedding used to convert token IDs → float vectors.
        split (str):         'train', 'valid', or 'test'.
        loaded_vocab:        myTokenizer instance.
        loop (bool):         if True, return an infinite generator; else a single-pass iter.

    Returns:
        generator or iterator: yields (batch_tensor, cond_dict) tuples where:
            batch_tensor: [B, seq_len, hidden_dim] float embeddings.
            cond_dict:    {'input_ids': [B, seq_len], 'input_mask': [B, seq_len]}
    """
    print('#' * 30, '\nLoading text data...')

    # Route to pretrain vs fine-tune data loading based on the 'notes' flag
    if "pretrain" in data_args.notes:
        print("#### Load Pretrain Data, fold=", data_args.data_split_num)
        training_data = get_corpus_pretrain(
            data_args, seq_len, split=split,
            loaded_vocab=loaded_vocab, split_num=data_args.data_split_num
        )
    else:
        # Standard seq2seq fine-tuning (e.g., CommonsenseConversation)
        training_data = get_corpus(data_args, seq_len, split=split, loaded_vocab=loaded_vocab)

    # Wrap the tokenized HuggingFace dataset in a PyTorch Dataset that embeds tokens
    dataset = TextDataset(
        training_data,
        data_args,
        model_emb=model_emb
    )

    if split != 'test':
        # DistributedSampler: automatically partitions the dataset across GPUs.
        # Each GPU process sees a disjoint subset of the data in each epoch.
        sampler = DistributedSampler(dataset)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,         # determinism is controlled by the sampler, not shuffle
            num_workers=4,           # parallel data prefetching
        )
    else:
        # Test split: no distributed sampler (single-GPU inference is typical)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=not deterministic,  # shuffle test set if non-deterministic
            num_workers=4,
        )

    if loop:
        # Training: wrap in an infinite generator so next(data) never raises StopIteration
        return infinite_loader(data_loader)
    else:
        # Inference: single pass through the dataset
        return iter(data_loader)


def infinite_loader(data_loader):
    """
    Wrap a DataLoader in an infinite generator that restarts at the end of each epoch.

    Used for training so that the main loop can simply call next(data) indefinitely
    without worrying about StopIteration.

    Args:
        data_loader (DataLoader): any PyTorch DataLoader.

    Yields:
        Batches from the DataLoader, cycling forever.
    """
    while True:
        yield from data_loader


def helper_tokenize(sentence_lst, vocab_dict, seq_len):
    """
    Tokenize, merge, mask, and pad a seq2seq dataset for fine-tuning.

    Produces a HuggingFace DatasetDict with 'train' split containing:
        input_ids  [seq_len]: [src_tokens | SEP | trg_tokens | PAD … PAD]
        input_mask [seq_len]: [0 … 0 | 1 … 1 | 1 … 1]
                                source positions = 0, target positions = 1

    The mask distinguishes tokens the model should condition on (source, mask=0)
    from tokens the model should generate (target, mask=1). During the forward
    diffusion, only target positions are noised.

    Steps:
        1. tokenize_function:  encode src and trg separately → input_id_x, input_id_y.
        2. merge_and_mask:     concatenate as [src | SEP | trg], trim if too long,
                               build the source/target mask.
        3. pad_function:       pad all sequences to exactly seq_len.

    Args:
        sentence_lst (dict):  {'src': list[str], 'trg': list[str]}
        vocab_dict (myTokenizer): tokenizer for encoding strings → IDs.
        seq_len (int):        target sequence length after padding.

    Returns:
        datasets.DatasetDict: {'train': HuggingFace Dataset with input_ids and input_mask}
    """
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    raw_datasets = Dataset2.from_dict(sentence_lst)
    print(raw_datasets)
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    def tokenize_function(examples):
        """Encode source and target texts into token ID lists."""
        input_id_x = vocab_dict.encode_token(examples['src'])
        input_id_y = vocab_dict.encode_token(examples['trg'])
        result_dict = {'input_id_x': input_id_x, 'input_id_y': input_id_y}
        return result_dict

    # Batch tokenization with 4 parallel workers
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=['src', 'trg'],
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )
    print('### tokenized_datasets', tokenized_datasets)
    print('### tokenized_datasets...example', tokenized_datasets['input_id_x'][0])
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    def merge_and_mask(group_lst):
        """
        Concatenate source and target token sequences and build the source/target mask.

        Format: [src_tokens… | SEP | trg_tokens…]
        Mask:   [  0    …    |  0  |  1    …   ]   (0 = source, 1 = target)

        Trimming strategy: if src + trg > seq_len - 3, alternately trim the longer
        sequence one token at a time until they fit.
        """
        lst = []    # list of merged token ID sequences
        mask = []   # list of corresponding source masks
        for i in range(len(group_lst['input_id_x'])):
            end_token = group_lst['input_id_x'][i][-1]   # [SEP] / [END] token
            src = group_lst['input_id_x'][i][:-1]         # remove trailing end token
            trg = group_lst['input_id_y'][i][:-1]

            # Trim until concatenated length fits in seq_len - 3 (space for end tokens + SEP)
            while len(src) + len(trg) > seq_len - 3:
                if len(src) > len(trg):
                    src.pop()
                elif len(src) < len(trg):
                    trg.pop()
                else:
                    src.pop()
                    trg.pop()

            # Re-append end tokens and join with SEP
            src.append(end_token)
            trg.append(end_token)

            # Merged sequence: [src... | SEP | trg...]
            lst.append(src + [vocab_dict.sep_token_id] + trg)
            # Mask: source positions = 0 (conditioned on), target positions not included here
            # (will be padded with 1 in pad_function)
            mask.append([0] * (len(src) + 1))  # +1 for the SEP token

        group_lst['input_ids'] = lst
        group_lst['input_mask'] = mask
        return group_lst

    tokenized_datasets = tokenized_datasets.map(
        merge_and_mask,
        batched=True,
        num_proc=1,   # must be 1: the batched operation has inter-sample dependencies
        desc="merge and mask",
    )

    def pad_function(group_lst):
        """
        Pad input_ids to seq_len with pad_token_id.
        Pad input_mask to seq_len with 1 (target position = noised in diffusion).
        """
        max_length = seq_len
        group_lst['input_ids'] = _collate_batch_helper(
            group_lst['input_ids'], vocab_dict.pad_token_id, max_length
        )
        group_lst['input_mask'] = _collate_batch_helper(
            group_lst['input_mask'], 1, max_length  # pad mask with 1 (= target / noised)
        )
        return group_lst

    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    lm_datasets = tokenized_datasets.map(
        pad_function,
        batched=True,
        num_proc=1,
        desc="padding",
    )

    print(lm_datasets, 'padded dataset')
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    raw_datasets = datasets.DatasetDict()
    raw_datasets['train'] = lm_datasets
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    return raw_datasets


def helper_tokenize_pretrain(sentence_lst, vocab_dict, seq_len, mask_ratio=0.5):
    """
    Tokenize and prepare raw text for pre-training (language model style).

    Unlike fine-tuning, there is no explicit src/trg split. Instead:
      - All texts are concatenated into one long sequence and chunked into seq_len blocks.
      - A random mask of length in [0, mask_ratio * seq_len] is applied to each block:
        the first `mask_len` positions are unmasked (source, mask=0),
        the remaining positions are masked (target, mask=1).

    Args:
        sentence_lst (dict):  {'text': list[str]} raw documents.
        vocab_dict (myTokenizer): tokenizer.
        seq_len (int):        chunk size.
        mask_ratio (float):   upper bound on the fraction of each chunk used as source.

    Returns:
        datasets.DatasetDict: {'train': Dataset with input_ids and input_mask}
    """
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    raw_datasets = Dataset2.from_dict(sentence_lst)
    print(raw_datasets)
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    def tokenize_function(examples):
        """Encode raw text into token ID lists."""
        input_id = vocab_dict.encode_token(examples['text'])
        result_dict = {'input_ids': input_id}
        return result_dict

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=['text'],
        keep_in_memory=True,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )
    print('### tokenized_datasets', tokenized_datasets)
    print('### tokenized_datasets...example', tokenized_datasets['input_ids'][0])
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    block_size = seq_len

    def group_texts(examples):
        """
        Concatenate all token sequences, cut into seq_len blocks, and assign random masks.

        For each block, a random number of prefix tokens are unmasked (mask=0),
        and the rest are masked (mask=1, i.e., treated as the generation target).
        """
        # Flatten all token lists into one long sequence per column
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # Trim to the largest multiple of block_size
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Cut into seq_len chunks
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        # Random mask: each block gets a random prefix length in [0, mask_ratio * block_size)
        random_mask_len = [
            np.random.randint(mask_ratio * block_size) for _ in range(len(result["input_ids"]))
        ]
        # First l tokens: mask=0 (source), remaining: mask=1 (generation target)
        result["input_mask"] = [[0] * l + [1] * (block_size - l) for l in random_mask_len]
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=4,
        keep_in_memory=True,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    print(lm_datasets, 'padded dataset')
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    raw_datasets = datasets.DatasetDict()
    raw_datasets['train'] = lm_datasets
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    return raw_datasets


def get_corpus(data_args, seq_len, split='train', loaded_vocab=None):
    """
    Load a seq2seq dataset from a .jsonl file and tokenize it for fine-tuning.

    Expected .jsonl format (one JSON object per line):
        {"src": "input sentence", "trg": "target sentence"}

    Args:
        data_args: argparse.Namespace with data_dir and dataset name.
        seq_len (int): maximum sequence length.
        split (str): 'train', 'valid', or 'test'.
        loaded_vocab (myTokenizer): the tokenizer.

    Returns:
        datasets.DatasetDict: tokenized and padded dataset.
    """
    print('#' * 30, '\nLoading dataset {} from {}...'.format(data_args.dataset, data_args.data_dir))

    sentence_lst = {'src': [], 'trg': []}

    # Map split name → file path
    if split == 'train':
        print('### Loading form the TRAIN set...')
        path = f'{data_args.data_dir}/train.jsonl'
    elif split == 'valid':
        print('### Loading form the VALID set...')
        path = f'{data_args.data_dir}/valid.jsonl'
    elif split == 'test':
        print('### Loading form the TEST set...')
        path = f'{data_args.data_dir}/test.jsonl'
    else:
        assert False, "invalid split for dataset"

    with open(path, 'r') as f_reader:
        for row in f_reader:
            sentence_lst['src'].append(json.loads(row)['src'].strip())
            sentence_lst['trg'].append(json.loads(row)['trg'].strip())

    print('### Data samples...\n', sentence_lst['src'][:2], sentence_lst['trg'][:2])

    vocab_dict = loaded_vocab
    train_dataset = helper_tokenize(sentence_lst, vocab_dict, seq_len)
    return train_dataset


def get_corpus_pretrain(data_args, seq_len, split='train', loaded_vocab=None, split_num=0):
    """
    Load a raw text corpus and prepare it for language model pre-training.

    Expected .jsonl format:
        {"text": "raw document text"}

    The corpus is sorted by filename and the file at index split_num is loaded.
    The second half of the file is used (first half discarded), then split
    into train (all but last 5000) and valid (last 5000).

    Args:
        data_args: argparse.Namespace with data_dir.
        seq_len (int): chunk size.
        split (str): 'train' or 'valid'.
        loaded_vocab (myTokenizer): the tokenizer.
        split_num (int): which .jsonl shard to load.

    Returns:
        datasets.DatasetDict: tokenized and chunked dataset.
    """
    print('#' * 30, '\nLoading dataset {} from {}...'.format(data_args.dataset, data_args.data_dir))

    sentence_lst = {'text': []}
    path = sorted(glob.glob(f"{data_args.data_dir}/*jsonl"))[split_num]
    with open(path, 'r') as f_reader:
        for row in f_reader:
            sentence_lst['text'].append(json.loads(row)['text'].strip())

    # Use only the second half of the file
    sentence_lst['text'] = sentence_lst['text'][len(sentence_lst['text']) // 2:]

    if split == 'train':
        print('### Loading the TRAIN set...')
        sentence_lst['text'] = sentence_lst['text'][:-5000]   # all but last 5000
    elif split == 'valid':
        print('### Loading the VALID set...')
        sentence_lst['text'] = sentence_lst['text'][-5000:]   # last 5000 for validation

    print('### Data samples...\n', sentence_lst['text'][:2])

    vocab_dict = loaded_vocab
    train_dataset = helper_tokenize_pretrain(sentence_lst, vocab_dict, seq_len)
    return train_dataset


class TextDataset(Dataset):
    """
    PyTorch Dataset that converts tokenized sequences to embedding-space tensors.

    For each sample:
      1. Retrieve the token ID sequence (input_ids) from the HuggingFace dataset.
      2. Pass it through model_emb (an nn.Embedding) to get continuous float vectors.
      3. Return the embedding tensor and metadata (input_ids, input_mask) for conditioning.

    The embedding conversion happens inside __getitem__ under torch.no_grad() to
    avoid building a computation graph for the frozen embedding lookup.

    Args:
        text_datasets (DatasetDict): {'train': HuggingFace Dataset} with input_ids, input_mask.
        data_args: argparse.Namespace (stored for potential future use).
        model_emb (nn.Embedding): pretrained word embedding matrix.
    """

    def __init__(self, text_datasets, data_args, model_emb=None):
        super().__init__()
        self.text_datasets = text_datasets
        self.length = len(self.text_datasets['train'])
        self.data_args = data_args
        self.model_emb = model_emb

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Fetch and embed a single sample.

        Returns:
            arr (np.ndarray):  [seq_len, hidden_dim] float32 embedding vectors.
            out_kwargs (dict): {
                'input_ids':  [seq_len] int array of token IDs,
                'input_mask': [seq_len] int array (0=source, 1=target/pad)
            }
        """
        with torch.no_grad():
            input_ids = self.text_datasets['train'][idx]['input_ids']
            # Look up the embedding for each token: [seq_len] → [seq_len, hidden_dim]
            hidden_state = self.model_emb(torch.tensor(input_ids))

            # Convert to numpy (float32) for compatibility with DataLoader collation
            arr = np.array(hidden_state, dtype=np.float32)

            out_kwargs = {}
            out_kwargs['input_ids'] = np.array(self.text_datasets['train'][idx]['input_ids'])
            out_kwargs['input_mask'] = np.array(self.text_datasets['train'][idx]['input_mask'])

            return arr, out_kwargs


def _collate_batch_helper(examples, pad_token_id, max_length, return_mask=False):
    """
    Pad a list of variable-length token ID sequences to a fixed length.

    Fills a [len(examples), max_length] tensor with pad_token_id, then copies
    each sequence into the first min(len(seq), max_length) positions.

    Args:
        examples (list[list[int]]): variable-length sequences.
        pad_token_id (int):         value to pad with (e.g., 0 for [PAD], 1 for target mask).
        max_length (int):           target sequence length.
        return_mask (bool):         if True, also return a binary attention mask.

    Returns:
        list[list[int]]: padded sequences as a nested list.
        (optional) list[list[int]]: corresponding attention mask (1 = real, 0 = pad).
    """
    result = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    mask_ = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    for i, example in enumerate(examples):
        curr_len = min(len(example), max_length)
        result[i][:curr_len] = example[:curr_len]
        mask_[i][:curr_len] = [1] * curr_len   # real tokens = 1
    if return_mask:
        return result, mask_
    return result