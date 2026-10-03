# Pipeline Fixes and AMR Input Redesign

- **Type**: plan
- **Status**: current — Phases 0–1 done; Phase 3 data and Phase 4 module built; training runs pending (GPU)
- **Last updated**: 2026-10-03
- **Related**: [PENDING.md](PENDING.md) · [../reports/2026-10-03_code-and-results-review.md](../reports/2026-10-03_code-and-results-review.md) · [../data/data-format.md](../data/data-format.md) · [`.agents/AGENTS.md`](../../.agents/AGENTS.md)

---

## 1. Status / gating

Bug IDs: B1–B10 from the [review](../reports/2026-10-03_code-and-results-review.md#5-bugs-found), B11–B15
from its [addendum](../reports/2026-10-03_code-and-results-review.md#12-addendum--implementation-pass-2026-10-03).

| Gate | Condition | State |
|---|---|---|
| G0 | `thesis_env` complete; `pytest tests/` passes on CPU | met 2026-10-03 (66 tests) |
| G1 | B1–B15 fixed, each with a unit test (§4) | met 2026-10-03 |
| G2 | plain baseline on `v2_plain_en_vi_chunk_1` clearly above 4.2 BLEU (en→vi) with < 5% adjacent duplicate tokens; target set after the first KD run (provisional ≥ 15) | open — needs GPU |
| G3 | text+AMR and plain runs decoded on the same test file (identical row count and hash) | open |

GPU jobs start only after the user confirms (GPU Sharing Rule in [`.agents/AGENTS.md`](../../.agents/AGENTS.md)).

---

## 2. Phases

### Phase 0 — Environment and test harness (done 2026-10-03)

- `thesis_env`: `blobfile wandb sacrebleu nltk pytest` installed; no existing package changed.
- `tests/` (CPU only, offline HF): 66 tests over tokenizer, linearizer, layout, graph shift, collate,
  microbatch slicing, denoise masking, GATv2, prep script, end-to-end loss, checkpoint loading.
- `scripts/eval_bleu.py`: corpus sacreBLEU, chrF, repetition rate, length ratio, MBR over seed files.

<p align="center">· · ·</p>

### Phase 1 — Bug fixes (done 2026-10-03)

| Bug | Fix | Where | Test |
|---|---|---|---|
| B1 | added tokens restricted to `:relation` labels and `-9x` frames, validated; vocabulary built from data (`amr_vocab.json`) | `basic_utils.py`, `scripts/build_amr_vocab.py` | `test_tokenizer.py` |
| B2 | relation digits never stripped; "simple" mode removed (`--drop_sense` drops sense suffixes only) | `diffuseq/amr_linearize.py` | `test_amr_linearize.py` |
| B3 | bracketed linearization | `diffuseq/amr_linearize.py` | `test_amr_linearize.py` |
| B4 | GATv2 module replaces gcn.py, raises without PyG; zero-init residual on graph nodes only; edges re-sliced per microbatch | `diffuseq/graph_encoder.py`, `train_util.slice_microbatch` | `test_graph_encoder.py`, `test_train_util.py` |
| B5 | per-row denoise mask | `gaussian_diffusion.build_denoise_mask` | `test_gaussian_diffusion.py` |
| B6 | `--max_seq_len` defaults to config `seq_len`; recorded in `meta.json`; filter uses the exact merged length | `prepare_docamr_datasets.py` | `test_prepare_docamr.py` |
| B7 | valid/test never filtered; train filter joint across variants | `prepare_docamr_datasets.py` | `test_prepare_docamr.py` |
| B8 | `init_pretrained='bert'` requires `hidden_dim == 768` (raises) | `transformer_model.py`, `load_model_emb` | `test_smoke.py` |
| B9 | references to variables outside the chunk dropped with their relation | `diffuseq/amr_linearize.py` | `test_amr_linearize.py` |
| B10 | TED `<url>` lines dropped before chunking (1,125 train sentences) | `prepare_docamr_datasets.py` | `test_prepare_docamr.py` |
| B11 | target-side graph never stored or read | `text_datasets.py`, `prepare_docamr_datasets.py` | `test_text_datasets.py` |
| B12 | `sample_seq2seq.py` load call fixed; graph kwargs passed | `sample_seq2seq.py` | — (GPU script) |
| B13 | decode output file truncated at start instead of appended to | both samplers | — (GPU script) |
| B14 | microbatch losses weighted by their batch share | `train_util.forward_backward` | — |
| B15 | parser concepts `1/3`, `Auf Wiedersehen` sanitized before PENMAN parsing (4 docs) | `amr_linearize.sanitize_docamr` | `test_amr_linearize.py` |

Also: `paths.py` (single owner of data paths); decoding reloads the run folder's tokenizer verbatim.

<p align="center">· · ·</p>

### Phase 2 — Baseline credibility (open, GPU)

1. **Knowledge distillation:**
   - **what:** an autoregressive en→vi model translates the train sources; its outputs replace the human
     references as training targets;
   - **why:** distilled targets carry one translation style per input, which reduces the multimodality
     behind repeated tokens in non-autoregressive models;
   - **example:** a free human rendering becomes a shorter, literal teacher translation; the diffusion model
     learns one consistent mapping.
   - **Check before use:** the teacher's training data must exclude IWSLT tst2015 (test leakage).
   - **Script (2026-10-03):** `scripts/build_kd_dataset.py` — any Hugging Face encoder-decoder teacher;
     translates the 119,429 unique train sources of `v2_plain_en_vi_chunk_1` once (length-sorted batches,
     resumable JSONL cache), replaces `trg` in the plain and companion variants row by row, re-applies the
     joint length filter, copies valid/test unchanged, writes `<dataset>_kd-<tag>/`. `--check_test` reports
     the teacher's BLEU and exact-match rate on tst2015. Default device CPU. Tests:
     `tests/test_build_kd_dataset.py`.
   - **Open:** teacher choice — (a) a public en→vi model (fast; leakage risk because TED talks appear in
     public corpora) or (b) an autoregressive Transformer trained on the v2 train split (no leakage; one
     extra GPU training run).
2. **Vocabulary:** joint en+vi BPE/WordPiece of 10k–16k entries vs mBERT (119,727 with the AMR tokens).
3. **Budget:** effective batch ≥ 1,024 via microbatch accumulation.
4. **Decoding:** solver steps {10, 20, 50}; MBR over 5–10 seeds (`scripts/eval_bleu.py`).

<p align="center">· · ·</p>

### Phase 3 — Text + AMR input (data built 2026-10-03; training after G2)

- **Datasets:** `v2_plain_en_vi_chunk_1`, `v2_amr_en_vi_chunk_1`, `v2_text_amr_en_vi_chunk_1`
  (`--max_seq_len 256`): 122,125 train rows each (same sentences; 3,064 chunks dropped jointly), 833 valid,
  1,098 test. Format: [data-format.md](../data/data-format.md).
- **Verified:** on 5,000 train + all test rows, 100% of relation positions land on the relation token and
  99.96% of node positions on concept tokens.
- **Open:** AMR-only trimming at load time for over-budget test rows (merge_pair pops from the end of the
  longer side, which is the AMR tail for text+AMR rows but can be the target when the target is longer).

<p align="center">· · ·</p>

### Phase 4 — Graph module (built 2026-10-03; ablation after Phase 3)

GATv2, not GCN: `GCNConv` treats every edge type alike, so `:ARG0` and `:ARG1` would be indistinguishable.

| Component | What it does | Why | Example |
|---|---|---|---|
| Edge types (`--graph_mode edge_attr`) | concept → concept edges; type = relation id × direction, embedded as the GATv2 edge feature | the attention score depends on the relation | `sing → i` with `:ARG0`, reverse `i → sing` with `:ARG0`-reverse |
| Levi graph (`--graph_mode levi`) | relation tokens become nodes: head → label → dep (+ reverses) | the relation enters through its own token embedding; 4 edge types only | `sing → :ARG0 → i` (positions 6 → 7 → 8 in §3.2) |
| Self-loops | each node also attends to itself (GATv2 `add_self_loops`, edge feature = mean) | keeps the node's own state in the update | `i` keeps its embedding plus neighbour messages |
| Pre-norm residual layers | `h = h + GELU(GATv2(LayerNorm(h)))`, 2 layers | stable stacking | — |
| Zero-init output, graph nodes only | `x + out_proj(h)` on positions that appear in an edge; `out_proj` starts at 0 | at initialisation the model equals the no-graph model; noised target positions are never modified | a position without edges passes through unchanged |

The source is clean during diffusion, so the graph output is identical at every denoising step.
**Open:** intra-concept edges from continuation subwords to the first subword (e.g. `inform ##ation`).

<p align="center">· · ·</p>

### Phase 5 — Comparison protocol (open)

Same chunk size, `seq_len`, model size, batch, steps, KD data, decode settings and test file for plain vs
text+AMR vs text+AMR+GATv2 (edge_attr / levi); 3 seeds each; corpus BLEU, chrF, repetition rate, length
ratio.

---

## 3. Index, mask and slice walkthroughs

Each step: input, operation, output on a concrete example. Tests in §4 assert the table values.

### 3.1 Plain layout (real output of `helper_tokenize`, `seq_len 16`)

<pre style="font-size:1rem;line-height:1.5">
pos         0      1      2      3      4      5      6      7      8      9     10     11     12     13     14     15
token   [CLS]  Thank    you      .  [SEP]  [SEP]  [CLS]      C   ##ảm      ơ    ##n    các    bạn      .  [SEP]  [PAD]
id        101  91327  13028    119    102    102    101    140 102539    375  10115  10792  43094    119    102      0
mask        0      0      0      0      0      0      1      1      1      1      1      1      1      1      1      1
</pre>

`text_datasets.merge_pair`:
1. `input_id_x = [CLS] Thank you . [SEP]` (5 ids); `input_id_y = [CLS] C ##ảm ơ ##n các bạn . [SEP]` (9 ids).
2. `src = x[:-1]` (4), `trg = y[:-1]` (8); trim while `len(src) + len(trg) > seq_len - 3`, popping the longer.
3. Re-append `[SEP]` to both; `input_ids = src + [SEP] + trg` → 5 + 1 + 9 = 15 ids; `src_len = 5`,
   `trg_start = 6`.
4. `input_mask = [0] * 6`; `pad_to` fills ids with `[PAD]` and the mask with 1 up to 16.
5. Decode (`split_source_target`): `len_x = 16 - 10 = 6` → source = pos 0–5, target = pos 6–15.

Invariants (asserted): equal lengths; one 0→1 transition in `input_mask`; `input_ids[trg_start] == [CLS]`.

<p align="center">· · ·</p>

### 3.2 Text + AMR layout (`seq_len 20`, real mBERT ids, `:ARG0` added)

`src` string `I sing . [SEP] ( sing :ARG0 i )`, `trg` `Tôi hát .`; `[SEP]` in the string is tokenized as the
special token, so the source block is `[CLS] EN [SEP] AMR [SEP]`.

<pre style="font-size:1rem;line-height:1.5">
pos           0      1      2      3      4      5      6      7      8      9     10     11     12     13     14     15     16     17     18     19
token     [CLS]      I   sing      .  [SEP]      (   sing  :ARG0      i      )  [SEP]  [SEP]  [CLS]      T   ##ôi    hát      .  [SEP]  [PAD]  [PAD]
id          101    146  21253    119    102    113  21253 119547    177    114    102    102    101    157  26596  25573    119    102      0      0
segment       0      0      0      0      0      1      1      1      1      1      1      2      3      3      3      3      3      3      3      3
mask          0      0      0      0      0      0      0      0      0      0      0      0      1      1      1      1      1      1      1      1
</pre>

1. `src ids` = 11 (pos 0–10): EN block 0–4, AMR block 5–10 (its `[SEP]` at 10 ends the source).
2. `merge_pair`: separator at `src_len = 11`; target starts at `trg_start = 12`.
3. `segment` is not materialized; it follows from the first `[SEP]` (EN | AMR) and `input_mask` (source |
   target).
4. Budget: the build drops train chunks with merged length > `--max_seq_len`; valid/test rows are kept and,
   if over budget, trimmed by `merge_pair` (Phase 3 open item).

`sing` has id 21253 at pos 2 (EN) and pos 6 (AMR): concept and word share an embedding.

<p align="center">· · ·</p>

### 3.3 Graph index chain (one edge `sing → :ARG0 → i`)

| Step | Operation | Function | `sing` | `:ARG0` | `i` |
|---|---|---|---|---|---|
| 1 | word index in `lin.words = [(, sing, :ARG0, i, )]` | `linearize_sentences` | 1 | 2 | 3 |
| 2 | char start in `src` = len(`"I sing . [SEP] "`) (15) + offset in the AMR string | `word_char_starts` | 17 | 22 | 28 |
| 3 | token whose offset span starts at that char ([CLS] = 0) | `map_words_to_tokens` | 6 | 7 | 8 |
| 4 | stored as `graph_src` entry `[head, dep, label, label_pos]` | `triples_to_token_graph` | `[6, 8, ":ARG0", 7]` | | |
| 5 | +1 for positions ≥ 1 if a direction token is prepended; keep if all in `1 .. src_len - 2` | `shift_source_graph` | 6 | 7 | 8 |
| 6 | edges (`levi`): 6→7 (type 0), 7→8 (1), 7→6 (2), 8→7 (3) | `build_edges` | | | |
| 7 | collate: + `b * seq_len` for sample `b` (`b = 3`, `seq_len 20`) | `collate_with_adj` | 66 | 67 | 68 |
| 8 | microbatch `[2, 4)`: keep source node in `[40, 80)`, subtract 40 | `slice_microbatch` | 26 | 27 | 28 |
| 9 | check: `26 // 20 = 1` (sample 1 of the microbatch = batch sample 3), `26 % 20 = 6` | | | | |

Step 3 uses character offsets because the BERT pre-tokenizer splits punctuation (`U.S.` → `U . S .`), so
whitespace word indices do not match tokenizer word ids.

<p align="center">· · ·</p>

### 3.4 Microbatch slicing of the condition dict

| Key | Shape (batch) | Slice for microbatch `[i, i+mb)` |
|---|---|---|
| `input_ids`, `input_mask`, `rel_mask` | `[B, L]` | `v[i:i+mb]` |
| `edge_index` | `[2, E]`, node ids in `[0, B*L)` | `keep = (v[0] >= i*L) & (v[0] < min(i+mb, B)*L)`; `v[:, keep] - i*L` |
| `edge_type` | `[E]` | `v[keep]` |

Each microbatch loss is multiplied by `micro_size / B` (B14), so the accumulated gradient is the batch mean.

<p align="center">· · ·</p>

### 3.5 Per-row denoise masking

| Step | Tensor | Shape |
|---|---|---|
| 1 | `mask_rate = sqrt(1 - ᾱ_t) * denoise_rate` | `[B, L]` |
| 2 | `random_mask = bernoulli(mask_rate)` | `[B, L]` |
| 3 | `has_rel = rel_mask.sum(dim=1, keepdim=True) > 0` — per row | `[B, 1]` |
| 4 | `row_mask = where(has_rel, rel_mask * random_mask, random_mask)` | `[B, L]` |
| 5 | `x_t = where(row_mask[..., None] == 0, x_t, mean_embed)` | `[B, L, D]` |
| 6 | `x_t = where(input_mask[..., None] == 0, x_start, x_t)` — source restored | `[B, L, D]` |

Example: `random_mask = [[1,1,1,0],[1,0,1,1]]`, `rel_mask = [[0,0,1,0],[0,0,0,0]]` → `[[0,0,1,0],[1,0,1,1]]`.

---

## 4. Test matrix (pytest, CPU only, `tests/`)

| Target | Test file | Cases |
|---|---|---|
| `myTokenizer`, `validate_added_tokens` | `test_tokenizer.py` | bare words rejected; legacy simple vocab rejected; plain EN/VI tokenization identical to base mBERT (test set + examples); relation / direction tokens single ids; saved run tokenizer reloaded verbatim |
| linearizer, parser | `test_amr_linearize.py` | sentence order; single-sentence file; brackets/leaves/triples; digits and senses; `-9x` kept; quotes stripped; cross-sentence reference dropped; re-entrancy inside a chunk; sanitizer; char starts; offsets → tokens (AMR only, text prefix, literal) |
| `merge_pair`, `pad_to`, `build_rel_mask`, `shift_source_graph`, `split_source_target`, `build_edges`, `collate_with_adj`, `helper_tokenize`, `TextDataset` | `test_text_datasets.py` | §3.1 table; exact fit; trimming (longer side, equal); rel-mask alignment; direction shift; trimmed / `[SEP]` positions dropped; malformed mask rejected; levi / edge_attr / empty edges; collate offsets with an empty sample; end-to-end with a direction token |
| `slice_microbatch` | `test_train_util.py` | §3.3 steps 7–9; first microbatch; partial last microbatch; every edge assigned once; no graph keys |
| `build_denoise_mask`, `q_sample` | `test_gaussian_diffusion.py` | §3.5 example; disabled / missing rel mask; source restored; mean-embed replacement only on target |
| `GraphEncoder` | `test_graph_encoder.py` | zero init = identity; only graph nodes change; edge type changes the output; empty graph; out-of-range node id; heads must divide D |
| prep script | `test_prepare_docamr.py` | alignment + mismatch skip; metadata drop; rows per variant incl. §3.2 graph; merged length = merge_pair; train-only joint filter, test/valid kept |
| model + diffusion | `test_smoke.py` | loss + backward (plain, GATv2); B8 guard; existing checkpoint loads strictly with its saved tokenizer |
| KD data builder | `test_build_kd_dataset.py` | alignment rejects mismatches; only train targets replaced, other fields and valid/test unchanged; cache resume incl. a truncated line; joint length filter + empty output; one output per input; test exact-match report |

Run: `CUDA_VISIBLE_DEVICES="" conda run -n thesis_env pytest tests/ -q`.

---

## 5. Rollback

All changes are on branch `fix/tests-and-pipeline-bugs` (off `thesis-doc-amr`). Existing datasets
(`plain_text_*`) and checkpoints are untouched; new datasets use the `v2_` prefix. Old runs decode with their
saved tokenizer; `diffuseq_env` still imports and runs the data pipeline (Python 3.8 checked).
