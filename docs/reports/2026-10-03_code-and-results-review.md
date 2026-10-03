# Code and Results Review — DiffuSeq + DocAMR for English–Vietnamese Translation

- **Type**: report
- **Date**: 2026-10-03
- **Related**: [plans/PENDING.md](../plans/PENDING.md) · [plans/pipeline-fixes-and-amr-redesign.md](../plans/pipeline-fixes-and-amr-redesign.md) · [`.agents/AGENTS.md`](../../.agents/AGENTS.md)

---

## 1. Executive summary

- **Scope:** every commit on top of upstream DiffuSeq (`3b71ccc`, "update readme file") up to
  `d8760b2` on branch `thesis-doc-amr`, the three model folders under `diffusion_models/`, the two
  decode files under `generation_outputs/`, and the configs of 22 W&B offline runs.
- **Plain-text path:** functionally identical to upstream DiffuSeq. No bug found that explains the
  low scores. The added `scaler.unscale_()` before gradient clipping is a correct fix.
- **Results on disk are genuine but weak:** corpus BLEU 4.2 (en→vi) and 6.4 (vi→en); 24.8% of
  adjacent en→vi output tokens are duplicates (references: 0.2%). An autoregressive Transformer on
  IWSLT'15 en–vi is reported in the high 20s to low 30s.
- **All three runs on disk are plain-text baselines.** No AMR checkpoint, decode output, or AMR
  dataset survives; AMR runs exist only as W&B logs.
- **The AMR datasets replace the English sentence with its AMR** — they do not concatenate text and
  AMR. The model translates from a lossy parse.
- **The AMR path has ten confirmed bugs** (§5). The most severe: the simple-AMR vocabulary changes
  the tokenization of 55.9% of English sentences; simple mode deletes `:ARGn` numbers; the
  linearization drops all brackets; the GCN never ran (`torch_geometric` missing, silent fallback to
  `nn.Linear`).
- **No AMR-vs-text comparison is fair yet:** the runs differ in chunk size, `seq_len`, hidden size,
  batch size, direction mix and tokenizer, and the test sets differ in row count (1,061 / 1,097 /
  1,096) because the prep script length-filters `test.jsonl`.
- **Environment:** `thesis_env` runs the repo's classes and loads the existing checkpoints on CPU,
  tokenizes identically to `diffuseq_env`, and has `torch_geometric`; it lacks `blobfile`,
  `wandb`, `sacrebleu`, `nltk`, `pytest` (§9).

---

## 2. Method

1. Diff of every changed file against upstream `3b71ccc` (20 files, +3,407 / −505 lines), comments
   stripped to isolate functional changes.
2. Corpus BLEU of both decode files against the raw references
   ([`scripts/_oneshot/review_corpus_bleu.py`](../../scripts/_oneshot/review_corpus_bleu.py)).
3. Tokenizer audits with the run's saved tokenizer and with base `bert-base-multilingual-cased`.
4. AMR datasets regenerated into a scratch directory with `prepare_docamr_datasets.py
   --chunk_size 1 --simple {False,True}` and inspected (lengths, variable leaks, edge alignment).
5. W&B run configs and final metrics read from the offline `.wandb` protobuf records.
6. Every Python command ran with `CUDA_VISIBLE_DEVICES=""`; the GPU was not touched.

---

## 3. Runs on disk and their results

| Run folder (prefix `plain_text_…_h256_lr0.0001_t2000_sqrt_lossaware_seed102_…`) | Data | Key settings | Outcome |
|---|---|---|---|
| `en_vi_chunk_1 … 20260514-21:13:10` | plain EN→VI, 122,040 train rows | `use_plm_init no`, clip off, 50k steps | ckpts 5k–50k, decoded at 50k |
| `vi_en_chunk_1 … 20260518-08:17:28` | plain VI→EN | `use_plm_init no`, clip 1.0, 45k steps | ckpts 5k–45k, decoded at 45k |
| `en_vi_chunk_1 … 20260519-22:17:30` | plain EN→VI | `use_plm_init bert`, clip 1.0 | crashed before step 0 (bug B8) |

Shared settings: `seq_len 128`, `hidden_dim 256`, batch 128, microbatch 8, `diffusion_steps 2000`,
`sqrt` schedule, `lossaware` sampler, fp16, `denoise_rate 0.5`, vocab = mBERT + 5,436 AMR tokens
(124,983). Decoding: DPM-Solver++ 10 steps, seed 110, greedy argmax, one sample, no MBR.

<p align="center">· · ·</p>

### 3.1 Scores

| Run | Corpus BLEU | 1/2/3/4-gram precision | BP | Hyp / ref length | Adjacent duplicate tokens |
|---|---|---|---|---|---|
| en→vi @50k | **4.22** | 34.9 / 9.7 / 2.1 / 0.5 | 0.971 | 21,770 / 22,410 | 24.8% (ref 0.2%) |
| vi→en @45k | **6.43** | 46.4 / 14.1 / 4.3 / 1.3 | 0.822 | 15,162 / 18,133 | 6.8% (ref 0.1%) |

- **Validity:** decode rows follow `test.jsonl` order; scoring against the mBERT-decoded references
  instead of the raw ones changes BLEU by < 0.1.
- **Failure pattern:** the unigram precision shows correct word choice; the collapse in higher-order
  precision and the repeats (`một một`, `khi khi khi`) are the standard non-autoregressive
  multimodality failure. vi→en outputs are too short (BP 0.82).
- **Metric caveat:** `scripts/eval_seq2seq.py` averages smoothed sentence-level BLEU (NLTK
  method4), which is higher than corpus BLEU and is not the MT-standard metric.
- **Tokenizer caveat:** the BLEU script re-implements sacreBLEU's 13a tokenizer (sacreBLEU is not
  installed); small deviations from sacreBLEU are expected.

<p align="center">· · ·</p>

### 3.2 Training curves

Loss falls smoothly (en→vi: 1.22 at step 0 → 0.025 at 50k; nll of the predicted x0 20.8 → 1.22).
The learning rate decays linearly to 0 at `learning_steps`, so the late plateau reflects the
schedule, not convergence. No divergence and no NaN.

---

## 4. Plain-text path versus upstream

| File | Functional change | Verdict |
|---|---|---|
| `train.py`, `diffuseq/step_sample.py`, `diffuseq/rounding.py` | comments only | unchanged behaviour |
| `train_util.py` | `scaler.unscale_()` before clipping; NaN guard that raises | correct |
| `diffuseq/gaussian_diffusion.py` | optional `rel_mask` branch in `q_sample` | inert when `mask_docamr_rel False`; see B5 |
| `diffuseq/transformer_model.py` | optional GCN before the encoder; vocab resize for `use_plm_init bert` | inert when `enable_gcn False`; see B4, B8 |
| `diffuseq/text_datasets.py` | direction tokens, `rel_mask`, adjacency columns, `collate_with_adj` | plain layout identical to upstream |
| `sample_seq2seq_dpmSolver.py` | `filter_direction`, edge kwargs, `ensure_ascii=False` | unchanged decoding math |
| `basic_utils.py` | AMR tokens added to the tokenizer; `use_plm_init bert` embedding seeding | see B1, B8 |

The sequence layout produced by `helper_tokenize` is the upstream one. Worked example (real output,
`seq_len 16`, source "Thank you .", target "Cảm ơn các bạn ."):

<pre style="font-size:1rem;line-height:1.5">
pos         0      1      2      3      4      5      6      7      8      9     10     11     12     13     14     15
token   [CLS]  Thank    you      .  [SEP]  [SEP]  [CLS]      C   ##ảm      ơ    ##n    các    bạn      .  [SEP]  [PAD]
id        101  91327  13028    119    102    102    101    140 102539    375  10115  10792  43094    119    102      0
mask        0      0      0      0      0      0      1      1      1      1      1      1      1      1      1      1
decode: len_x = seq_len - sum(mask) = 6 → source = pos 0–5, generated target = pos 6–15
</pre>

The double `[SEP]` and the target-side `[CLS]` are upstream behaviour. `[PAD]` positions carry
`mask = 1`: the model generates padding, which is how DiffuSeq decides output length.

---

## 5. Bugs found

| ID | Severity | Location | Effect | Affected runs |
|---|---|---|---|---|
| B1 | critical | [basic_utils.py:70-78](../../basic_utils.py) + `doc_amrs_token_simple.json` | 2,422 bare English words added as tokens split ordinary words; 55.9% of EN and 2.6% of VI sentences tokenize differently from base mBERT | every `use_simple_amr True` run (e.g. `src_docamr_en_trg_doc_vi_chunk_5_simple`, final nll 5.79) |
| B2 | critical | [prepare_docamr_datasets.py:299](../../prepare_docamr_datasets.py) | simple mode strips digits: `:ARG0`/`:ARG1` → `:ARG`, `:op1`/`:op2` → `:op`; agent/patient distinction lost | every simple run |
| B3 | critical | [prepare_docamr_datasets.py:289](../../prepare_docamr_datasets.py) | linearization drops `(` `)`; subtree boundaries unrecoverable without the graph module | every AMR run |
| B4 | critical | `diffuseq/gcn.py:4-21` (file removed 2026-10-03, replaced by `diffuseq/graph_encoder.py`), [basic_utils.py:328](../../basic_utils.py), [text_datasets.py:703](../../diffuseq/text_datasets.py), [train_util.py:386](../../train_util.py) | (a) `torch_geometric` absent in `diffuseq_env` → silent fallback to `nn.Linear` that ignores edges; (b) relation file path `datasets/docAMR/doc_amrs_token.json` does not exist → every edge gets relation id 0; (c) microbatch slicing `v[i:i+microbatch]` cuts `edge_index [2, E]` along dim 0 and keeps batch-level node offsets → crash or wrong messages once PyG is installed and `microbatch < batch_size`; (d) GCN output replaces `x` with no residual | `src_docamr_en_trg_doc_vi_chunk_3` (`enable_gcn True`) |
| B5 | major | [gaussian_diffusion.py:274](../../diffuseq/gaussian_diffusion.py) | `rel_mask.any()` is evaluated per microbatch: AMR→text rows lose denoise masking whenever a text→AMR row shares the microbatch | bidirectional runs with `mask_docamr_rel True` |
| B6 | major | [prepare_docamr_datasets.py:876](../../prepare_docamr_datasets.py) vs `--seq_len` | prep filters at 256 tokens, training trims at `seq_len`; with `seq_len 128`, 14.4% of full-AMR chunk-1 pairs are trimmed (mostly the AMR source) | `bidirection_docamr_en_doc_vi_chunk_1` |
| B7 | major | `write_jsonl` / `write_plain_jsonl` | `test.jsonl` is length-filtered like train: the longest test sentences are dropped and test sets differ per variant (1,061 on disk; 1,097 plain/full-AMR; 1,096 simple-AMR from the current script) | all comparisons |
| B8 | minor | [transformer_model.py:175](../../diffuseq/transformer_model.py), `load_model_emb` | `use_plm_init bert` with `hidden_dim ≠ 768` ties 768-d mBERT embeddings to a 256-d head; `load_model_emb` keeps the first 256 of 768 mBERT dims | the 2026-05-19 run (crash) |
| B9 | minor | `build_pairs` chunk normalization | 587 chunk-1 rows (0.5%) keep raw variable names (`s1.p`) from cross-sentence references; renaming `sN → s1` can collide with genuine `s1.*` references | all chunked AMR runs |
| B10 | minor | data | about 1.1k TED `<url>` metadata lines remain as training pairs (1,324 rows with `src == trg`); 5,436 unused AMR tokens in the plain baselines' vocabulary | plain runs |

<p align="center">· · ·</p>

### 5.1 B1 — added-token substring matching

HF added tokens are matched as substrings before WordPiece runs. Examples with the simple vocabulary:

| Input | Base mBERT | With `doc_amrs_token_simple.json` |
|---|---|---|
| `rachel` (in a URL) | `ra ##chel` | `r ache l` |
| `information` (AMR concept) | `information` | `inform ati ##on` |
| `collapse` (AMR concept) | `collapse` | `col lapse` |

The full vocabulary (`doc_amrs_token.json`) contains only `:relation` labels and sense-tagged
frames (`abandon-01`); it changes 0.00% of plain sentences (7 of 122,040 train rows, e.g. `AK-47`).

<p align="center">· · ·</p>

### 5.2 B4 — what the "GCN" run computed

With `torch_geometric` missing, `GCNConv` resolves to the fallback class whose `forward(x,
edge_index)` returns `self.lin(x)`. The `chunk_3` run therefore trained a per-position linear layer
in front of the encoder; the adjacency data had no effect. Edge alignment itself is mostly sound:
91.5% of edge endpoints land on a concept's first subword; 8.5% land on a quote, hyphen or relation
token (quoted literals such as `"U.S."` split into several pre-tokenizer words).

<p align="center">· · ·</p>

### 5.3 Parser output, not a bug

Sentences starting with "And …" parse as `(a / and :op2 …)` with no `:op1` (the parser leaves the
first conjunct implicit). This comes from the raw `.out` files, not from the prep script.

---

## 6. Experimental-design issues

1. **AMR replaces the sentence.** `src_docamr_en_trg_doc_vi*` uses the linearized AMR as the only
   source. AMR omits tense, number, articles and word order, and carries parser errors, so AMR-only
   translation is expected to score below text-only translation. AMR-augmented NMT work feeds the
   AMR alongside the sentence.
2. **No controlled comparison.** The only AMR run at the baseline's size (chunk 1, `seq_len 128`,
   h256, batch 128, 50k steps) is bidirectional, so half its data is VI→AMR parsing.
3. **Document chunks.** Most AMR runs use chunks of 3–5 sentences with `seq_len` 256–512 and batch
   64–128; non-autoregressive diffusion quality degrades with output length.
4. **Baseline limits (setup, not code):**
   - **Compute:** 50k steps × batch 128 = 6.4M samples, about 16× fewer than upstream QQP (50k ×
     2048).
   - **Vocabulary:** 124,983 randomly initialized 256-d embeddings; joint en+vi BPE vocabularies of
     10k–16k are the norm for IWSLT-size data.
   - **Decoding:** 10 solver steps, one sample, no MBR.
   - **No knowledge distillation:** non-autoregressive and diffusion MT is usually trained on
     outputs of an autoregressive teacher; without it, token repetition of the observed kind is
     expected.
   - **Padding share:** with `seq_len 128` and ~48 real tokens per pair (src 21.7 + trg 25.4 on
     average), ~62% of the MSE positions are padding and another ~18% are the copied source.

---

## 7. Earlier AMR runs (W&B only)

| Date | Dataset | seq_len | hidden | batch / micro | steps | GCN | simple | rel-mask | Final nll (train / eval) |
|---|---|---|---|---|---|---|---|---|---|
| 2026-04-19 | `src_doc_vi_trg_docamr_en_chunk_5` | 256 | 128 | 425 / 1 | 12,260 of 20k | — | — | yes | 1.19 / 1.51 |
| 2026-04-21 | `src_doc_vi_trg_docamr_en_chunk_5` | 256 | 256 | 128 / 4 | 20k | — | — | yes | 1.60 / 2.24 |
| 2026-04-26 | `src_docamr_en_trg_doc_vi_chunk_5_simple` | 256 | 256 | 128 / 4 | 20k | no | yes | no | 5.79 / 5.96 |
| 2026-04-29 | `src_docamr_en_trg_doc_vi_chunk_3` | 256 | 256 | 128 / 4 | 50k | yes (fallback) | no | no | 2.83 / 2.90 |
| 2026-05-01 | `bidirection_docamr_en_doc_vi_chunk_4` | 256 | 128 | 256 / 4 | 30k | no | no | yes | 2.80 / 2.90 |
| 2026-05-03 | `bidirection_docamr_en_doc_vi_chunk_4` | 512 | 128 | 64 / 2 | 50k | no | no | yes | 1.16 / 1.24 |
| 2026-05-09 | `bidirection_docamr_en_doc_vi_chunk_4` | 512 | 128 | 64 / 2 | 47,220 of 50k | no | no | yes | 1.21 / 1.40 |
| 2026-05-11 | `bidirection_docamr_en_doc_vi_chunk_4` | 512 | 128 | 64 / 2 | 50k | no | no | yes | 0.78 / 0.83 |
| 2026-05-14 | `bidirection_docamr_en_doc_vi_chunk_1` | 128 | 256 | 128 / 8 | 50k | no | no | yes | 0.89 / 0.97 |

"nll" is the token cross-entropy of the predicted x0 averaged over all timesteps and all target
positions (padding included); it is comparable only within one dataset and `seq_len`. The plain
en→vi baseline ends at 1.22. Bidirectional values mix two tasks. No decode output exists for any
AMR run.

---

## 8. Design questions

### 8.1 Relation tokens, inverse roles, and `have-…-91` frames

Counts over all 1,144 DocAMR `.out` files: 395 distinct relation labels, 1,806,756 relation
occurrences.

- **Inverse roles are relations.** `:ARG1-of`, `:ARG0-of`, `:part-of`, `:manner-of`, … — 55 labels,
  171,586 occurrences. The `-of` suffix reverses the edge direction (`(c / cell :ARG1-of (g /
  grow-01))` = the cell is ARG1 of grow-01). They must be atomic tokens like any relation.
- **Coverage gap:** 21 labels in the data (564 occurrences) are missing from `doc_amrs_token.json`
  — e.g. `:op2-of`, `:same-as-of`, `:mod-of`, `:domain-of`, `:name-of`, `:value-of`,
  `:prep-with-of`. The relation inventory must be derived from the train split, not a fixed list.
- **`have-org-role-91`, `have-rel-role-91`, `have-degree-91` are concepts**, not relations (special
  `-91` frames; top counts: `have-degree-91` 12,410, `include-91` 5,704, `have-rel-role-91` 5,582).
  The token file holds 33 such frames.

| Vocabulary option | Effect on frames | AMR tokens p50 (parens) |
|---|---|---|
| relations only | `have-org-role-91` → `have - org - role - 91` (WordPiece) | 44 |
| relations + 33 `-9x` frames | `-9x` frames atomic, other predicates WordPiece | 42 |
| relations + all 5,059 frames | every frame atomic; 1,913 of 4,020 frames seen in a 30k sample occur < 5 times → weakly trained rows | 35 |

Recommendation: relations (all labels from train) + `-9x` frames as atomic tokens; ordinary
predicates through WordPiece with the sense suffix dropped (`grow-01` → `grow`). With text+AMR
concatenation, the concept `grow` then shares its token with the English word in the same input,
which links the two segments. Guard: a unit test asserts that the extended tokenizer leaves the
plain EN and VI corpora tokenized exactly as base mBERT does (the B1 audit, as a test).

<p align="center">· · ·</p>

### 8.2 Is 256 tokens enough for text + AMR concatenation?

Layout measured: `[CLS] EN [SEP] AMR [SEP]` + `[SEP]` + `[CLS] VI [SEP]`, 30k train sentences, mBERT
WordPiece ([`scripts/_oneshot/review_amr_length_study.py`](../../scripts/_oneshot/review_amr_length_study.py)).
EN text alone: p50 19 / p95 52 / p99 75 tokens; VI: 23 / 62 / 90. Some lines hold several
sentences (max 729 EN tokens).

| Linearization | Vocabulary | AMR p50 / p95 | Total p50 / p95 / p99 | > 192 | > 256 | > 320 | > 384 |
|---|---|---|---|---|---|---|---|
| flat (existing) | relations + all frames | 24 / 72 | 68 / 184 / 265 | 4.3% | 1.2% | 0.3% | 0.1% |
| parens | relations only | 44 / 130 | 87 / 241 / 347 | 10.9% | 4.0% | 1.5% | 0.6% |
| parens | relations + `-9x` | 42 / 126 | 86 / 238 / 342 | 10.4% | 3.7% | 1.4% | 0.5% |
| parens | relations + all frames | 35 / 105 | 79 / 217 / 311 | 7.7% | 2.5% | 0.9% | 0.3% |
| parens, sense dropped | relations + `-9x` | 36 / 108 | 80 / 219 / 315 | 8.0% | 2.6% | 0.9% | 0.3% |

- **256 covers ~97.4%** of sentences with the recommended setting (parens, sense dropped,
  relations + `-9x`); 320 covers ~99.1%.
- **Over-budget policy:** never trim the VI target or the EN text; trim the AMR segment only (drop
  deepest subtrees first, keeping brackets balanced). Train: drop rows still over budget. Test:
  never drop rows.
- **Cost:** attention is O(L²): 320 costs ~1.56× the attention of 256. DiffuSeq also spends MSE on
  every position, so a longer `seq_len` adds padding to the loss.

<p align="center">· · ·</p>

### 8.3 GCN or GAT?

| Option | Uses relation labels | Fit for AMR in this model |
|---|---|---|
| GCN (`GCNConv`) | no — `:ARG0` and `:ARG1` edges are identical | poor; fixed degree normalization |
| R-GCN | yes — one weight matrix per relation | 395 relations need basis decomposition; rare relations undertrained |
| GATv2 + edge features | yes — relation embedding enters the attention score (`edge_dim`) | good; same design as the edge-conditioned GAT in the STKGQA project |
| GAT on a Levi graph | yes — relation tokens become nodes between concepts | good; maps onto the linearized sequence, where relation tokens already occupy positions |

Recommendation: first a no-graph run with the bracketed linearization (the Transformer sees the
structure through the brackets); then GATv2, 2 layers, residual + LayerNorm, applied only to AMR
positions of the source segment, with edges `parent → relation-token → child` (Levi) or
`parent → child` with the relation as edge feature, both directions plus self-loops, and
intra-concept edges from continuation subwords to the first subword. Literature on AMR-to-text
generation reports that linearized AMR with pretrained Transformers is competitive with or better
than GNN encoders; a graph module is an ablation, not a prerequisite.

---

## 9. Conda environment survey

| Env | Python | torch | transformers | torch_geometric | Missing for this repo |
|---|---|---|---|---|---|
| `diffuseq_env` | 3.8 | 1.13.1 (cu117) | 4.22.2 | no | torch_geometric, sacrebleu, pytest, penman |
| `thesis_env` | 3.11 | 2.10.0 (cu128) | 5.3.0 | 2.7.0 | blobfile, wandb, sacrebleu, nltk, pytest |
| `stkgqa` | 3.11 | 2.6.0 (cu124) | 4.45.2 | 2.8.0 | datasets, blobfile, wandb, sacrebleu, … (belongs to STKGQA) |
| `amr_parser_env` | 3.8 | 1.13.1 | 4.46.3 | no | parser env; not for training |
| `nlp`, `nlp_final` | 3.10 | 2.12–2.13 | 5.x | no | most dependencies |

CPU-only checks in `thesis_env`:

| Check | Result |
|---|---|
| `myTokenizer` with AMR tokens | pass |
| `TransformerNetModel` forward | pass |
| `GaussianDiffusion.training_losses` | pass |
| `import train_util` | fail — `blobfile` missing |
| token ids vs `diffuseq_env` (42k strings, MD5 of all ids) | identical |
| strict `load_state_dict` of `ema_0.9999_045000.pt` (trained under torch 1.13) | pass |

Recommendation: `thesis_env`, after installing the missing packages (needs approval, not done):

```bash
conda run -n thesis_env pip install blobfile wandb sacrebleu nltk pytest
```

`diffuseq_env` stays as the environment that reproduces the existing runs.

---

## 10. Reproduce

```bash
# corpus BLEU + repetition of the two decode files (CPU, no dependencies)
python scripts/_oneshot/review_corpus_bleu.py \
  generation_outputs/plain_text_en_vi_*/ema_0.9999_050000.pt.samples/seed110_solverstep10_AMR_TO_TEXT_none.json \
  datasets/docAMR/plain_text_en_vi_chunk_1/test.jsonl
# length study (§8.2)
CUDA_VISIBLE_DEVICES="" conda run -n thesis_env python scripts/_oneshot/review_amr_length_study.py
```

---

## 11. Reference links

- [plans/PENDING.md](../plans/PENDING.md) — open work at a glance.
- [plans/pipeline-fixes-and-amr-redesign.md](../plans/pipeline-fixes-and-amr-redesign.md) — fix
  plan, index walkthroughs, test matrix.
- Upstream DiffuSeq: <https://github.com/Shark-NLP/DiffuSeq> (base commit `3b71ccc`).

---

## 12. Addendum — implementation pass (2026-10-03)

Found while fixing B1–B10 (fix plan: [pipeline-fixes-and-amr-redesign](../plans/pipeline-fixes-and-amr-redesign.md)).

| ID | Severity | Location | Effect |
|---|---|---|---|
| B11 | major | `text_datasets.merge_and_mask` (`adj_trg`) | for TEXT_TO_AMR rows the gold target graph was shifted into the input and handed to the graph module; at decoding time this exposes the structure of the answer |
| B12 | major | `sample_seq2seq.py` | `dist_util.load_state_dict(path, False, "model", …)` does not match the signature `(path, **kwargs)`: the DDPM/DDIM sampler crashes on load (inherited from upstream); graph tensors were never passed to the model |
| B13 | minor | both samplers | output files are opened in append mode: a re-run with the same seed and note doubles the rows |
| B14 | minor | `train_util.forward_backward` | microbatch losses were summed, so the gradient scale (and the effect of `--gradient_clipping`) depended on `batch_size / microbatch`, and a shorter last microbatch weighed as much as a full one |
| B15 | minor | DocAMR parser output | 4 train documents contain concepts PENMAN cannot read (`1/3`, `Auf Wiedersehen`); 11 single-sentence files have no `document` root |

Corrections to earlier numbers:

- **Relation vocabulary:** §8.1 counted 395 relation labels over all splits including `:snt1 … :sntN`
  markers. From the train split, as emitted by the new linearizer and without `:sntN`: **145 labels**
  (55 inverse `-of` roles) + 33 `-9x` frames (`amr_vocab.json`).
- **Official sacreBLEU** (2.6.0) on the on-disk decodes: 4.22 (en→vi, chrF 22.0) and 6.44 (vi→en, chrF
  25.6), matching §3.1.
- **Rebuilt data** (`v2_*_chunk_1`, `--max_seq_len 256`): all 1,125 train documents align; 1,125 metadata
  sentences dropped; 122,125 train rows per variant; test 1,098 rows (unfiltered) vs 1,061 in the on-disk
  `plain_text_*` set.
- **Environment (§9):** `blobfile wandb sacrebleu nltk pytest` installed into `thesis_env`; no existing
  package changed; all CPU checks pass.
- **Length study re-measured with the shipped pipeline** (all 125,189 train sentences, real linearizer and
  tokenizer; [`review_amr_length_study.py`](../../scripts/_oneshot/review_amr_length_study.py)) — supersedes
  the §8.2 table for the shipped layout:

| Layout | AMR tokens p50 / p95 | Merged p50 / p95 / p99 | > 128 | > 192 | > 256 | > 320 |
|---|---|---|---|---|---|---|
| plain EN → VI | — | 43 / 114 / 164 | 3.1% | 0.4% | 0.1% | 0.0% |
| text + AMR, sense kept | 41 / 120 | 85 / 234 / 337 | 26.9% | 9.9% | 3.5% | 1.3% |
| text + AMR, sense dropped | 34 / 102 | 79 / 215 / 310 | 22.9% | 7.6% | 2.4% | 0.8% |

  With sense dropped, `seq_len 256` covers 97.6% of train sentences (the v2 build dropped 3,064 of 125,189
  chunks, 2.4%).
- **Line references** in §4–§5 point to commit `d8760b2` (before the fixes); view them with
  `git show d8760b2:<file>`.
