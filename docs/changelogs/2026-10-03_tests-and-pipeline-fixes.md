# Tests and Pipeline Fixes (B1–B15)

- **Type**: changelog
- **Date**: 2026-10-03
- **Related**: [../reports/2026-10-03_code-and-results-review.md](../reports/2026-10-03_code-and-results-review.md) · [../plans/pipeline-fixes-and-amr-redesign.md](../plans/pipeline-fixes-and-amr-redesign.md) · [../data/data-format.md](../data/data-format.md)

---

## What changed

1. **Test harness:** `tests/` with 60 CPU-only tests (offline Hugging Face), covering tokenizer, linearizer,
   sequence layout, graph shifting, collate, microbatch slicing, denoise masking, GATv2, prep script,
   end-to-end loss and checkpoint loading.
2. **Tokenizer (B1):** added tokens restricted to relation labels and `-9x` frames
   (`validate_added_tokens`); vocabulary built from the train split by `scripts/build_amr_vocab.py`
   (`amr_vocab.json`: 145 relations, 33 frames); `--amr_vocab` modes `relations | none | legacy_full |
   legacy_simple`; decoding reloads the run folder's tokenizer verbatim.
3. **AMR linearization (B2, B3, B9, B15):** new `diffuseq/amr_linearize.py` — PENMAN-based, brackets kept,
   `:ARGn` digits kept, optional sense-suffix drop (`-9x` kept), out-of-chunk references dropped, parser
   concepts sanitized, graph positions mapped through character offsets.
4. **Prep script (B6, B7, B10):** `prepare_docamr_datasets.py` rewritten — aligned EN / VI / AMR documents,
   TED metadata dropped, variants `plain_en_vi | plain_vi_en | amr_en_vi | vi_amr | text_amr_en_vi |
   bidirectional`, joint train-only length filter, valid/test never filtered, `meta.json` per folder.
5. **Graph module (B4):** `diffuseq/gcn.py` removed; `diffuseq/graph_encoder.py` (GATv2, edge features,
   zero-init residual on graph nodes only, raises without PyG); flags `--graph_encoder --graph_mode
   --graph_layers --graph_heads` replace `--enable_gcn --use_relational_gcn --use_simple_amr`.
6. **Data loader (B11):** `diffuseq/text_datasets.py` split into pure helpers (`merge_pair`,
   `shift_source_graph`, `pad_to`, `split_source_target`, `build_edges`, `build_rel_mask`); target-side graphs
   never read.
7. **Training (B4c, B5, B14):** `train_util.slice_microbatch` re-numbers graph edges per microbatch;
   `gaussian_diffusion.build_denoise_mask` decides per row; microbatch losses weighted by batch share;
   non-finite-loss dump deduplicated and written to the run folder.
8. **Model (B8):** `init_pretrained='bert'` requires `hidden_dim == 768`.
9. **Decoding (B12, B13):** `sample_seq2seq.py` load call fixed and graph kwargs passed; both samplers
   truncate the output file instead of appending; shared `split_source_target`.
10. **Evaluation:** `scripts/eval_bleu.py` (corpus sacreBLEU, chrF, repetition rate, length ratio, MBR);
    `scripts/eval_seq2seq.py` marked legacy.
11. **Repo:** `paths.py`; launchers updated (`run_train.py`, `train.sh`, `run_decode*.py/.sh`); misleading
    comments corrected (`train.py`, `train_util.py`, samplers); `test_nan.py`, `decode_nan_sample.py`,
    tracked `.DS_Store` removed; `README.md`, `requirements.txt`, `.gitignore` (logs, generated `v2_*`
    datasets) updated; docs tree (`.agents/AGENTS.md`, `docs/`) added.

## Why

The 2026-10-03 review found the AMR path broken in ways that invalidated every AMR run (corrupted
tokenization, flat linearization without argument numbers, a GCN that never ran, target-graph leakage) and
no test coverage for any index or mask computation. Details per bug:
[review §5](../reports/2026-10-03_code-and-results-review.md#5-bugs-found) and
[§12](../reports/2026-10-03_code-and-results-review.md#12-addendum--implementation-pass-2026-10-03).

## Files touched

- **New:** `paths.py`, `diffuseq/amr_linearize.py`, `diffuseq/graph_encoder.py`, `scripts/build_amr_vocab.py`,
  `scripts/eval_bleu.py`, `scripts/check_mermaid.py`, `scripts/_oneshot/review_*.py`, `tests/*`,
  `datasets/docAMR/output_doc_amr/amr_vocab.json`, `.agents/AGENTS.md`, `docs/**`.
- **Rewritten:** `prepare_docamr_datasets.py`, `diffuseq/text_datasets.py`, `basic_utils.py`,
  `diffuseq/transformer_model.py`.
- **Edited:** `diffuseq/gaussian_diffusion.py`, `diffuseq/config.json`, `train.py`, `train_util.py`,
  `sample_seq2seq.py`, `sample_seq2seq_dpmSolver.py`, `scripts/run_train.py`, `scripts/train.sh`,
  `scripts/run_decode*.py/.sh`, `scripts/eval_seq2seq.py`, `README.md`, `requirements.txt`, `.gitignore`.
- **Removed:** `diffuseq/gcn.py`, `test_nan.py`, `decode_nan_sample.py`, `.DS_Store`, `scripts/.DS_Store`.

## Verification

- `CUDA_VISIBLE_DEVICES="" conda run -n thesis_env pytest tests/ -q` → 60 passed.
- `diffuseq_env` (Python 3.8, transformers 4.22): all modules compile; `helper_tokenize` with a graph row runs.
- Existing checkpoint `plain_text_vi_en … ema_0.9999_045000.pt` loads strictly with its saved tokenizer.
- `v2_*_chunk_1` datasets built (`--max_seq_len 256`): 122,125 / 833 / 1,098 rows per variant; 100% of
  relation positions and 99.96% of node positions verified on 5,000 train + all test rows.
- `scripts/eval_bleu.py` reproduces the on-disk decode scores (4.22 BLEU en→vi).
- GPU not used.
