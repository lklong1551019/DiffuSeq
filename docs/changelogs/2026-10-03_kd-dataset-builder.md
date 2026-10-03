# Knowledge-Distillation Dataset Builder

- **Type**: changelog
- **Date**: 2026-10-03
- **Related**: [../plans/pipeline-fixes-and-amr-redesign.md](../plans/pipeline-fixes-and-amr-redesign.md) · [../data/data-format.md](../data/data-format.md) · [2026-10-03_tests-and-pipeline-fixes.md](2026-10-03_tests-and-pipeline-fixes.md)

---

## What changed

- **New `scripts/build_kd_dataset.py`:** sequence-level KD data for the diffusion student.
  1. checks that `--src_dataset` (plain EN→VI) and `--companions` (e.g. text + AMR) are row-aligned (row
     count, build arguments, identical targets, text+AMR source prefix);
  2. translates every unique train source with a Hugging Face encoder-decoder teacher (length-sorted
     batches; JSONL cache flushed per batch, resumable after a crash);
  3. replaces `trg` row by row in every variant, re-applies the joint length filter (drop a row from all
     variants if any exceeds `max_seq_len` or the output is empty);
  4. writes `<dataset>_kd-<tag>/` with distilled train, valid/test copied unchanged, `meta.json` + `kd` block;
  5. `--check_test`: teacher BLEU, chrF and exact-match rate on the test split (quality + leakage signal).
- **New `tests/test_build_kd_dataset.py`** (6 tests, fake teacher).
- `.gitignore`: generated `datasets/docAMR/v2_*/` folders (KD folders and caches included).
- Docs: plan Phase 2, PENDING, data-format (distilled variants), handoff.

## Why

Phase 2 of the plan: non-autoregressive / diffusion translation is usually trained on distilled targets;
the 2026-05 baselines repeat tokens (24.8% adjacent duplicates en→vi), the expected symptom of training on
multimodal human references.

## Files touched

`scripts/build_kd_dataset.py` (new), `tests/test_build_kd_dataset.py` (new),
`docs/plans/pipeline-fixes-and-amr-redesign.md`, `docs/plans/PENDING.md`, `docs/data/data-format.md`,
`docs/AGENT_HANDOFF.md`, `docs/changelogs/*`.

## Verification

- `CUDA_VISIBLE_DEVICES="" conda run -n thesis_env pytest tests/ -q` → 66 passed.
- Alignment check passes on the real `v2_plain_en_vi_chunk_1` + `v2_amr_en_vi_chunk_1` +
  `v2_text_amr_en_vi_chunk_1` (122,125 rows; 119,429 unique sources).
- No teacher run yet (teacher choice open; GPU not used).
