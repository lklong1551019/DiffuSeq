# Cross-Sentence Coreference Context Variant

- **Type**: changelog
- **Date**: 2026-10-03
- **Related**: [../plans/pipeline-fixes-and-amr-redesign.md](../plans/pipeline-fixes-and-amr-redesign.md) · [../data/data-format.md](../data/data-format.md) · [2026-10-03_kd-dataset-builder.md](2026-10-03_kd-dataset-builder.md)

---

## What changed

- **`diffuseq/amr_linearize.py`:**
  - `linearize_sentences(..., max_depth=None)`: nodes at depth ≥ `max_depth` are emitted as bare concepts;
    a node reached through `:name` is always expanded; `LinearizedAMR.var_word` maps each variable to the
    word index of its concept.
  - `index_document`, `find_coref_links` (references to earlier sentences, `:same-as` chains followed to
    the earliest mention, dedupe, cap), `append_coref_context` (`[SEP] :same-as ( antecedent ) …` +
    link triples mention → `:same-as` → antecedent root).
- **`prepare_docamr_datasets.py`:** variant `text_amr_coref_en_vi`; flags `--coref_depth` (2),
  `--coref_max` (4), `--coref_follow_chain` (true); chunks carry `doc_trees` (metadata sentences as None,
  never antecedents) and the original `start` position; `meta.json` counts rows with a context.
- **`scripts/build_kd_dataset.py`:** alignment check accepts the coreference variant.
- **Tests:** 7 new (depth limit and `:name` rule, `var_word`, chain / order / cap, in-chunk and metadata
  references ignored, exact triples of the docstring example, two links, prep row and real token
  positions); 73 total.
- **Docs:** plan Phase 3b + walkthrough §3.6 + comparison arms; PENDING; data-format; handoff.

## Why

63.5% of train sentences reference an earlier sentence (113,604 `:same-as` links, 59% of them 6+
sentences back); sentence-level rows dropped all of them, and multi-sentence chunks keep at most 25% of
the links (5 sentences) while pushing 89% of rows past 256 tokens. Vietnamese pronoun choice depends on the
referent, which the antecedent's AMR describes.

## Files touched

`diffuseq/amr_linearize.py`, `prepare_docamr_datasets.py`, `scripts/build_kd_dataset.py`,
`tests/test_amr_linearize.py`, `tests/test_prepare_docamr.py`, `docs/plans/pipeline-fixes-and-amr-redesign.md`,
`docs/plans/PENDING.md`, `docs/data/data-format.md`, `docs/AGENT_HANDOFF.md`, `docs/changelogs/*`.

## Verification

- `CUDA_VISIBLE_DEVICES="" conda run -n thesis_env pytest tests/ -q` → 73 passed.
- v2 rebuilt with `plain_en_vi, amr_en_vi, text_amr_en_vi, text_amr_coref_en_vi` (`--max_seq_len 256`):
  121,453 / 833 / 1,098 rows per variant; 63.0% of train and 61.0% of test rows carry a context; all 930
  test link edges land on mention → `:same-as` token → antecedent root; +7 tokens median, +26 at p95.
- KD alignment check passes on the four rebuilt datasets. GPU not used.
