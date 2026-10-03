# KD Teacher, Vietnamese Tokenizer Guard, AMR Vocabulary Rationale

- **Type**: changelog
- **Date**: 2026-10-03
- **Related**: [../plans/teacher-model.md](../plans/teacher-model.md) · [../data/amr-vocabulary.md](../data/amr-vocabulary.md) · [2026-10-03_amr-config-diagrams.md](2026-10-03_amr-config-diagrams.md)

---

## What changed

- **KD teacher (option B):** `teacher/` package (`tokenizer.py` joint EN+VI BPE 16k with NFC + Metaspace;
  `data.py` token-bucket batching + collate with `-100` labels; `model.py` Marian encoder-decoder, presets
  `iwslt` 40.3M params / `tiny`; `train_loop.py` label-smoothed CE, inverse-sqrt LR, per-epoch valid BLEU,
  best / last checkpoints, resume, final beam-5 valid / test BLEU) and `scripts/train_teacher.py` (CPU by
  default). Output `teacher_models/<name>/` (git-ignored); `best/` loads in `build_kd_dataset.py` unchanged.
- **Multilingual Tokenizer Rule:** `basic_utils.check_vietnamese_round_trip` in `myTokenizer` rejects
  tokenizers that cannot reproduce Vietnamese (`bert-base-uncased` strips diacritics); the unused
  `bert-base-uncased` default in `TransformerNetModel` became `bert-base-multilingual-cased`; rule added to
  `.agents/AGENTS.md`.
- **AMR vocabulary rationale:** `docs/data/amr-vocabulary.md` + `scripts/_oneshot/amr_vocab_study.py`
  (relations and `-9x` frames added; predicates and concepts not, with measurements).
- **Docs:** `docs/plans/teacher-model.md` (architecture, pipeline, batch walkthrough, tests, run commands,
  risks); plan Phase 2, PENDING, plans index, data index, data-format, architecture doc, handoff.
- `paths.TEACHER_DIR`, `.gitignore` (`teacher_models/`).

## Why

Distillation needs a teacher that has never seen the test talks; a public en→vi model cannot guarantee
that. Vietnamese text must keep its diacritics, which English-only BERT destroys. The AMR token choices
needed an explicit, measured justification.

## Files touched

`teacher/*` (new), `scripts/train_teacher.py` (new), `tests/test_teacher.py` (new),
`scripts/_oneshot/amr_vocab_study.py` (new), `docs/plans/teacher-model.md` (new),
`docs/data/amr-vocabulary.md` (new), `basic_utils.py`, `diffuseq/transformer_model.py`, `paths.py`,
`.gitignore`, `tests/test_tokenizer.py`, `.agents/AGENTS.md`, `docs/**` indexes and plans.

## Verification

- `CUDA_VISIBLE_DEVICES="" conda run -n thesis_env pytest tests/ -q` → 87 passed (12 teacher tests: tokenizer
  round trip, batching coverage / budget, collate + label shift, LR, loss, tied embeddings, tiny training that
  learns, reload through `build_kd_dataset.hf_translator`, resume; 2 Vietnamese guard tests).
- CPU smoke run `train_teacher.py --preset tiny --max_steps 30` wrote `best/`, `last.pt`, `metrics.jsonl`,
  `final_metrics.json`.
- `iwslt` preset: 40.26M unique parameters. Mermaid diagrams pass `check_mermaid.py`; no broken doc links.
- GPU not used.
