# Changelogs

- **Type**: guide
- **Status**: current
- **Last updated**: 2026-10-03

---

Index of `docs/changelogs/`, newest first. Each bullet links to a dated record
`YYYY-MM-DD_<slug>.md` (Changelog Maintenance Rule in [`.agents/AGENTS.md`](../../.agents/AGENTS.md)).

- **2026-10-03** — KD teacher (`teacher/`, `scripts/train_teacher.py`), Vietnamese tokenizer guard, AMR vocabulary rationale → [details](2026-10-03_teacher-model-and-amr-vocabulary.md)
- **2026-10-03** — diagrams and charts of every AMR input config (`docs/architecture/01-amr-input-configs.md`, `scripts/plot_amr_configs.py`) → [details](2026-10-03_amr-config-diagrams.md)
- **2026-10-03** — cross-sentence coreference context variant `text_amr_coref_en_vi` (antecedent subtrees + link edges; 63% of rows) → [details](2026-10-03_coref-context-variant.md)
- **2026-10-03** — KD dataset builder `scripts/build_kd_dataset.py` (teacher-agnostic, resumable, joint length filter, valid/test untouched) → [details](2026-10-03_kd-dataset-builder.md)
- **2026-10-03** — test harness (60 CPU tests) and fixes B1–B15: AMR tokenizer, bracketed linearization, prep rewrite, GATv2 replaces GCN, microbatch graph slicing, per-row masking, sampler fixes, sacreBLEU evaluation → [details](2026-10-03_tests-and-pipeline-fixes.md)
