# Pending work — roadmap at a glance

- **Type**: guide
- **Status**: current
- **Last updated**: 2026-10-03
- **Related**: [README.md](README.md) (plans index) · [pipeline-fixes-and-amr-redesign.md](pipeline-fixes-and-amr-redesign.md) · [../reports/2026-10-03_code-and-results-review.md](../reports/2026-10-03_code-and-results-review.md) · [../data/data-format.md](../data/data-format.md) · [../AGENT_HANDOFF.md](../AGENT_HANDOFF.md)

---

One-screen view of open work. The plan files hold the detail; keep this file in sync when an item
lands, changes status, or dies (Pending-Roadmap Maintenance Rule in
[`.agents/AGENTS.md`](../../.agents/AGENTS.md)). Bug IDs: B1–B10 in
[review §5](../reports/2026-10-03_code-and-results-review.md#5-bugs-found), B11–B15 in
[review §12](../reports/2026-10-03_code-and-results-review.md#12-addendum--implementation-pass-2026-10-03).

## Status timeline

Newest first; one dated entry per milestone.

**2026-10-03 (evening)** — tests and bug fixes committed on branch `fix/tests-and-pipeline-bugs`
([changelog](../changelogs/2026-10-03_tests-and-pipeline-fixes.md)); generated `v2_*` datasets are git-ignored.
- 60 CPU tests pass (`tests/`); B1–B15 fixed with tests where testable on CPU
  ([plan Phase 1](pipeline-fixes-and-amr-redesign.md#phase-1--bug-fixes-done-2026-10-03)).
- New: `paths.py`, `diffuseq/amr_linearize.py` (bracketed AMR, char-offset graph positions),
  `diffuseq/graph_encoder.py` (GATv2, replaces gcn.py), `scripts/build_amr_vocab.py`,
  `scripts/eval_bleu.py`; `prepare_docamr_datasets.py` rewritten.
- Data built: `v2_{plain_en_vi,amr_en_vi,text_amr_en_vi}_chunk_1` (`seq_len 256`, 122,125 train rows each,
  same sentences; test 1,098 unfiltered).
- Comments and docs refreshed across the repo; dataset contract in [data-format.md](../data/data-format.md).

**2026-10-03 (handoff)** — all work pushed (`fix/tests-and-pipeline-bugs`, over SSH); `scripts/train.sh` set to
`--seq_len 256` / `--microbatch 16` to match the v2 datasets; handoff rewritten for a new session
([AGENT_HANDOFF.md](../AGENT_HANDOFF.md)).

**2026-10-03 (teacher)** — KD teacher decided (option B: trained on the v2 train split, no leakage) and
built: `teacher/` + `scripts/train_teacher.py` (Marian Transformer, iwslt preset 40.3M params, joint EN+VI
BPE 16k), [teacher plan](teacher-model.md); CPU smoke run end to end. Multilingual Tokenizer Rule added
(`check_vietnamese_round_trip` rejects English-only BERT for Vietnamese). 87 CPU tests pass.
- AMR vocabulary decisions documented with measurements: [amr-vocabulary.md](../data/amr-vocabulary.md)
  (bare concepts as tokens would change 95% of EN / 99.8% of VI sentences; predicates as tokens cut
  EN-AMR token sharing from 53.3% to 38.6%).

**2026-10-03 (late night, docs)** — diagrams and charts of every AMR input config:
[architecture/01-amr-input-configs.md](../architecture/01-amr-input-configs.md) (7 Mermaid diagrams, 3 charts from
`scripts/plot_amr_configs.py`).

**2026-10-03 (late night)** — cross-sentence coreference context built and committed
([changelog](../changelogs/2026-10-03_coref-context-variant.md))
([plan Phase 3b](pipeline-fixes-and-amr-redesign.md#phase-3b--cross-sentence-coreference-context-built-2026-10-03)).
- 63.5% of train sentences link to an earlier sentence (`:same-as`, 59% of links 6+ sentences back);
  chunking keeps ≤ 25% of links at ≤ 5 sentences and pushes 89% of rows past 256 tokens.
- New variant `text_amr_coref_en_vi`: one-sentence target, antecedent subtrees appended to the source,
  link edges mention → `:same-as` → antecedent; 63% of train rows carry a context (+7 tokens median).
- v2 rebuilt with four variants together: 121,453 train rows each (3.0% dropped jointly), test 1,098.
  73 CPU tests pass.

**2026-10-03 (night)** — KD data builder `scripts/build_kd_dataset.py` (teacher-agnostic, resumable cache,
joint length filter, valid/test untouched); 66 CPU tests pass. Teacher choice open.

**2026-10-03 (later)** — `thesis_env` complete; graph module decided: GATv2, not GCN.

**2026-10-03** — code and results review ([report](../reports/2026-10-03_code-and-results-review.md)):
plain path matches upstream; on-disk baselines BLEU 4.22 / 6.43; ten AMR-path bugs.

---

## Open — needs GPU (ask before launching)

| Item | Detail | Gate |
|---|---|---|
| Teacher training run | `scripts/train_teacher.py --name iwslt-bpe16k --preset iwslt --device cuda --bf16` ([teacher plan §7](teacher-model.md#7-run-commands-gpu-steps-need-approval)) | before KD |
| KD translation run | `scripts/build_kd_dataset.py --teacher teacher_models/iwslt-bpe16k/best --device cuda` over 119,429 unique train sources (+ `--check_test`) | before G2 |
| Credible plain baseline on `v2_plain_en_vi_chunk_1_kd-<tag>` | effective batch ≥ 1,024, solver-step sweep, MBR | G2 |
| Text + AMR run on `v2_text_amr_en_vi_chunk_1` | same settings as the baseline, `--seq_len 256`, `--graph_encoder none` | after G2 |
| Text + AMR + coreference run on `v2_text_amr_coref_en_vi_chunk_1` | same settings; report also the 61% of test rows with a context | after G2 |
| GATv2 ablation | `--graph_encoder gatv2 --graph_mode edge_attr` and `levi` on the text + AMR and coreference data | after the text + AMR runs |
| Controlled comparison | same test file, 3 seeds per arm, `scripts/eval_bleu.py` | G3 |

## Open — CPU work

| Item | Detail | Source |
|---|---|---|
| AMR-only trimming of over-budget test rows | trim the AMR segment (deepest subtrees first) instead of `merge_pair` popping the longer side | [plan Phase 3](pipeline-fixes-and-amr-redesign.md#phase-3--text--amr-input-data-built-2026-10-03-training-after-g2) |
| Intra-concept subword edges | connect `##` pieces of a concept to its first subword | [plan Phase 4](pipeline-fixes-and-amr-redesign.md#phase-4--graph-module-built-2026-10-03-ablation-after-phase-3) |
| Joint en+vi BPE vocabulary | 10k–16k entries; re-check graph positions with the new tokenizer | plan Phase 2 |

## Proposed directions (not scheduled)

- `seq_len` 320 (covers ~99.1%) if the 256 run shows a long-sentence deficit.
- Document-level chunks (2–5 sentences) only after chunk 1 shows an AMR effect.
- Relation-aware attention bias inside the encoder as an alternative to a separate graph module.
- Pointer tokens (SPRING-style) marking which mention each coreference context entry belongs to, for
  text-only (no graph) runs.
- Coreference depth / cap sweep (`--coref_depth`, `--coref_max`) and following the nearest instead of the
  earliest mention.

## Closed by evidence

- **Environment** — `thesis_env` has every dependency (2026-10-03); 60 CPU tests pass; the data pipeline
  also runs in `diffuseq_env` (Python 3.8).
- **GCN vs GAT** — GATv2 chosen; `diffuseq/gcn.py` removed.
- **On-disk plain baselines are genuine but setup-limited** — BLEU 4.22 / 6.44 (sacreBLEU), no code bug
  in the plain path ([review §3–4](../reports/2026-10-03_code-and-results-review.md#3-runs-on-disk-and-their-results)).
- **The 2026-04-29 "GCN" run did not use a graph** — fallback `nn.Linear` (B4).
- **Simple-AMR runs are invalid** — tokenizer corruption (B1) and lost argument numbers (B2).
- **Graph positions on the v2 data** — 100% of relation positions on the relation token, 99.96% of nodes on
  concept tokens (5,000 train + all test rows).

## Housekeeping

- `tmp_tokenizer/` is no longer written by the prep script; delete it when convenient (git-ignored).
- `plain_text_*_chunk_1` datasets and runs stay as the record of the 2026-05 baselines; new work uses `v2_*`.

## Suggested next order

1. teacher training run (GPU) → 2. KD translation run (GPU) + `--check_test` → 3. plain baseline (GPU, G2) →
4. text + AMR and text + AMR + coreference runs → 5. GATv2 ablation (edge_attr, levi) →
6. controlled comparison.
