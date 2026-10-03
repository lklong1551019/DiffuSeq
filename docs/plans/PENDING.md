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

**2026-10-03 (later)** — `thesis_env` complete; graph module decided: GATv2, not GCN.

**2026-10-03** — code and results review ([report](../reports/2026-10-03_code-and-results-review.md)):
plain path matches upstream; on-disk baselines BLEU 4.22 / 6.43; ten AMR-path bugs.

---

## Open — needs GPU (ask before launching)

| Item | Detail | Gate |
|---|---|---|
| Credible plain baseline on `v2_plain_en_vi_chunk_1` | KD targets (teacher must exclude tst2015), effective batch ≥ 1,024, solver-step sweep, MBR | G2 |
| Text + AMR run on `v2_text_amr_en_vi_chunk_1` | same settings as the baseline, `--seq_len 256`, `--graph_encoder none` | after G2 |
| GATv2 ablation | `--graph_encoder gatv2 --graph_mode edge_attr` and `levi` on the text + AMR data | after the text + AMR run |
| Controlled comparison | same test file, 3 seeds per arm, `scripts/eval_bleu.py` | G3 |

## Open — CPU work

| Item | Detail | Source |
|---|---|---|
| KD data pipeline | script that translates train sources with a teacher and writes a `v2_kd_*` variant | [plan Phase 2](pipeline-fixes-and-amr-redesign.md#phase-2--baseline-credibility-open-gpu) |
| AMR-only trimming of over-budget test rows | trim the AMR segment (deepest subtrees first) instead of `merge_pair` popping the longer side | [plan Phase 3](pipeline-fixes-and-amr-redesign.md#phase-3--text--amr-input-data-built-2026-10-03-training-after-g2) |
| Intra-concept subword edges | connect `##` pieces of a concept to its first subword | [plan Phase 4](pipeline-fixes-and-amr-redesign.md#phase-4--graph-module-built-2026-10-03-ablation-after-phase-3) |
| Joint en+vi BPE vocabulary | 10k–16k entries; re-check graph positions with the new tokenizer | plan Phase 2 |

## Proposed directions (not scheduled)

- `seq_len` 320 (covers ~99.1%) if the 256 run shows a long-sentence deficit.
- Document-level chunks (2–5 sentences) only after chunk 1 shows an AMR effect.
- Relation-aware attention bias inside the encoder as an alternative to a separate graph module.

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

1. KD data script (CPU) → 2. teacher choice + leakage check → 3. plain baseline (GPU, G2) →
4. text + AMR run → 5. GATv2 ablation (edge_attr, levi) → 6. controlled comparison.
