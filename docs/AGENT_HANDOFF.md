# DiffuSeq + DocAMR Handoff

- **Type**: guide
- **Status**: current
- **Last updated**: 2026-10-03
- **Related**: [plans/PENDING.md](plans/PENDING.md) · [plans/teacher-model.md](plans/teacher-model.md) · [plans/pipeline-fixes-and-amr-redesign.md](plans/pipeline-fixes-and-amr-redesign.md) · [`.agents/AGENTS.md`](../.agents/AGENTS.md)

---

## START HERE (2026-10-03) — code complete and CPU-tested; next step is the first GPU job (KD teacher)

**Project.** Thesis: English→Vietnamese translation with DiffuSeq (continuous text diffusion; fork of
`Shark-NLP/DiffuSeq`, base commit `3b71ccc`), using English DocAMR (`transition-amr-parser` + `docAMR`) as
an extra source signal. Data: IWSLT'15 TED en–vi.

**Repository state.**
- Branch `fix/tests-and-pipeline-bugs` (off `thesis-doc-amr`), pushed to
  `github.com/lklong1551019/DiffuSeq`. Commits on top of the 2026-05 work (`d8760b2`): `70f0453` fixes +
  tests · `c96ccfc` KD builder · `2f80e25` coreference variant · `2f1115b` diagrams · `32ff228` teacher +
  tokenizer guard + AMR vocabulary doc · `0d19bfd` `train.sh` aligned to `seq_len 256` · the handoff commit.
- **Push over SSH** (HTTPS has no stored credentials; `origin` is an HTTPS URL, left unchanged):
  `git push git@github.com:lklong1551019/DiffuSeq.git fix/tests-and-pipeline-bugs`.
- Not merged into `thesis-doc-amr` / `main`; no pull request opened.

**Rules (read [`.agents/AGENTS.md`](../.agents/AGENTS.md) first — it is not auto-loaded).** Highlights:
- **GPU is shared** with other repositories (STKGQA): CPU by default (`CUDA_VISIBLE_DEVICES=""`), run
  `nvidia-smi` and **ask the user before any GPU job**, never touch other processes. Long jobs as systemd user
  units (command in §3).
- **Commits:** only when the user asks; **no `Co-Authored-By` / generated-by lines**; at each commit write a
  `docs/changelogs/YYYY-MM-DD_<slug>.md` record + index bullet and reconcile `docs/plans/PENDING.md`.
- **Indexing / masking / slicing:** step-by-step comments, asserted invariants, unit tests with hand-computed
  values, walkthrough tables generated from real code.
- **Docs:** impersonal, timeless voice (no "we/you/currently"); record vs reference naming; Mermaid validated
  with `python scripts/check_mermaid.py <file>`.
- **Multilingual Tokenizer Rule:** Vietnamese only through multilingual cased tokenizers (mBERT cased for
  DiffuSeq, joint BPE for the teacher); English-only BERT only for English-English tasks.
- **User preferences seen so far:** careful, test-backed changes; measured justification for design choices;
  plain-language summaries in chat, terse register in docs; diagrams for designs.

**Read next, in order:**
1. [plans/PENDING.md](plans/PENDING.md) — open items and suggested order;
2. [plans/teacher-model.md](plans/teacher-model.md) — the next GPU job (teacher), run commands §7;
3. [plans/pipeline-fixes-and-amr-redesign.md](plans/pipeline-fixes-and-amr-redesign.md) — bugs B1–B15, phases,
   walkthroughs §3, test matrix §4;
4. [architecture/01-amr-input-configs.md](architecture/01-amr-input-configs.md) — diagrams of every input config;
5. [data/data-format.md](data/data-format.md) and [data/amr-vocabulary.md](data/amr-vocabulary.md) — dataset
   contract and token decisions;
6. [reports/2026-10-03_code-and-results-review.md](reports/2026-10-03_code-and-results-review.md) — why all of it.

---

## 1. What exists

| Area | State | Where |
|---|---|---|
| Review of the 2026-05 code and runs | done; 2026-05 baselines BLEU 4.22 (en→vi) / 6.44 (vi→en), 24.8% repeated tokens | [report](reports/2026-10-03_code-and-results-review.md) |
| Bug fixes B1–B15 + 87 CPU tests | done | `tests/`, [changelog](changelogs/2026-10-03_tests-and-pipeline-fixes.md) |
| AMR linearization (brackets, `:ARGn` digits, sense drop, char-offset graph positions) | done | `diffuseq/amr_linearize.py` |
| Datasets v2, chunk 1, `seq_len 256`: plain, AMR only, text + AMR, text + AMR + coref | built (git-ignored; rebuild in ~2 min) | `datasets/docAMR/v2_*_chunk_1/` |
| Coreference context (antecedent subtrees + link edges; 63% of rows) | done | Phase 3b of the plan |
| GATv2 graph encoder (`edge_attr` / `levi`) | done, not trained | `diffuseq/graph_encoder.py` |
| KD dataset builder | done, not run | `scripts/build_kd_dataset.py` |
| KD teacher trainer (Marian, 40.3M params, joint BPE 16k) | done, CPU smoke only | `teacher/`, `scripts/train_teacher.py` |
| Evaluation (corpus sacreBLEU, chrF, repetition, MBR) | done | `scripts/eval_bleu.py` |
| Diagrams / charts | done | `docs/architecture/` |

Decisions already made (do not re-litigate without new evidence):
- **Input:** one-sentence targets; source = EN text + bracketed AMR (+ coreference context); AMR-only is a
  diagnostic arm.
- **Graph module:** GATv2, not GCN (GCN ignores relation labels).
- **AMR tokens:** 145 relations + 33 `-9x` frames only ([why](data/amr-vocabulary.md)).
- **KD teacher:** option B, trained here on the v2 train split (no test leakage).
- **Generated data** is git-ignored; `amr_vocab.json` is tracked and must stay fixed during a comparison.

---

## 2. Running

Nothing from this repo. GPU at last check: ~3 GB of 8 GB used by another repository.

---

## 3. Next steps (in order)

1. **Teacher training run (GPU — ask first).** Check `nvidia-smi`, then:
   ```bash
   systemd-run --user --unit=diffuseq-teacher --collect -E PATH="$PATH" -E HOME="$HOME" \
     --working-directory=$PWD bash -c "/home/long/miniconda3/envs/thesis_env/bin/python scripts/train_teacher.py \
     --name iwslt-bpe16k --preset iwslt --device cuda --bf16 >> logs/teacher_iwslt.out 2>&1; \
     echo '=== DONE ===' >> logs/teacher_iwslt.out"
   ```
   Watch the first epoch (tokens/s, memory, valid BLEU in `teacher_models/iwslt-bpe16k/metrics.jsonl`);
   resume with `--resume`. Gate: final beam-5 valid/test BLEU in the high 20s–low 30s
   ([teacher plan §1](plans/teacher-model.md#1-status--gating)).
2. **Distill (GPU — ask first):** `build_kd_dataset.py --teacher teacher_models/iwslt-bpe16k/best --tag iwslt16k`
   with all three companions and `--check_test` ([teacher plan §7](plans/teacher-model.md#7-run-commands-gpu-steps-need-approval)).
3. **Plain baseline on KD data (GPU):** `cd scripts && DATASET=v2_plain_en_vi_chunk_1_kd-iwslt16k bash train.sh`
   (`train.sh`: hidden 768, batch 384, microbatch 16, `seq_len 256` — check memory), decode with
   `MODEL_DIR=… bash run_decode_solver.sh`, score with `scripts/eval_bleu.py` (gate G2 in the plan).
4. **Text + AMR and text + AMR + coref runs**, then the **GATv2 ablation** (`--graph_encoder gatv2
   --graph_mode edge_attr|levi`), 3 seeds per arm, same test file.

---

## 4. Environment

| Env | Use | Notes |
|---|---|---|
| `thesis_env` | everything in this repo | torch 2.10 (cu128), transformers 5.3, PyG 2.7, penman 1.3, sacreBLEU 2.6, pytest 9.1, matplotlib 3.11 |
| `diffuseq_env` | reproducing the 2026-05 runs only | torch 1.13, transformers 4.22, no PyG / penman; the data pipeline still imports under Python 3.8 |

- Tests: `CUDA_VISIBLE_DEVICES="" conda run -n thesis_env pytest tests/ -q` → 87 pass in ~15 s.
- Tests set `HF_HUB_OFFLINE=1`; `bert-base-multilingual-cased` and `bert-base-uncased` are in the local HF cache.
- transformers 5.3 prints an "incorrect regex pattern … fix_mistral_regex" warning for mBERT: harmless here
  (tokenization verified identical to transformers 4.22 on 42k strings).
- Rebuild datasets: `python scripts/build_amr_vocab.py` (only if `amr_vocab.json` is missing), then
  `python prepare_docamr_datasets.py --chunk_size 1 --max_seq_len 256 --variants plain_en_vi,amr_en_vi,text_amr_en_vi,text_amr_coref_en_vi`.

---

## 5. On disk (not in git)

- `datasets/docAMR/v2_{plain_en_vi,amr_en_vi,text_amr_en_vi,text_amr_coref_en_vi}_chunk_1/` — 121,453 /
  833 / 1,098 rows each, same sentences.
- `datasets/docAMR/plain_text_{en_vi,vi_en}_chunk_1/` — 2026-05 datasets (old prep; do not mix with v2).
- `diffusion_models/diffuseq_docAMR/` — three 2026-05 plain runs; `generation_outputs/` — their decodes.
- `teacher_models/smoke/` — tiny CPU smoke teacher (disposable).
- `logs/` — script logs; `wandb/` — offline runs of 2026-03 to 2026-05.

---

## 6. Gotchas

- **Graph positions depend on the tokenizer:** build and train with `--amr_vocab relations`; decoding reloads
  the run folder's saved tokenizer.
- **`--seq_len` must equal the dataset's `--max_seq_len`** (256 for all v2 data; recorded in `meta.json`).
- **Never filter valid/test**, and compare arms only on the same test file (`eval_bleu.py` checks row counts).
- **The coreference context is precise only through the graph link edges;** text-only runs see the antecedents
  but not which mention each belongs to.
- **Resuming DiffuSeq training restarts the step counter and LR schedule** (upstream behaviour).
