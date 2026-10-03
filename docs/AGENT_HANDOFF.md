# DiffuSeq + DocAMR Handoff

- **Type**: guide
- **Status**: current
- **Last updated**: 2026-10-03
- **Related**: [plans/PENDING.md](plans/PENDING.md) · [reports/2026-10-03_code-and-results-review.md](reports/2026-10-03_code-and-results-review.md) · [data/data-format.md](data/data-format.md) · [`.agents/AGENTS.md`](../.agents/AGENTS.md)

---

## Start here (2026-10-03) — fixes done and tested on CPU; v2 datasets built; no training run yet

**Project.** DiffuSeq (continuous text diffusion, fork of `Shark-NLP/DiffuSeq`, base commit `3b71ccc`)
for English→Vietnamese translation with English DocAMR (`transition-amr-parser` + `docAMR`). Work branch
`fix/tests-and-pipeline-bugs` off `thesis-doc-amr` (committed; generated `v2_*` datasets git-ignored). Data: IWSLT'15 TED en–vi.

**Read first, in order:**
1. [plans/PENDING.md](plans/PENDING.md) — open items and suggested order;
2. [plans/pipeline-fixes-and-amr-redesign.md](plans/pipeline-fixes-and-amr-redesign.md) — what was fixed
   (B1–B15), index walkthroughs (§3), test matrix (§4);
3. [data/data-format.md](data/data-format.md) — dataset contract (`graph_src`, `amr_vocab.json`; token decisions in
   [data/amr-vocabulary.md](data/amr-vocabulary.md)), with diagrams in
   [architecture/01-amr-input-configs.md](architecture/01-amr-input-configs.md);
4. [plans/teacher-model.md](plans/teacher-model.md) — the KD teacher (next GPU job);
5. [reports/2026-10-03_code-and-results-review.md](reports/2026-10-03_code-and-results-review.md) — why.

## Running

Nothing from this repo. The GPU (RTX 3060 Ti, 8 GB) is shared with other repositories; check `nvidia-smi`
and ask the user before any GPU job (GPU Sharing Rule).

## Environment

- `thesis_env` — target env (torch 2.10, transformers 5.3, PyG 2.7, penman 1.3, sacreBLEU 2.6, pytest 9.1).
  Tests: `CUDA_VISIBLE_DEVICES="" conda run -n thesis_env pytest tests/ -q` (87 pass).
- `diffuseq_env` — reproduces the 2026-05 runs (torch 1.13, transformers 4.22, no PyG / penman).

## On disk

- `datasets/docAMR/v2_{plain_en_vi,amr_en_vi,text_amr_en_vi,text_amr_coref_en_vi}_chunk_1/` — rebuilt
  datasets (`seq_len 256`, built together: 121,453 train rows each; git-ignored).
- `datasets/docAMR/output_doc_amr/amr_vocab.json` — 145 relation labels + 33 `-9x` frames (train split).
- `datasets/docAMR/plain_text_{en_vi,vi_en}_chunk_1/` — 2026-05 datasets (old prep; do not mix with v2).
- `diffusion_models/diffuseq_docAMR/` — three 2026-05 plain runs (two complete, one crashed);
  `generation_outputs/` — their decodes (BLEU 4.22 en→vi, 6.44 vi→en).
- `logs/` — script logs (git-ignored).
