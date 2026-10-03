# AMR Input Configuration Diagrams and Charts

- **Type**: changelog
- **Date**: 2026-10-03
- **Related**: [../architecture/01-amr-input-configs.md](../architecture/01-amr-input-configs.md) · [2026-10-03_coref-context-variant.md](2026-10-03_coref-context-variant.md)

---

## What changed

- **New `docs/architecture/01-amr-input-configs.md`:** 7 Mermaid diagrams (overview, sequence layout per
  config, `edge_attr` vs `levi` graph, graph encoder in the denoiser, coreference chain and prep steps,
  coreference link edge, experiment arms) and 3 charts with matching tables.
- **New `scripts/plot_amr_configs.py`:** computes the chart numbers with the shipped pipeline (20,000 train
  sentences, unfiltered) and writes `docs/architecture/assets/{length_ecdf,chunk_tradeoff,coref_links_per_row}.png`
  + `amr_configs_stats.json`; `--replot` redraws from a cache in `logs/`.
- `docs/architecture/README.md` index; links from `docs/README.md`, data-format, plan, handoff, PENDING;
  doc-standards folder map; `requirements.txt` lists matplotlib.
- `thesis_env`: matplotlib 3.11.2 installed (new packages only).

## Why

Requested visual reference of what each AMR input configuration looks like, for the thesis write-up and for
choosing experiment arms.

## Files touched

`docs/architecture/**` (new), `scripts/plot_amr_configs.py` (new), `docs/README.md`, `docs/data/data-format.md`,
`docs/plans/pipeline-fixes-and-amr-redesign.md`, `docs/plans/PENDING.md`, `docs/AGENT_HANDOFF.md`,
`docs/doc-standards.md`, `docs/changelogs/*`, `requirements.txt`.

## Verification

- `python scripts/check_mermaid.py docs/architecture/01-amr-input-configs.md` → 7 / 7 PASS; the four complex
  diagrams and all charts were rendered and inspected (one label collision fixed).
- Chart numbers match the earlier measurements (e.g. text + AMR + coref: median 86, 3.1% > 256 tokens).
- No broken links in `docs/`. Palette: dataviz reference slots 1–4 (validator not run: Node unavailable);
  low-contrast slots carry direct labels and tables.
