"""
plot_amr_configs.py — charts for docs/architecture/01-amr-input-configs.md.

Computes, with the shipped pipeline (prepare_docamr_datasets.build_rows, real tokenizer), on a fixed random
sample of train chunks (no length filter):
  1. length_ecdf.png           cumulative share of rows by merged DiffuSeq length, per input config
  2. chunk_tradeoff.png        sentences per row vs coreference links kept inside the row / rows > 256 tokens
  3. coref_links_per_row.png   antecedents attached per row in text_amr_coref_en_vi
and writes the numbers to docs/architecture/assets/amr_configs_stats.json (the doc tables use them).

Colors: categorical slots 1-4 of the dataviz reference palette (blue, orange, aqua, yellow), assigned in
fixed order; light surface; every series direct-labeled (aqua / yellow are below 3:1 contrast).

Run from the repo root (CPU only, thesis_env; ~7 min):
    CUDA_VISIBLE_DEVICES="" python scripts/plot_amr_configs.py
    CUDA_VISIBLE_DEVICES="" python scripts/plot_amr_configs.py --replot   # redraw from logs/ cache only
"""

import argparse
import json
import os
import random
import sys
from collections import Counter
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paths  # noqa: E402
from basic_utils import myTokenizer  # noqa: E402
from diffuseq.amr_linearize import find_coref_links  # noqa: E402
from prepare_docamr_datasets import build_rows, chunk_documents, load_split, merged_length  # noqa: E402

OUT_DIR = os.path.join(paths.REPO_ROOT, "docs", "architecture", "assets")
CACHE = os.path.join(paths.LOGS_DIR, "plot_amr_configs_cache.npz")   # git-ignored
SAMPLE = 20000
SEQ_LEN = 256

SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # reference palette slots 1-4 (light)

CONFIGS = [  # (variant, label) in fixed color order
    ("plain_en_vi", "text only"),
    ("amr_en_vi", "AMR only"),
    ("text_amr_en_vi", "text + AMR"),
    ("text_amr_coref_en_vi", "text + AMR + coref"),
]


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=TEXT, fontsize=12, pad=10)
    ax.set_xlabel(xlabel, color=TEXT_2)
    ax.set_ylabel(ylabel, color=TEXT_2)
    ax.tick_params(colors=TEXT_2, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def new_figure(width=8.0, height=4.2):
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def chunks_for(size, docs):
    return chunk_documents(docs, size, Counter())


def compute():
    """Lengths per config, chunk trade-off and antecedent counts (the slow part)."""
    tok = myTokenizer(SimpleNamespace(vocab="bert", config_name="bert-base-multilingual-cased",
                                      amr_vocab="relations", checkpoint_path=""))
    hf = tok.tokenizer
    docs = load_split("train", Counter())
    rng = random.Random(0)
    chunks1 = chunks_for(1, docs)
    sample = rng.sample(chunks1, min(SAMPLE, len(chunks1)))
    stats = {"sample_rows": len(sample), "seq_len": SEQ_LEN}

    # 1. merged length per config -----------------------------------------------------------------
    lengths = {}
    for variant, _ in CONFIGS:
        lengths[variant] = np.array([merged_length(build_rows(c, variant, hf, True)[0], tok) for c in sample])
    stats["length"] = {v: {"p50": float(np.median(x)), "p95": float(np.percentile(x, 95)),
                           "over_256_pct": round(float((x > SEQ_LEN).mean() * 100), 2)}
                       for v, x in lengths.items()}

    # 2. chunk size trade-off ---------------------------------------------------------------------
    trade = {}
    for size in (1, 2, 3, 5):
        chunks = chunks1 if size == 1 else chunks_for(size, docs)
        kept = total = 0
        for c in chunks:
            # links of each sentence that stay inside the row = references resolved by the chunk itself
            for k, tree in enumerate(c["trees"]):
                start = c["start"] + k
                for link in find_coref_links([tree], c["doc_trees"], start, follow_chain=False, max_links=10 ** 6):
                    total += 1
                    kept += link.antecedent_sentence >= c["start"]
        sub = rng.sample(chunks, min(5000, len(chunks)))
        over = np.mean([merged_length(build_rows(c, "text_amr_en_vi", hf, True)[0], tok) > SEQ_LEN for c in sub])
        trade[size] = {"links_kept_pct": round(kept / max(total, 1) * 100, 1), "over_256_pct": round(over * 100, 1)}
    stats["chunk_tradeoff"] = trade

    counts = Counter(len(find_coref_links(c["trees"], c["doc_trees"], c["start"])) for c in chunks1)
    n = sum(counts.values())
    stats["antecedents_per_row_pct"] = {k: round(counts.get(k, 0) / n * 100, 1) for k in range(0, 5)}
    return stats, lengths


def plot(stats, lengths):
    # 1. merged length per config: ECDF; each curve labelled at its own height (no collisions) --------
    fig, ax = new_figure(8.0, 4.6)
    label_levels = [92, 76, 60, 44]
    for (variant, label), color, level in zip(CONFIGS, SERIES, label_levels):
        x, counts = np.unique(lengths[variant], return_counts=True)
        y = np.cumsum(counts) / counts.sum() * 100
        ax.plot(x, y, color=color, linewidth=2, label=f"{label} (median {np.median(lengths[variant]):.0f})")
        at = x[np.searchsorted(y, level)]
        ax.plot([at], [level], marker="o", markersize=6, color=color, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(label, xy=(at, level), xytext=(8, 0), textcoords="offset points", color=TEXT, fontsize=8.5,
                    va="center", bbox=dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor="none"))
    ax.axvline(SEQ_LEN, color=TEXT_2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(SEQ_LEN + 4, 30, "seq_len 256", color=TEXT_2, fontsize=8.5)
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 101)
    style(ax, "Merged sequence length by input config (train sample, unfiltered)",
          "merged length (source + separator + target tokens)", "rows at or below this length (%)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=8.5,
              labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "length_ecdf.png"), facecolor=SURFACE)
    plt.close(fig)

    trade = {int(k): v for k, v in stats["chunk_tradeoff"].items()}
    fig, ax = new_figure()
    sizes = list(trade)
    pos = np.arange(len(sizes))
    width = 0.36
    series = [("coreference links kept inside the row", "links_kept_pct"),
              ("rows longer than 256 tokens (text + AMR)", "over_256_pct")]
    for j, ((label, key), color) in enumerate(zip(series, SERIES)):
        vals = [trade[s][key] for s in sizes]
        bars = ax.bar(pos + (j - 0.5) * (width + 0.02), vals, width, color=color, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", color=TEXT, fontsize=8.5)
    ax.set_xticks(pos, [f"{s} sentence{'s' if s > 1 else ''}" for s in sizes])
    ax.set_ylim(0, 105)
    style(ax, "Chunking documents keeps few coreference links and makes rows too long",
          "sentences per row", "share (%)")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5, labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "chunk_tradeoff.png"), facecolor=SURFACE)
    plt.close(fig)

    # 3. antecedents per row ----------------------------------------------------------------------
    dist = {int(k): v for k, v in stats["antecedents_per_row_pct"].items()}

    fig, ax = new_figure(6.4, 3.8)
    keys = list(dist)
    bars = ax.bar([str(k) for k in keys], [dist[k] for k in keys], 0.6, color=SERIES[0])
    for b, k in zip(bars, keys):
        ax.text(b.get_x() + b.get_width() / 2, dist[k] + 1, f"{dist[k]:.1f}%", ha="center", color=TEXT, fontsize=8.5)
    ax.set_ylim(0, max(dist.values()) + 10)
    style(ax, "Antecedents attached per row (text + AMR + coref, all train rows)",
          "antecedents in the row's context (cap 4)", "rows (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "coref_links_per_row.png"), facecolor=SURFACE)
    plt.close(fig)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replot", action="store_true", help="redraw from the cached numbers only")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(paths.LOGS_DIR, exist_ok=True)
    stats_path = os.path.join(OUT_DIR, "amr_configs_stats.json")
    if args.replot:
        with open(stats_path) as f:
            stats = json.load(f)
        cached = np.load(CACHE)
        lengths = {v: cached[v] for v, _ in CONFIGS}
    else:
        stats, lengths = compute()
        np.savez(CACHE, **lengths)
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
    plot(stats, lengths)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
