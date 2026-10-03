# AMR Input Configurations

- **Type**: architecture
- **Status**: current
- **Last updated**: 2026-10-03
- **Related**: [../data/data-format.md](../data/data-format.md) · [../plans/pipeline-fixes-and-amr-redesign.md](../plans/pipeline-fixes-and-amr-redesign.md) · [`scripts/plot_amr_configs.py`](../../scripts/plot_amr_configs.py)

---

## Purpose

Visual reference for every way the pipeline feeds English AMR to the DiffuSeq translator: what the source
sequence looks like per dataset variant, how the graph module reads it, how the coreference context is
attached, and how long the resulting sequences are. Field-level contract:
[data-format.md](../data/data-format.md).

## Overview

```mermaid
flowchart LR
  EN["EN document"] --> P["prepare_docamr_datasets.py"]
  VI["VI document"] --> P
  AMR["English DocAMR"] --> P
  P --> C1["plain_en_vi: text only"]
  P --> C2["amr_en_vi: AMR only"]
  P --> C3["text_amr_en_vi: text + AMR"]
  P --> C4["text_amr_coref_en_vi: text + AMR + coref"]
  C1 --> M["DiffuSeq denoiser"]
  C2 --> M
  C3 --> M
  C4 --> M
  C2 -. "graph_src" .-> G["GATv2 graph encoder (optional)"]
  C3 -. "graph_src" .-> G
  C4 -. "graph_src + coref links" .-> G
  G --> M
  M --> OUT["Vietnamese sentence"]
```

All four configs share the same sentences (built together, 121,453 train rows each), the same target (one
Vietnamese sentence) and the same unfiltered test file (1,098 rows); they differ only in the source.

---

## Details

### 1. Sequence layout per config

Constructed example. Document sentence 1: *"Alan Turing was a scientist."*; the row is sentence 2:
*"He sang."* → *"Ông ấy đã hát."* (the pronoun *ông ấy* fits an older, respected man; the
coreference context tells the model who *he* is).

```mermaid
flowchart TB
  subgraph A["plain_en_vi: text only"]
    direction LR
    a1["[CLS] He sang . [SEP]"]:::txt ~~~ a2["[SEP]"]:::sep ~~~ a3["[CLS] Ông ấy đã hát . [SEP] [PAD] ..."]:::tgt
  end
  subgraph B["amr_en_vi: AMR only"]
    direction LR
    b1["[CLS] ( sing :ARG0 he ) [SEP]"]:::amr ~~~ b2["[SEP]"]:::sep ~~~ b3["[CLS] Ông ấy đã hát . [SEP] [PAD] ..."]:::tgt
  end
  subgraph C["text_amr_en_vi: text + AMR"]
    direction LR
    c1["[CLS] He sang . [SEP]"]:::txt ~~~ c2["( sing :ARG0 he ) [SEP]"]:::amr ~~~ c3["[SEP]"]:::sep ~~~ c4["[CLS] Ông ấy đã hát . [SEP] [PAD] ..."]:::tgt
  end
  subgraph D["text_amr_coref_en_vi: text + AMR + coref"]
    direction LR
    d1["[CLS] He sang . [SEP]"]:::txt ~~~ d2["( sing :ARG0 he ) [SEP]"]:::amr ~~~ d3[":same-as ( person :name ( name :op1 Alan :op2 Turing ) ) [SEP]"]:::ctx ~~~ d4["[SEP]"]:::sep ~~~ d5["[CLS] Ông ấy đã hát . [SEP] [PAD] ..."]:::tgt
  end
  A ~~~ B
  B ~~~ C
  C ~~~ D
  classDef txt fill:#cde2fb,stroke:#2a78d6,color:#0b0b0b
  classDef amr fill:#fbe0d5,stroke:#eb6834,color:#0b0b0b
  classDef ctx fill:#d3f0e4,stroke:#1baf7a,color:#0b0b0b
  classDef sep fill:#f0efec,stroke:#52514e,color:#0b0b0b
  classDef tgt fill:#ffffff,stroke:#52514e,stroke-dasharray: 4 3,color:#52514e
```

| Segment | Box style | `input_mask` | Diffusion |
|---|---|---|---|
| English text | blue | 0 (source) | kept clean, re-anchored every step |
| linearized AMR of the sentence | orange | 0 | kept clean |
| coreference context | green | 0 | kept clean |
| separator `[SEP]` | gray | 0 | kept clean |
| Vietnamese target incl. `[PAD]` | dashed | 1 (target) | starts as noise, denoised to the translation |

The `bidirectional` variant adds a direction token after `[CLS]` (`[AMR_TO_TEXT]` rows have the AMR-only
layout; `[TEXT_TO_AMR]` rows translate Vietnamese into AMR) and is kept as an auxiliary-task option.

<p align="center">· · ·</p>

### 2. Length cost of each config

![Merged sequence length per input config](assets/length_ecdf.png)

| Config | Median | p95 | Rows > 256 tokens |
|---|---|---|---|
| text only | 43 | 114 | 0.1% |
| AMR only | 60 | 165 | 0.7% |
| text + AMR | 79 | 216 | 2.6% |
| text + AMR + coref | 86 | 226 | 3.1% |

20,000 train sentences, before the length filter. The build drops a train sentence from every config when
any config exceeds 256 tokens (3.0% of sentences), so all configs train on the same rows.

<p align="center">· · ·</p>

### 3. Graph input: `edge_attr` vs `levi`

The AMR part of the source carries `graph_src` entries `[head, dep, label, label_pos]` (token positions).
`--graph_encoder gatv2` turns them into edges in one of two ways:

```mermaid
flowchart LR
  subgraph EA["graph_mode edge_attr: relation as edge feature"]
    direction LR
    s1(("sing")) -- ":ARG0, forward" --> h1(("he"))
    h1 -. ":ARG0, reverse" .-> s1
  end
  subgraph LV["graph_mode levi: relation token as node"]
    direction LR
    s2(("sing")) -- "type 0" --> r2[":ARG0"]
    r2 -- "type 1" --> h2(("he"))
    r2 -. "type 2" .-> s2
    h2 -. "type 3" .-> r2
  end
```

| Mode | Nodes | Edge types | Relation identity enters through |
|---|---|---|---|
| `edge_attr` | concepts | 2 × (145 relations + unknown) | the edge-feature embedding of the GATv2 attention |
| `levi` | concepts + relation tokens | 4 (head→label, label→dep, reverses) | the relation token's own embedding |

Where the graph module sits in the denoiser:

```mermaid
flowchart LR
  X["x_t: source clean, target noised"] --> Q{"graph_encoder"}
  Q -- "none" --> UP["up-projection + position + timestep embedding"]
  Q -- "gatv2" --> GAT["GATv2 on graph positions, zero-init residual"]
  GAT --> UP
  UP --> ENC["BERT encoder, 12 layers"]
  ENC --> DOWN["down-projection"]
  DOWN --> X0["predicted x_0"]
  X0 --> TOK["nearest token per position"]
```

The GATv2 output is added only on positions that appear in an edge (all in the source); noised target
positions pass through unchanged, and at initialization the model equals the no-graph model.

<p align="center">· · ·</p>

### 4. Coreference context

How an earlier-sentence antecedent becomes part of the row (`find_coref_links` →
`append_coref_context`):

```mermaid
flowchart LR
  subgraph DOC["document graph (DocAMR)"]
    direction TB
    p1["sentence 1: s1.p / person, :name Alan Turing"]
    p2["sentence 4: s4.p / person"]
    h5["sentence 5 (the row): s5.h / he"]
    h5 -- ":same-as" --> p2
    p2 -- ":same-as" --> p1
  end
  subgraph STEPS["prepare_docamr_datasets.py"]
    direction TB
    st1["1. find links to earlier sentences"] --> st2["2. follow the chain to the earliest mention"]
    st2 --> st3["3. linearize it, depth 2, names complete"]
    st3 --> st4["4. append after the AMR, add link edge"]
  end
  DOC --> STEPS
  STEPS --> ROW["He sang . [SEP] ( sing :ARG0 he ) [SEP] :same-as ( person :name ( name :op1 Alan :op2 Turing ) )"]
```

In the row, the link is a graph entry from the mention to the antecedent root through the `:same-as`
token, so GATv2 passes the antecedent's information (name, role, gender-bearing concepts) to *he*:

```mermaid
flowchart LR
  subgraph AMRSEG["AMR segment"]
    sing(("sing")) -- ":ARG0" --> he(("he"))
  end
  subgraph CTXSEG["context segment"]
    rel[":same-as"] --> person(("person"))
    person -- ":name" --> name(("name"))
    name -- ":op1 / :op2" --> alan(("Alan Turing"))
  end
  he -- "link edge" --> rel
```

Why not translate several sentences per row instead:

![Chunk size trade-off](assets/chunk_tradeoff.png)

| Sentences per row | Coreference links kept inside the row | Rows > 256 tokens (text + AMR) |
|---|---|---|
| 1 | 0% | 2.3% |
| 2 | 10% | 18.6% |
| 3 | 17% | 48.5% |
| 5 | 25% | 88.2% |

59% of links point 6+ sentences back, so chunks recover few of them while the Vietnamese output grows past
what the diffusion model handles well. The context approach covers every backward link at +7 tokens median:

![Antecedents per row](assets/coref_links_per_row.png)

| Antecedents in the row | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Share of train rows | 36.5% | 43.5% | 16.0% | 3.3% | 0.7% |

<p align="center">· · ·</p>

### 5. Experiment arms

```mermaid
flowchart TB
  BASE["same 121,453 train rows, same model and budget, same 1,098-row test file, 3 seeds"]
  BASE --> T["text only"]
  BASE --> TA["text + AMR"]
  BASE --> TC["text + AMR + coref"]
  TA --> TAG["+ GATv2: edge_attr / levi"]
  TC --> TCG["+ GATv2: edge_attr / levi"]
  TC --> SLICE["also scored on the 61% of test rows with a coref context"]
```

AMR only (`amr_en_vi`) is a diagnostic arm: AMR omits tense, number, articles and word order, so it is
expected to score below text only.

---

## Gotchas

- **Which AMR items are single tokens** (relations, `-9x` frames) and why the rest are not:
  [amr-vocabulary.md](../data/amr-vocabulary.md).
- **Graph positions are tokenizer-bound.** Every `graph_src` position was computed with mBERT +
  `amr_vocab.json`; build and train with the same `--amr_vocab`.
- **Text-only runs cannot see which mention a context entry belongs to** (entries follow mention order);
  only the graph link edge is precise. Pointer tokens are a proposed alternative.
- **Charts are regenerated, not edited:** `CUDA_VISIBLE_DEVICES="" python scripts/plot_amr_configs.py`
  (~7 min, CPU; `--replot` redraws from the cache). The tables in this file are copied from
  `assets/amr_configs_stats.json`.
- **The example in §1 is constructed** for readability; real rows are in `datasets/docAMR/v2_*` (e.g. the
  "we, Earthlings" row in [plan §3b](../plans/pipeline-fixes-and-amr-redesign.md#phase-3b--cross-sentence-coreference-context-built-2026-10-03)).
