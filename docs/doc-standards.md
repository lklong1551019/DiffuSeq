# Documentation Standards

- **Type**: guide
- **Status**: current
- **Last updated**: 2026-10-03
- **Related**: [`.agents/AGENTS.md`](../.agents/AGENTS.md) · [README.md](README.md)

---

This is the single source of truth for how documentation in this repository is named,
structured, and written. It exists so that any file is identifiable at a glance — its
*type*, *subject*, and *date / version* are clear from the name and the header alone,
without opening the body. English only.

## 1. Two document classes

Every doc is either a **record** (a point-in-time snapshot of an event) or a
**reference** (a living document revised in place). The class decides where the date
lives.

| Class | Examples | Filename format | Date lives in |
|---|---|---|---|
| Record | changelog, experiment, report / progress | `YYYY-MM-DD_<slug>.md` | the filename (event date) |
| Reference | architecture, data-spec, plan, guide | `NN-<slug>.md` (ordered) or `<slug>.md` | a `Last updated:` header field |

Rationale: a reference doc is edited continuously, so a date in its filename would rot
immediately and force churn-renames; it carries `Last updated:` in the header instead. A
record is a snapshot, so its date *is* its identity and makes the folder sort
chronologically.

## 2. Filename grammar (applies to both classes)

- **`slug`** = kebab-case, lowercase, hyphen-separated, descriptive
  (`amr-gat-ablation`, not `AMR_GAT_ABL` or `AmrGatAbl`).
- **One token style repo-wide.** No `Title_Case`, `SHOUTY_CASE`, or `snake_case`
  filenames.
- **No project prefix** (`DIFFUSEQ_`, `THESIS_`) — the folder already names the category.
- **`_vN` suffix only** when a new revision must coexist with the old one. Otherwise
  revise in place and bump `Last updated:`. Version tokens that are part of the subject
  (e.g. `english-v2`) stay in the slug.
- **`README.md`** is the one reserved name exempt from the grammar; every subfolder has
  one as its index.

## 3. Standard header (every file)

Open with an H1 title, a compact metadata block, then `---`, then the body:

```markdown
# <Human Title>

- **Type**: architecture | data-spec | plan | guide | changelog | experiment | report
- **Status**: current | draft | superseded → [<name>](path)   ← reference docs only
- **Date**: YYYY-MM-DD              ← records only (event date)
- **Last updated**: YYYY-MM-DD      ← reference docs only
- **Related**: [<name>](path) · [<name>](path)

---
```

Include the fields relevant to the class (a record uses `Date`; a reference uses
`Status` + `Last updated`). `Related` is optional but encouraged.

## 4. Per-class body skeletons

Light conventions, enforced by example not tooling:

- **experiment**: `Result (one line)` → `Setup` → `Numbers (table)` → `Analysis` →
  `Reproduce`.
- **changelog**: `What changed` → `Why` → `Files touched` → `Verification`.
- **architecture / data-spec**: `Purpose` → `Overview (Mermaid diagram)` → `Details` →
  `Gotchas`.
- **plan**: `Status / gating` → `Phases` → `Rollback`. Model file:
  [plans/pipeline-fixes-and-amr-redesign.md](plans/pipeline-fixes-and-amr-redesign.md).
- **report**: see the Progress Report Rule in `.agents/AGENTS.md`.

## 5. Folder map

<pre style="font-size:1rem;line-height:1.5">
docs/
├── README.md            # top-level index of the docs tree
├── doc-standards.md     # this file
├── AGENT_HANDOFF.md     # live handoff / current-state pointer
├── architecture/        # living design docs (NN-&lt;slug&gt;.md) + assets/ (generated charts)
├── data/                # data specs &amp; guides, created on first use
├── experiments/         # dated experiment write-ups (YYYY-MM-DD_&lt;slug&gt;.md)
├── changelogs/          # dated changelog entries + README.md index
├── plans/               # active implementation plans (&lt;slug&gt;.md)
│   └── completed/       # done/superseded plans, kept as design records
└── reports/             # dated progress reports and reviews
</pre>

## 6. Prose formatting (readability)

Write for **scanning**, not just reading — a dense multi-fact paragraph renders as a wall of text
that is hard to skim.

- **Terse, technical register.** State facts directly; prefer tables / lists / example blocks over
  narrative. Cut framing and meta-commentary (*"the most important thing"*, *"a common source of
  confusion"*, *"worth noting"*), restatement of what a table already shows, and motivational/tutorial
  prose. One fact per clause; drop hedges and information-free adjectives. Give the fact, the example,
  the consequence — nothing else. (Step-by-step code comments may stay detailed when the file teaches a flow.)
- **Break up walls.** If a paragraph runs ~4+ sentences **and** packs several parallel facts
  (a set of properties, steps, cases, or reasons), convert it to a **bulleted or numbered list** —
  ideally a one-line lead-in, then the items. Use **numbered** lists for ordered steps, **bulleted**
  for unordered facts.
- **Keep genuine prose as prose.** A single flowing argument (one claim developed over a few
  sentences) should stay a paragraph — don't shred every sentence into a bullet.
- **Lead with the point.** Bold the key phrase at the start of each bullet (`- **Coverage:** …`) so
  the scan-line carries the meaning.
- **One idea per bullet;** split a bullet that becomes a run-on.
- **Don't let bold lead-ins become pseudo-headings.** Reserve `###`/`####` for real subsections. A
  bold lead-in is a short **label:** (a few words) opening a paragraph or bullet — never a whole bolded
  *sentence* standing in for a heading. If a section accumulates several *ungrouped* bold-lead
  paragraphs that each read like a sub-heading (so the reader can't tell where one subtopic ends and the
  next begins), promote the distinct ones to `####` or merge them into a list. Labeled *sequences*
  (Step A/B/C, Matcher A/B, S1–S9) and short colon-intros to a list/table are fine — they aid the scan.
- **Bold sparingly — only what's important or easy to miss.** Bold marks a genuine attention point (a
  warning, the one load-bearing term, a critical caveat), not ordinary emphasis. Inline, bold at most
  the single key word/phrase per point — bolding whole clauses, or every number, flattens the hierarchy
  so nothing stands out. **In tables, bold may spotlight a key figure, but don't saturate** — a table
  whose cells are mostly bold reads as no emphasis at all; lean on the table's own structure instead.
- **Separate subsections subtly.** Major (`##`) sections are divided by a full `---` rule. Between
  `###` subsections, use a light centered divider — `<p align="center">· · ·</p>` — so subsection
  boundaries are visible without competing with the major-section rule (skip it before the first
  subsection sitting directly under a `##`). Reserve the full `---` for `##` breaks only.
- **Right structure for the content:** a **table** for 3+ items of the same shape (variant × metric,
  field × meaning); **fenced code** for tensors/shapes/commands; **Mermaid** for data/architecture flow.
- **Validate every Mermaid diagram before committing.** A parse error renders as an ugly red box in the
  viewer but is invisible in the raw Markdown, so it must be caught mechanically — run
  `python scripts/check_mermaid.py <file.md>` (validates each `mermaid` block via the local `mmdc` CLI,
  or kroki.io as a fallback) and fix any `FAIL` before committing. Common gotchas it catches:
  - **Reserved keywords** (`graph`, `end`, `class`, …) as a `classDef` name or node id — e.g. `:::graph`
    fails with `got 'GRAPH'`; rename the class (`:::kg`).
  - **A bare `&` inside a label** lexes as the chaining operator — write "and".
  - Quote every label as `["…"]`; inside quotes, parens / `→` / `+` / `≥` are fine.
- **Enlarge plain-text listing blocks.** A substantial (~6+ line) *plain-text* gray block that a reader
  actually studies — an AMR parse, a dependency parse, an ASCII diagram, a directory tree, a
  tensor-shape walkthrough — is easier to read one step up from the default code font. Wrap it in a raw
  `<pre style="font-size:1rem;line-height:1.5">…</pre>` instead of a ` ``` ` fence (HTML-escape any
  `<`, `>`, `&` in the content → `&lt;`, `&gt;`, `&amp;`). **Only these listing/diagram blocks** —
  leave syntax-highlighted code (` ```python `, ` ```bash `, ` ```json `) and Mermaid as fences (they
  read fine and keep highlighting), and don't bother enlarging short (<6-line) snippets or formula
  one-liners. **GitHub caveat:** github.com strips inline `style`, so there the block reverts to the
  default code font (no data lost); see the viewing note in [`README.md`](README.md). Keep the size
  value consistent at `1rem` across docs.
- **No decorative icons / emoji in doc bodies — warning-callout markers included.** Icon glyphs
  (checkmarks, crosses, coloured dots, stars, warning/target/rocket emoji, …) render inconsistently and
  usually *ugly* across viewers, and misalign inside tables. **Do not use them as content** — say
  "held / varies", "yes / no", "done / dropped" in words, or lean on the table's own structure; for a
  callout use a bold **Warning:** / **Note:** / **Caution:** label, not an emoji. There are **no grandfathered markers**.
  The centered `· · ·` subsection divider is text punctuation, not an icon, and stays.
- Wrap source lines at ~100 chars (doesn't change rendering, keeps diffs readable).

Rule of thumb: if "what are the N things this says?" can't be answered at a glance, make it a list.

## 7. Adding or moving a doc

1. Pick the class (record vs reference) → pick the filename format from §1–2.
2. Add the standard header (§3) and follow the body skeleton (§4); keep prose scannable (§6).
3. Add/refresh the row for it in the subfolder `README.md` index.
4. When a file is renamed/moved, update every internal link that pointed at it and
   confirm no dangling links remain (`grep -rn "](" docs`).
