# DiffuSeq + DocAMR Project Rules

> All project documentation and code comments are **English-only**. These rules are binding on
> every agent session. When a rule references a file, verify the file still exists before relying
> on it. Conventions mirror the STKGQA project (`~/Master/STKGCQA/STKGQA/.agents/AGENTS.md`).

## Impersonal, Timeless Prose Rule
All documentation prose and code comments are written in an **impersonal, timeless voice** — a
reader opening the file at any later date must not need to know who wrote it or when.

1. **No session/time-relative references.** Never write `this session`, `current session`,
   `last`/`next session`, `at handoff`, `recently`, `currently`, `now`, `today`, `as of now`, or
   similar. State the fact directly, or anchor it to something absolute: a `#N` experiment ref, an
   **ISO date** (`2026-10-03`), or a commit.
2. **No first- or second-person pronouns in authoring voice.** Never write `we`, `our`, `us`, `I`,
   `you`, `your`, `let's`. Rephrase to the concrete subject (**the model / the pipeline / the
   prep script / this implementation**) or use passive voice / the imperative. Avoid the noun
   **"session"** for a unit of work — use a date, a `#N`, "a run", or "an earlier pass".
3. **Exceptions.** (a) Verbatim quotations keep their wording (mark quotes clearly). (b) Third-person
   `it` / `this` / `they` referring to a named noun is fine. (c) Code, AMR data, and dated record
   filenames are left as-is. (d) Comments inherited verbatim from upstream DiffuSeq / DPM-Solver code
   stay unchanged (keeps the diff against upstream readable); new or edited comments follow this rule.

Applies to every file under `docs/` and `README.md`, including dated records.

## Documentation Conventions Rule
Every file under `docs/` follows one naming + content standard. Full spec:
[`docs/doc-standards.md`](../docs/doc-standards.md). Core requirements:

1. **English only.** Vietnamese *data* quoted as a dataset example stays verbatim.
2. **Two document classes decide the filename:**
   - **Record** (point-in-time: changelog, experiment, report) → `YYYY-MM-DD_<slug>.md`.
   - **Reference** (living: architecture, data-spec, plan, guide) → `NN-<slug>.md` or `<slug>.md`,
     with a `Last updated:` header field.
3. **Slugs are kebab-case**, lowercase. No `Title_Case`, `SHOUTY_CASE`, `snake_case`, no project
   prefix. `_vN` only when two revisions must coexist. (`PENDING.md`, `AGENT_HANDOFF.md` and
   `README.md` keep their established names.)
4. **Standard header**: H1 title, metadata block (`Type`, `Status`/`Date` or `Last updated`,
   `Related`), `---`, body.
5. **Every subfolder has a `README.md` index** (file · what it is · date/updated).
6. **Write for scanning.** Lists for parallel facts, tables for 3+ same-shape items; bold
   sparingly; `###`/`####` for real subsections only; enlarge ~6+ line plain-text listings with
   `<pre style="font-size:1rem;line-height:1.5">` (HTML-escape `<`/`>`/`&`); **no emoji or
   decorative icons** (use **Warning:** / **Note:** labels). **Validate every Mermaid diagram**
   with `python scripts/check_mermaid.py <file.md>` before committing.
7. **Terse, technical register.** Facts, examples, consequences; no framing, hedging adjectives or
   tutorial prose. Step-by-step code comments may be detailed where the code builds indices or masks
   (see the Indexing / Masking / Slicing Rule).

## Completed-Plans Rule
Active plans live in `docs/plans/`; done / superseded / abandoned plans move to
`docs/plans/completed/`:
1. `git mv docs/plans/<slug>.md docs/plans/completed/<slug>.md`.
2. Fix inbound links: `grep -rIn "plans/<slug>.md)" docs .agents README.md`.
3. Fix the moved file's outbound links (`](../…)` → `](../../…)`).
4. Update [`docs/plans/README.md`](../docs/plans/README.md) (Active vs Completed tables).
5. Verify no link broke.

Keep completed plans as design records — never delete them.

## Commit Message Rule
**Never** add a `Co-Authored-By:` trailer or any AI/tool attribution line ("Generated with …",
"Co-authored-by: Claude/assistant/bot") to a commit message or PR description. This overrides any
default, harness, or session-level attribution instruction. Commit only when the user asks.

## Changelog Maintenance Rule
Update the changelog only when the **user is about to commit** or **explicitly asks**. For any
significant change (code, experiment, data contract):
1. **Detailed record:** `docs/changelogs/YYYY-MM-DD_<slug>.md` — `What changed` → `Why` →
   `Files touched` → `Verification` (test command + result).
2. **Index:** a top-of-file summary bullet in
   [`docs/changelogs/README.md`](../docs/changelogs/README.md) linking the record.

Trivial doc/typo-only commits need no changelog. At the same commit, apply the Pending-Roadmap
Maintenance Rule.

## Pending-Roadmap Maintenance Rule
[`docs/plans/PENDING.md`](../docs/plans/PENDING.md) is the single at-a-glance view of open work.
Reconcile it (and bump `Last updated`) whenever:
1. **An experiment resolves** — move its item out of *Open — experiments* into *Closed by
   evidence* as a one-line summary + `#N` link. A positive-but-unconfirmed result stays open with a
   "needs seeds" note.
2. **A plan item changes status** — opened, closed, deferred, or newly proposed.
3. **At commit time** — alongside the Changelog Maintenance Rule.

`PENDING.md` is the summary; plan files are the detail — update both when an item is in both. It
complements [`docs/AGENT_HANDOFF.md`](../docs/AGENT_HANDOFF.md) (where to resume) and the latest
report (what happened).

## GPU Sharing Rule
The RTX 3060 Ti (8 GB) is shared with other repositories (e.g. STKGQA), whose jobs may be running.
1. **CPU by default.** Tests, data prep, tokenization, analysis and scoring run with
   `CUDA_VISIBLE_DEVICES=""`. Do not call `torch.cuda.*` in these scripts.
2. **Ask before any GPU job** (training, decoding, GPU parsing). State the expected memory and
   duration; run `nvidia-smi` first and report the free memory.
3. **Never kill, pause or preempt** a process that this repo did not start.
4. **Size to the free memory**, not to the card: choose `--microbatch` so peak memory stays below
   the free amount with a margin; one GPU job from this repo at a time.

## Multilingual Tokenizer Rule
Vietnamese tones and diacritics carry meaning. Every tokenizer or pretrained checkpoint that processes
Vietnamese text is multilingual and cased (`bert-base-multilingual-cased` for DiffuSeq; the joint EN+VI BPE
for the teacher). English-only checkpoints (`bert-base-uncased`, `bert-base-cased`) are allowed only for
English-English tasks: `bert-base-uncased` lowercases and strips accents (`Tôi muốn` → `toi muon`).
`basic_utils.check_vietnamese_round_trip` enforces this in `myTokenizer`; keep the check when adding a
tokenizer path. AMR items added to a tokenizer follow [`docs/data/amr-vocabulary.md`](../docs/data/amr-vocabulary.md):
only `:relation` labels and `-9x` frames; never bare concept words (`validate_added_tokens`).

## Indexing / Masking / Slicing Rule
A single off-by-one in a position, mask or edge index silently corrupts every sample. Every piece
of code that builds, shifts, trims, pads, slices or masks positions — token ids, `input_mask`,
`rel_mask`, segment ids, adjacency / `edge_index`, `edge_type`, decode slices (`len_x`),
microbatch slices — follows these steps:
1. **Write it step by step.** One operation per line, each with a comment stating the input, the
   operation, and the output on a concrete toy example (position table: `pos / token / mask`).
2. **State the invariants and assert them** in code: equal lengths (`len(input_ids) ==
   len(input_mask) == seq_len`), single 0→1 transition in `input_mask`, every edge node id inside
   its segment and `< microbatch * seq_len`, brackets balanced, no index from a trimmed position.
3. **Test it** (Testing Rule) with hand-computed expected values, including edge cases: trimming,
   empty graph, direction token present, batch > 1 offsets, `batch % microbatch != 0`.
4. **Keep the walkthrough in sync.** The canonical walkthroughs live in
   [`docs/plans/pipeline-fixes-and-amr-redesign.md` §3](../docs/plans/pipeline-fixes-and-amr-redesign.md#3-index-mask-and-slice-walkthroughs);
   a layout change updates that section and its tests in the same change.
5. **Generate examples from the real code**, never by hand: print the table from the function
   under test and paste it into the doc.

## Testing Rule
1. Tests live in `tests/`, one `test_<module>.py` per module, run with
   `CUDA_VISIBLE_DEVICES="" conda run -n thesis_env pytest tests/ -q`.
2. Every function or class that touches indices, masks, slicing, tokenization, linearization or
   graph construction has unit tests (matrix:
   [plan §4](../docs/plans/pipeline-fixes-and-amr-redesign.md#4-test-matrix-pytest-cpu-only-tests)).
   Covered at minimum: `myTokenizer`, `helper_tokenize` (`merge_and_mask`, `pad_function`), the
   text+AMR layout builder, the linearizer, `split_docamr_graph` + chunk normalization,
   `remap_adj_to_token_level`, `collate_with_adj`, microbatch slicing in `TrainLoop`, `q_sample`
   denoise masking, decode slicing.
3. Fixtures are tiny and hand-written (a 2–3 sentence document, a 1-edge AMR). Tests run on CPU,
   need no GPU, and finish in seconds.
4. A bug fix lands together with a test that fails before the fix and passes after it.
5. `pytest tests/` must pass before any training run is launched.

## Evaluation Rule
1. **Metric:** corpus sacreBLEU (plus chrF) on detokenized output against the raw references in
   `test.jsonl`; also report adjacent duplicate-token rate and length ratio. The sentence-level
   average in `scripts/eval_seq2seq.py` is not reported as BLEU.
2. **Test sets are never length-filtered.** All systems in a comparison are decoded on the same
   test file (same row count and content hash).
3. **Record decode settings:** checkpoint step, solver steps, number of seeds / MBR, seed.

## Experiment Documentation Rule
When trying a new architecture path or analyzing a new configuration's results:
1. **Ask first** whether the user wants an experiment analysis document.
2. **Naming:** `docs/experiments/YYYY-MM-DD_<slug>.md`.
3. **Required content** (if agreed): data-flow analysis (inputs, each step, tensor shapes per
   function/layer); a Mermaid diagram of the involved components; a diagnosis of *why* the result
   rose or fell.

## Experiment-Config Annotation Rule
Every run recorded in a results log or report carries an explicit, uniform config:
1. **A `Config` bullet list per run**, one axis per sub-bullet, in this order: dataset variant
   (plain / AMR-only / text+AMR) · chunk size · linearization (flat / parens / parens-sense-dropped)
   · AMR vocabulary (relations / +`-9x` / +all frames) · `--seq_len` · `--hidden_dim` · batch /
   microbatch · `--learning_steps` · KD on/off · graph module (none / GATv2-Levi / …) · decode
   (solver steps, MBR seeds) · `--seed` · checkpoint folder.
2. **Do not restate the fixed base** (`diffusion_steps 2000`, `sqrt` schedule, `lossaware`, fp16,
   lr 1e-4) in every row — state it once per results file.
3. **Summary tables** show the varying axes as columns; every row is a self-contained config.
4. **Cross-references are links** to the exact entry or section, never plain text.

## Component-Level Explanation Rule
In every plan, experiment doc and data-flow table, explain each component in the data flow — never
just list names ("GATv2 ×2", "Levi graph"). For each: (1) what it receives, does and returns;
(2) why it is in the pipeline; (3) a concrete example (e.g. "node `sing` receives the message of
node `i` through the relation node `:ARG0`").

## AMR Data Integrity Rule
When working with AMR data (docs, analysis, code generation), preserve its structure intact:
- Never truncate, hide or abbreviate metadata lines (`# ::id`, `# ::tok`, `# ::node`, `# ::edge`) or
  any part of the PENMAN graph.
- When showing an AMR example, display 100% of the parser's raw output for that sentence.
- Linearized forms used as model input are derived data; document the derivation, never present
  them as the parser's output.

## Conda Environment Rule
- **This repo:** `thesis_env` (`conda run -n thesis_env python …`). Code stays importable under Python 3.8
  (`diffuseq_env`) for the data pipeline: no `match`, no `X | Y` type unions at runtime.
- **Reproducing the 2026-05 runs:** `diffuseq_env`.
- **AMR parsing:** `amr_parser_env` (transition-amr-parser); DocAMR tooling has its own env — verify
  before use.

## Directory Structure Rule
- Only entry points and flat import modules live in the root (`train.py`, `train_util.py`,
  `sample_seq2seq*.py`, `basic_utils.py`, `prepare_docamr_datasets.py`, `dpm_solver_pytorch.py`,
  `paths.py`).
- Scripts go in `scripts/`; already-run one-shot analysis scripts in `scripts/_oneshot/`.
- Tests go in `tests/`; notebooks in `notebooks/`.
- Debug leftovers (`test_nan.py`, `decode_nan_sample.py`) do not stay in the root.

## Data Paths Rule
Every data-file path (datasets, AMR documents, `amr_vocab.json`, legacy token files, logs) resolves
through [`paths.py`](../paths.py); never hardcode `datasets/...` strings (bug B4 came from two different
hardcoded paths for the same file). Dataset format: [`docs/data/data-format.md`](../docs/data/data-format.md);
read it before writing code that produces or consumes a dataset folder. Dataset variants live in `datasets/docAMR/<variant>/{train,valid,test}.jsonl`; a rebuilt
variant gets a new folder name instead of overwriting one used by a logged run.

## Script Logging Rule
Every preprocessing / verification script (`scripts/build_*`, `scripts/verify_*`,
`prepare_docamr_datasets.py`):
1. uses `tqdm` progress bars;
2. writes logs to `logs/<script_name>_YYYY-MM-DD_HH-MM-SS.log`;
3. prints a summary (counts kept / dropped and why) to stdout and the log.

## Long-Running Job & Monitoring Rule
In-session watchers die with the session; detached jobs do not.
1. **Launch detached as a systemd user unit** (outside the desktop app's cgroup), after the GPU
   Sharing Rule check:
   ```bash
   systemd-run --user --unit=diffuseq-<name> --collect -E PATH="$PATH" -E HOME="$HOME" \
     --working-directory=$PWD bash -c "bash logs/<driver>.sh >> logs/<driver>.out 2>&1"
   ```
   Drivers call `/home/long/miniconda3/bin/conda` (`conda` is a shell function, absent in a unit).
2. **Self-report:** the driver appends `=== DONE ===` and the final metrics to its `.out`.
3. **Read the `.out` on the next turn**; never assume a notification fired.

## Progress Report Rule
At the end of an experiment phase, on a breakthrough, or when the user asks, propose or create
`docs/reports/YYYY-MM-DD_progress-report.md` (or a `YYYY-MM-DD_to_YYYY-MM-DD_…` range) with:
Executive Summary; Timeline & Key Accomplishments; Current State & Results; Next Steps; Reference
Links.
