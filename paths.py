"""
paths.py — single owner of every data-file path in the repository.

Every script and module resolves dataset, AMR and vocabulary locations through this file
(Data Paths Rule in .agents/AGENTS.md). Paths are absolute, derived from the repository root,
so they work regardless of the working directory a script is launched from.

Layout:
    datasets/docAMR/
    ├── output_dataset_doc/<split folder>/doc-N.txt          plain EN / VI documents, one sentence per line
    ├── output_doc_amr/<split folder>/doc-N_docamr_docAMR.out  DocAMR graphs of the EN documents
    ├── output_doc_amr/<split folder>/doc-N.txt              VI documents aligned with the AMR graphs
    ├── output_doc_amr/amr_vocab.json                        relation labels + -9x frames (build_amr_vocab.py)
    ├── output_doc_amr/doc_amrs_token*.json                  legacy token lists (runs before 2026-10-03)
    └── <variant>/{train,valid,test}.jsonl                   model-ready datasets (prepare_docamr_datasets.py)
"""

import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

DOCAMR_DIR = os.path.join(REPO_ROOT, "datasets", "docAMR")
TEXT_DOC_DIR = os.path.join(DOCAMR_DIR, "output_dataset_doc")
AMR_DOC_DIR = os.path.join(DOCAMR_DIR, "output_doc_amr")

# Relation labels and -9x frames added to the tokenizer (amr_vocab = "relations").
AMR_VOCAB_FILE = os.path.join(AMR_DOC_DIR, "amr_vocab.json")

# Legacy token lists, kept only so checkpoints trained before 2026-10-03 can be rebuilt.
# doc_amrs_token_simple.json contains bare English words and corrupts tokenization (review bug B1).
LEGACY_AMR_TOKENS_FULL = os.path.join(AMR_DOC_DIR, "doc_amrs_token.json")
LEGACY_AMR_TOKENS_SIMPLE = os.path.join(AMR_DOC_DIR, "doc_amrs_token_simple.json")

DIFFUSEQ_CONFIG = os.path.join(REPO_ROOT, "diffuseq", "config.json")
LOGS_DIR = os.path.join(REPO_ROOT, "logs")

# Autoregressive teachers for knowledge distillation (scripts/train_teacher.py); git-ignored.
TEACHER_DIR = os.path.join(REPO_ROOT, "teacher_models")

# Split name -> (EN folder, VI folder); the same folder names exist under TEXT_DOC_DIR and AMR_DOC_DIR.
SPLIT_FOLDERS = {
    "train": ("train_en-vi.en", "train_en-vi.vi"),
    "valid": ("dev2010.en-vi.en", "dev2010.en-vi.vi"),
    "test": ("tst2015.en-vi.en", "tst2015.en-vi.vi"),
}


def dataset_dir(variant):
    """Folder of one model-ready dataset variant, e.g. dataset_dir('v2_plain_en_vi_chunk_1')."""
    return os.path.join(DOCAMR_DIR, variant)
