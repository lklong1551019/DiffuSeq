"""
Shared pytest setup (Testing Rule in .agents/AGENTS.md).

- CPU only: CUDA_VISIBLE_DEVICES="" is set before torch is imported (GPU Sharing Rule).
- Offline: HF_HUB_OFFLINE=1; bert-base-multilingual-cased must be in the local HF cache.
- The repo root is put on sys.path so tests import modules as the entry points do.

Run: CUDA_VISIBLE_DEVICES="" conda run -n thesis_env pytest tests/ -q
"""

import os
import sys
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest  # noqa: E402

BASE_MODEL = "bert-base-multilingual-cased"


@pytest.fixture(scope="session")
def base_tokenizer():
    """Plain mBERT tokenizer (no added tokens)."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(BASE_MODEL)


@pytest.fixture(scope="session")
def my_tokenizer():
    """myTokenizer with amr_vocab='relations' (needs datasets/docAMR/output_doc_amr/amr_vocab.json)."""
    import paths
    if not os.path.exists(paths.AMR_VOCAB_FILE):
        pytest.skip("amr_vocab.json missing; run python scripts/build_amr_vocab.py")
    from basic_utils import myTokenizer
    return myTokenizer(SimpleNamespace(vocab="bert", config_name=BASE_MODEL, amr_vocab="relations",
                                       checkpoint_path=""))
