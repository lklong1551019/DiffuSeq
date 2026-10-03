"""scripts/build_kd_dataset.py — alignment, cache/resume, joint length filter, outputs (fake teacher)."""

import json
import os
import sys

import pytest

import paths

sys.path.insert(0, os.path.join(paths.REPO_ROOT, "scripts"))
import build_kd_dataset as kd  # noqa: E402

ARGS = {"chunk_size": 1, "max_seq_len": 64, "drop_sense": True, "amr_vocab": "relations",
        "config_name": "bert-base-multilingual-cased", "prefix": "v2"}
PLAIN = [{"src": "I sing .", "trg": "Tôi hát ."}, {"src": "I want to go .", "trg": "Tôi muốn đi ."}]
TEXT_AMR = [
    {"src": "I sing . [SEP] ( sing :ARG0 i )", "trg": "Tôi hát .", "graph_src": [[6, 8, ":ARG0", 7]]},
    {"src": "I want to go . [SEP] ( want :ARG0 i :ARG1 go )", "trg": "Tôi muốn đi .", "graph_src": []},
]


def write_dataset(name, variant, train, args=ARGS):
    d = paths.dataset_dir(name)
    os.makedirs(d, exist_ok=True)
    kd.write_jsonl(os.path.join(d, "train.jsonl"), train)
    kd.write_jsonl(os.path.join(d, "valid.jsonl"), train[:1])
    kd.write_jsonl(os.path.join(d, "test.jsonl"), train)
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump({"variant": variant, "args": {**args, "variants": variant}}, f)


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DOCAMR_DIR", str(tmp_path))
    write_dataset("plain", "plain_en_vi", PLAIN)
    write_dataset("textamr", "text_amr_en_vi", TEXT_AMR)
    return tmp_path


class FakeTeacher:
    def __init__(self, outputs=None):
        self.calls = []
        self.outputs = outputs or {}

    def __call__(self, batch):
        self.calls.append(list(batch))
        return [self.outputs.get(s, "KD " + s) for s in batch]


def run_args(**kw):
    argv = ["--teacher", "fake", "--tag", "t", "--src_dataset", "plain", "--companions", "textamr"]
    for k, v in kw.items():
        argv += [f"--{k}", str(v)] if v is not True else [f"--{k}"]
    return kd.parse_args(argv)


def test_alignment_rejects_mismatch(data):
    meta = kd.load_meta("plain")
    bad = [dict(TEXT_AMR[0]), dict(TEXT_AMR[1], trg="khác")]
    with pytest.raises(ValueError, match="target differs"):
        kd.check_alignment(PLAIN, {"textamr": bad}, meta, {"textamr": kd.load_meta("textamr")})
    with pytest.raises(ValueError, match="does not start"):
        kd.check_alignment(PLAIN, {"textamr": [dict(TEXT_AMR[0], src="x [SEP] y"), TEXT_AMR[1]]}, meta,
                           {"textamr": kd.load_meta("textamr")})
    write_dataset("other", "text_amr_en_vi", TEXT_AMR, args={**ARGS, "max_seq_len": 128})
    with pytest.raises(ValueError, match="different arguments"):
        kd.check_alignment(PLAIN, {"other": TEXT_AMR}, meta, {"other": kd.load_meta("other")})


def test_end_to_end_replaces_train_targets_only(data):
    teacher = FakeTeacher()
    out = kd.run(run_args(), teacher)
    plain_train = kd.read_jsonl(os.path.join(out["__src__"], "train.jsonl"))
    amr_train = kd.read_jsonl(os.path.join(out["textamr"], "train.jsonl"))
    assert [r["trg"] for r in plain_train] == ["KD I sing .", "KD I want to go ."]
    assert [r["trg"] for r in amr_train] == ["KD I sing .", "KD I want to go ."]
    assert amr_train[0]["graph_src"] == [[6, 8, ":ARG0", 7]]          # other fields untouched
    assert kd.read_jsonl(os.path.join(out["textamr"], "test.jsonl")) == TEXT_AMR   # human references
    meta = json.load(open(os.path.join(out["textamr"], "meta.json")))
    assert meta["kd"]["teacher"] == "fake" and meta["kd"]["train_rows"] == 2
    assert os.path.basename(out["textamr"]) == "textamr_kd-t"


def test_cache_resume_translates_only_missing(data):
    cache = os.path.join(paths.dataset_dir("plain"), "kd-t_cache.jsonl")
    with open(cache, "w", encoding="utf-8") as f:
        f.write(json.dumps({"src": "I sing .", "hyp": "cached"}, ensure_ascii=False) + "\n")
        f.write('{"src": "broken')                                     # unfinished line from a crash
    teacher = FakeTeacher()
    out = kd.run(run_args(), teacher)
    assert teacher.calls == [["I want to go ."]]
    rows = kd.read_jsonl(os.path.join(out["__src__"], "train.jsonl"))
    assert rows[0]["trg"] == "cached"


def test_joint_length_filter_and_empty_output(data):
    long_vi = " ".join(["dài"] * 80)
    teacher = FakeTeacher({"I sing .": long_vi, "I want to go .": ""})
    out = kd.run(run_args(), teacher)
    for name in ("__src__", "textamr"):
        assert kd.read_jsonl(os.path.join(out[name], "train.jsonl")) == []
    meta = json.load(open(os.path.join(out["__src__"], "meta.json")))
    assert meta["kd"]["dropped_rows"] == 2


def test_teacher_must_return_one_output_per_input(data):
    with pytest.raises(RuntimeError):
        kd.run(run_args(), lambda batch: batch[:-1])


def test_check_test_reports_exact_match(data):
    teacher = FakeTeacher({r["src"]: r["trg"] for r in PLAIN})       # a teacher that memorized the refs
    out = kd.run(run_args(check_test=True), teacher)
    report = json.load(open(os.path.join(out["__src__"], "meta.json")))["kd"]["teacher_on_test"]
    assert report["exact_match_rate"] == 1.0 and report["rows"] == 2
