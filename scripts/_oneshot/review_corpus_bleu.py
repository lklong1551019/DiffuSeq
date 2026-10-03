"""Corpus BLEU + repetition diagnostics for DiffuSeq decode outputs (review of 2026-10-03).

Run from the repo root, CPU only, no third-party dependency:

    python scripts/_oneshot/review_corpus_bleu.py <decode.json> <test.jsonl> [<decode.json> <test.jsonl> ...]

Notes:
- The tokenizer is a re-implementation of sacreBLEU's 13a tokenizer; scores can deviate
  slightly from sacreBLEU. Replace with sacreBLEU once it is installed (see
  docs/plans/PENDING.md).
- Hypotheses are scored against the RAW references from test.jsonl (field `trg`), not
  against the mBERT-decoded references stored in the decode file.
- Row order of the decode file must match test.jsonl; the script asserts equal row counts.
"""
import collections
import json
import math
import re
import sys


def tok13a(s):
    s = re.sub(r'([\{-\~\[-\` -\&\(-\+\:-\@\/])', r' \1 ', s)
    s = re.sub(r'([^0-9])([\.,])', r'\1 \2 ', s)
    s = re.sub(r'([\.,])([^0-9])', r' \1 \2', s)
    s = re.sub(r'([0-9])(-)', r'\1 \2 ', s)
    return s.split()


def ngrams(tokens, n):
    return collections.Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(hyps, refs):
    match, total, hyp_len, ref_len = [0] * 4, [0] * 4, 0, 0
    for h, r in zip(hyps, refs):
        h, r = tok13a(h), tok13a(r)
        hyp_len += len(h)
        ref_len += len(r)
        for n in range(1, 5):
            hn, rn = ngrams(h, n), ngrams(r, n)
            match[n - 1] += sum(min(c, rn[g]) for g, c in hn.items())
            total[n - 1] += max(len(h) - n + 1, 0)
    if min(match) == 0:
        return 0.0, [0.0] * 4, 0.0, hyp_len, ref_len
    log_p = sum(math.log(match[i] / total[i]) for i in range(4)) / 4
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
    precisions = [round(100 * match[i] / total[i], 1) for i in range(4)]
    return 100 * bp * math.exp(log_p), precisions, bp, hyp_len, ref_len


def clean(s):
    s = s.replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '')
    return re.sub(r'\s+', ' ', s).strip()


def adjacent_repeat_rate(s):
    t = s.split()
    return sum(1 for a, b in zip(t, t[1:]) if a == b) / max(len(t) - 1, 1)


def main(argv):
    for decode_path, test_path in zip(argv[0::2], argv[1::2]):
        out = [json.loads(line) for line in open(decode_path, encoding='utf-8')]
        refs_raw = [json.loads(line)['trg'] for line in open(test_path, encoding='utf-8')]
        assert len(out) == len(refs_raw), (len(out), len(refs_raw))
        hyps = [clean(o['recover']) for o in out]
        refs_decoded = [clean(o['reference']) for o in out]
        bleu, prec, bp, hl, rl = corpus_bleu(hyps, refs_raw)
        print(decode_path)
        print(f'  corpus BLEU vs raw refs: {bleu:.2f}  p1-4={prec}  BP={bp:.3f}  hyp_len={hl}  ref_len={rl}')
        print(f'  corpus BLEU vs decoded refs: {corpus_bleu(hyps, refs_decoded)[0]:.2f}')
        hyp_rep = sum(map(adjacent_repeat_rate, hyps)) / len(hyps)
        ref_rep = sum(map(adjacent_repeat_rate, refs_decoded)) / len(refs_decoded)
        print(f'  adjacent duplicate-token rate: hyp {hyp_rep:.3f}  ref {ref_rep:.3f}')


if __name__ == '__main__':
    if len(sys.argv) < 3 or len(sys.argv) % 2 == 0:
        sys.exit(__doc__)
    main(sys.argv[1:])
