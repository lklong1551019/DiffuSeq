"""
eval_bleu.py — score DiffuSeq decode files with corpus sacreBLEU (Evaluation Rule in .agents/AGENTS.md).

Hypotheses ("recover") are cleaned of [CLS]/[SEP]/[PAD] and scored against the RAW references of
the test file (field "trg"), not the mBERT-decoded references stored in the decode file.

One decode file   -> BLEU, chrF, adjacent duplicate-token rate, length ratio.
Several files     -> scores per file, then MBR: per row, the candidate with the highest mean
                     sentence chrF against the other candidates (one file per seed).

Usage (repo root, CPU):
    python scripts/eval_bleu.py --test_file datasets/docAMR/<variant>/test.jsonl \
        --decode generation_outputs/<run>/<ckpt>.samples/seed110_*.json [more seed files ...]
    # bidirectional test files: add --direction AMR_TO_TEXT (rows the decoder was filtered to)
"""

import argparse
import json
import re

import sacrebleu


def clean(s):
    s = s.replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '')
    return re.sub(r'\s+', ' ', s).strip()


def adjacent_repeat_rate(texts):
    """Mean over rows of the fraction of adjacent whitespace tokens that repeat the previous one."""
    rates = []
    for s in texts:
        t = s.split()
        rates.append(sum(1 for a, b in zip(t, t[1:]) if a == b) / max(len(t) - 1, 1))
    return sum(rates) / max(len(rates), 1)


def load_references(path, direction=None):
    refs = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if direction is not None and row.get('direction') not in (None, direction):
                continue
            refs.append(row['trg'])
    return refs


def load_hypotheses(path):
    with open(path, encoding='utf-8') as f:
        return [clean(json.loads(line)['recover']) for line in f]


def score(hyps, refs):
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    chrf = sacrebleu.corpus_chrf(hyps, [refs])
    hyp_len = sum(len(h.split()) for h in hyps)
    ref_len = sum(len(r.split()) for r in refs)
    return {
        'bleu': round(bleu.score, 2),
        'bleu_signature': str(bleu),
        'chrf': round(chrf.score, 2),
        'repeat_rate': round(adjacent_repeat_rate(hyps), 4),
        'length_ratio': round(hyp_len / max(ref_len, 1), 3),
        'rows': len(hyps),
    }


def mbr_select(candidate_lists):
    """Per row, pick the candidate with the highest mean sentence chrF against the others."""
    chosen = []
    for cands in zip(*candidate_lists):
        if len(cands) == 1:
            chosen.append(cands[0])
            continue
        best, best_score = cands[0], -1.0
        for i, c in enumerate(cands):
            others = [o for j, o in enumerate(cands) if j != i]
            s = sum(sacrebleu.sentence_chrf(c, [o]).score for o in others) / len(others)
            if s > best_score:
                best, best_score = c, s
        chosen.append(best)
    return chosen


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--test_file', required=True)
    p.add_argument('--decode', nargs='+', required=True, help='decode .json files (one per seed)')
    p.add_argument('--direction', default=None, help='keep only test rows of this direction')
    args = p.parse_args()

    refs = load_references(args.test_file, args.direction)
    all_hyps = []
    for path in args.decode:
        hyps = load_hypotheses(path)
        if len(hyps) != len(refs):
            raise SystemExit(f'{path}: {len(hyps)} rows, test file has {len(refs)} — not the same test set')
        all_hyps.append(hyps)
        print(json.dumps({'file': path, **score(hyps, refs)}, ensure_ascii=False))
    if len(all_hyps) > 1:
        print(json.dumps({'file': f'MBR over {len(all_hyps)} files', **score(mbr_select(all_hyps), refs)},
                         ensure_ascii=False))


if __name__ == '__main__':
    main()
