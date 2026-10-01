"""Convert the PhysioNet credentialed datasets into typed-decision records. Run it only on infrastructure that
your PhysioNet data use agreement allows. Prints counts only, never record text. Output directories are chmod 700.

    python data_build/build_clinical.py --src /path/to/databases --out prepared

Tasks written (record IDs are neutral indices; no MIMIC identifiers are stored):
  mednli           choice {entailment, neutral, contradiction}           train / calib / dev / test
  mimic_trialq     choice {yes, no, not_stated} (boolean questions) +
                   noul "answerable from this note" (all questions)       test
  mimic_clinqa     choice: which passage answers the question (gold + 4 lexical distractors
                   from the same note; TF-IDF, so no encoder is favoured)  test
  mimic_deid       noul per note window: mentions a person's name?       train / calib / test / polysemy
"""
import argparse
import csv
import io
import json
import math
import os
import random
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s1 import schema  # noqa: E402
from s1.common import file_sha256, write_jsonl  # noqa: E402

SEED = 20260929


def find_zip(src, stem):
    hits = sorted(p for p in Path(src).glob('*.zip') if p.name.startswith(stem))
    if not hits:
        raise SystemExit(f'missing {stem}*.zip in {src}')
    return zipfile.ZipFile(hits[0])


def member(z, suffix):
    return next(n for n in z.namelist() if n.endswith(suffix))


def rec(task, split, i, state, qs, gold):
    return schema.validate(dict(id=f'{task}:{split}:{i:06d}', task=task, split=split, state=state, questions=qs,
                                gold={q: {'probabilities': p} for q, p in gold.items()}, meta={}))


def carve(rows, n, seed):
    """Split off a calibration slice: at most n and at most 10% of the rows (never empties train)."""
    n = min(n, max(1, len(rows) // 10))
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    cut = set(idx[:n])
    return [r for i, r in enumerate(rows) if i not in cut], [r for i, r in enumerate(rows) if i in cut]


NLI = {'entailment': 'the hypothesis must be true given the clinical note',
       'neutral': 'the hypothesis may or may not be true given the clinical note',
       'contradiction': 'the hypothesis cannot be true given the clinical note'}


def mednli(src):
    z = find_zip(src, 'mednli')
    out = {}
    for split, f in (('train', 'mli_train_v1.jsonl'), ('dev', 'mli_dev_v1.jsonl'), ('test', 'mli_test_v1.jsonl')):
        rows = []
        for line in z.open(member(z, f)):
            r = json.loads(line)
            q = {'type': 'choice', 'instructions': f'How does this hypothesis relate to the clinical note? Hypothesis: {r["sentence2"]}',
                 'criteria': NLI}
            rows.append(rec('mednli', split, len(rows), r['sentence1'], {'relation': q}, {'relation': {r['gold_label']: 1.0}}))
        out[split] = rows
    out['train'], out['calib'] = carve(out['train'], 400, SEED)
    return out


ANSWER = {'yes': 'the note shows the answer is yes', 'no': 'the note shows the answer is no',
          'not_stated': 'the note does not contain the information needed to answer'}


def mimic_trialq(src):
    z = find_zip(src, 'mimic-iii-ext-synthetic-clinical-trial-questions')
    rows = []
    for r in csv.DictReader(io.TextIOWrapper(z.open(member(z, 'annotated_synthetic_questions.csv')), encoding='utf-8')):
        avail = r['answer_available'].strip() == '1'
        qs = {'answerable': {'type': 'noul', 'instructions': f'Can this question be answered from the information in the note: {r["question"]}'}}
        gold = {'answerable': {'true' if avail else 'false': 1.0}}
        if r['type'] in ('yes', 'na-bool'):
            ans = r['answer'].strip().lower()
            key = 'not_stated' if not avail else ('yes' if ans.startswith('y') else 'no' if ans.startswith('n') else None)
            if key:
                qs['answer'] = {'type': 'choice', 'instructions': r['question'], 'criteria': ANSWER}
                gold['answer'] = {key: 1.0}
        rows.append(rec('mimic_trialq', 'test', len(rows), r['text'], qs, gold))
    return {'test': rows}


def _segments(text, min_len=40, max_len=400):
    parts = [p.strip() for p in re.split(r'\n\s*\n|\n(?=[A-Z][A-Za-z /]+:)|(?<=[.!?])\s+(?=[A-Z])', text) if p.strip()]
    segs, cur, start = [], '', 0
    pos = 0
    for p in parts:
        at = text.find(p, pos)
        pos = at + len(p) if at >= 0 else pos
        if not cur:
            start = max(0, at)
        cur = (cur + ' ' + p).strip()
        if len(cur) >= min_len:
            segs.append((start, pos, cur[:max_len]))
            cur = ''
    if cur:
        segs.append((start, len(text), cur[:max_len]))
    return segs


def _tfidf_rank(query, docs):
    tokenize = lambda s: re.findall(r'[a-z0-9]+', s.lower())  # noqa: E731
    dtoks = [Counter(tokenize(d)) for d in docs]
    df = Counter(t for d in dtoks for t in d)
    n = len(docs)
    idf = {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}
    q = Counter(tokenize(query))

    def vec(c):
        return {t: v * idf.get(t, 1.0) for t, v in c.items()}
    qv = vec(q)
    qn = math.sqrt(sum(v * v for v in qv.values())) or 1.0
    scores = []
    for c in dtoks:
        dv = vec(c)
        dn = math.sqrt(sum(v * v for v in dv.values())) or 1.0
        scores.append(sum(qv.get(t, 0) * v for t, v in dv.items()) / (qn * dn))
    return sorted(range(n), key=lambda i: -scores[i])


def mimic_clinqa(src, k=5):
    z = find_zip(src, 'annotated-question-answer-pairs-for-clinical-notes')
    data = json.load(z.open(member(z, 'test.final.json')))
    rows, skipped = [], 0
    for doc in data['data']:
        for para in doc['paragraphs']:
            segs = _segments(para['context'])
            for qa in para['qas']:
                a = qa['answers'][0]
                gold = next((i for i, (s, e, _) in enumerate(segs) if s <= a['answer_start'] < e), None)
                if gold is None or len(segs) < k:
                    skipped += 1
                    continue
                ranked = [i for i in _tfidf_rank(qa['question'], [t for _, _, t in segs]) if i != gold][:k - 1]
                opts = sorted(ranked + [gold])
                crit = {f'passage_{j + 1}': segs[i][2] for j, i in enumerate(opts)}
                q = {'type': 'choice', 'instructions': f'Which passage of the note answers: {qa["question"]}', 'criteria': crit}
                state = 'Candidate passages are listed as options; choose the one that answers the question.'
                rows.append(rec('mimic_clinqa', 'test', len(rows), state, {'passage': q},
                                {'passage': {f'passage_{opts.index(gold) + 1}': 1.0}}))
    print(f'mimic_clinqa: skipped {skipped} questions without a locatable passage', flush=True)
    return {'test': rows}


def _windows(note, spans, size=700):
    lines = note.split('\n')
    wins, cur, start, pos = [], '', 0, 0
    for ln in lines:
        if cur and len(cur) + len(ln) + 1 > size:
            wins.append((start, pos, cur))
            cur, start = '', pos
        cur = cur + ln + '\n'
        pos += len(ln) + 1
    if cur.strip():
        wins.append((start, pos, cur))
    # A window is positive when any labelled name span overlaps it.
    return [(w, any(ps < e and pe > s for ps, pe in spans)) for s, e, w in wins if w.strip()]


def mimic_deid(src, n_train_notes=300, test_variants=2):
    z = find_zip(src, 'annotated-mimic-iv-discharge-summaries')
    root = z.namelist()[0].split('/')[0]

    def load(path):
        return [json.loads(l) for l in z.open(f'{root}/{path}')]

    def convert(inputs, labels, split, notes_filter=None):
        spans = {}
        for r in labels:
            spans.setdefault(tuple(r['ID']), []).append(tuple(r['position']))
        rows = []
        q = {'type': 'noul', 'instructions': "Does this passage mention a person's name (a patient, relative or clinician)?"}
        for r in inputs:
            key = tuple(r['ID'])
            if notes_filter and not notes_filter(key):
                continue
            for w, has in _windows(r['note'], spans.get(key, [])):
                rows.append(rec('mimic_deid', split, len(rows), w, {'has_name': dict(q)}, {'has_name': {'true' if has else 'false': 1.0}}))
        return rows
    rng = random.Random(SEED)
    tr_in = load('finetune/input/inputs-clinical+diverse.jsonl')
    keep = {tuple(r['ID']) for r in rng.sample(tr_in, min(n_train_notes, len(tr_in)))}
    train = convert(tr_in, load('finetune/input/labels-clinical+diverse.jsonl'), 'train', lambda k: k in keep)
    te_in = load('finetune/input/inputs-test.jsonl')
    by_base = {}
    for r in te_in:
        by_base.setdefault(r['ID'][0], []).append(tuple(r['ID']))
    keep_t = {k for v in by_base.values() for k in sorted(v)[:test_variants]}
    test = convert(te_in, load('finetune/input/labels-test.jsonl'), 'test', lambda k: k in keep_t)
    poly = convert(load('polysemy/input/polysemies-input.jsonl'), load('polysemy/input/polysemies-label.jsonl'), 'polysemy')
    train, calib = carve(train, 400, SEED)
    return {'train': train, 'calib': calib, 'test': test, 'polysemy': poly}


# mimic_clinqa is implemented but not in the v1 default: its options are the passages themselves, so it
# measures passage matching more than a decision over a state, and 36 notes give very wide intervals.
TASKS = ['mednli', 'mimic_trialq', 'mimic_deid']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', default=str(Path(__file__).resolve().parents[1] / 'prepared'))
    ap.add_argument('--tasks', nargs='*', default=TASKS)
    a = ap.parse_args()
    out = Path(a.out)
    manifest_path = out / 'manifest_clinical.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for t in a.tasks:
        splits = globals()[t](a.src)
        (out / t).mkdir(parents=True, exist_ok=True)
        os.chmod(out / t, 0o700)
        for split, rows in splits.items():
            for i, r in enumerate(rows):
                r['id'], r['split'] = f'{t}:{split}:{i:06d}', split
            path = out / t / f'{split}.jsonl'
            write_jsonl(path, rows)
            os.chmod(path, 0o600)
            labels = Counter(max(g['probabilities'], key=g['probabilities'].get) for r in rows for g in r['gold'].values())
            manifest[f'{t}/{split}'] = dict(states=len(rows), decisions=sum(labels.values()), sha256=file_sha256(path),
                                            labels=dict(labels), credentialed=True)
            print(f'{t}/{split}: {len(rows)} states, {sum(labels.values())} decisions, labels {dict(labels)}', flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))


if __name__ == '__main__':
    main()
