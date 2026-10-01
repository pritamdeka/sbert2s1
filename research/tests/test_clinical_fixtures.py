"""Test build_clinical.py on SYNTHETIC zips with the same structure as the PhysioNet releases.
No real data is involved (the real files are never read by tests).

    python tests/test_clinical_fixtures.py
"""
import csv
import io
import json
import random
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TMP = Path(tempfile.mkdtemp(prefix='s1clin_'))
rng = random.Random(0)
WORDS = 'patient admitted with chest pain dyspnea fever cough history of hypertension diabetes discharged stable'.split()


def text(n):
    return ' '.join(rng.choice(WORDS) for _ in range(n)).capitalize() + '.'


def note(lines=40):
    return '\n'.join(text(rng.randint(5, 15)) for _ in range(lines))


def mednli(z):
    root = 'mednli-a-natural-language-inference-dataset-for-the-clinical-domain-1.0.0'
    for f, n in (('mli_train_v1.jsonl', 30), ('mli_dev_v1.jsonl', 9), ('mli_test_v1.jsonl', 9)):
        rows = [json.dumps(dict(sentence1=text(12), sentence2=text(6), gold_label=['entailment', 'neutral', 'contradiction'][i % 3],
                                pairID=str(i))) for i in range(n)]
        z.writestr(f'{root}/{f}', '\n'.join(rows) + '\n')


def trialq(z):
    root = 'mimic-iii-ext-synthetic-clinical-trial-questions-1.0.0'
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=['subject_id', 'hadm_id', 'question', 'answer_available', 'answer', 'difficulty',
                                        'text', 'type', 'same_question', 'same_answer', 'changed'])
    w.writeheader()
    for i, t in enumerate(['yes', 'yes', 'na-bool', 'numeric', 'na-numeric', 'yes']):
        avail = '0' if t.startswith('na') else '1'
        w.writerow(dict(subject_id=i, hadm_id=i, question='Did the patient ' + text(4), answer_available=avail,
                        answer='' if avail == '0' else ('Yes' if i % 2 else 'No') if t == 'yes' else '12', difficulty=3,
                        text=note(), type=t, same_question=1, same_answer=1, changed=0))
    z.writestr(f'{root}/annotated_synthetic_questions.csv', buf.getvalue())


def deid(z):
    root = 'annotated-mimic-iv-discharge-summaries-for-a-study-on-deidentification-of-names-1.0'

    def notes(prefix, base_ids, variants):
        inputs, labels = [], []
        for b in base_ids:
            for v in range(variants):
                body = note(30)
                name = 'Alex Smith'
                pos = body.find('\n', 200) + 1
                body = body[:pos] + name + ' ' + body[pos:]
                inputs.append(dict(ID=[b, v, 0], note=body))
                labels.append(dict(ID=[b, v, 0], position=[pos, pos + len(name)], name=[name, 'Alex', 'Smith']))
        z.writestr(f'{root}/{prefix}-input.jsonl' if 'poly' in prefix else f'{root}/{prefix}',
                   '\n'.join(json.dumps(r) for r in inputs) + '\n')
        return labels
    lab = notes('finetune/input/inputs-clinical+diverse.jsonl', [1, 2], 6)
    z.writestr(f'{root}/finetune/input/labels-clinical+diverse.jsonl', '\n'.join(json.dumps(r) for r in lab) + '\n')
    lab = notes('finetune/input/inputs-test.jsonl', [10, 11, 12], 3)
    z.writestr(f'{root}/finetune/input/labels-test.jsonl', '\n'.join(json.dumps(r) for r in lab) + '\n')
    lab = notes('polysemy/input/polysemies', [10, 11], 2)
    z.writestr(f'{root}/polysemy/input/polysemies-label.jsonl', '\n'.join(json.dumps(r) for r in lab) + '\n')


def main():
    src = TMP / 'zips'
    src.mkdir()
    for name, fn in (('mednli-a-natural-language-inference-dataset-for-the-clinical-domain-1.0.0.zip', mednli),
                     ('mimic-iii-ext-synthetic-clinical-trial-questions-1.0.0.zip', trialq),
                     ('annotated-mimic-iv-discharge-summaries-for-a-study-on-deidentification-of-names-1.0.zip', deid)):
        with zipfile.ZipFile(src / name, 'w') as z:
            fn(z)
    out = TMP / 'prepared'
    p = subprocess.run([sys.executable, 'data_build/build_clinical.py', '--src', str(src), '--out', str(out)],
                       cwd=ROOT, capture_output=True, text=True)
    print(p.stdout[-2000:], p.stderr[-2000:])
    assert p.returncode == 0
    m = json.loads((out / 'manifest_clinical.json').read_text())
    assert m['mednli/test']['states'] == 9 and m['mednli/train']['states'] == 27 and m['mednli/calib']['states'] == 3
    assert m['mimic_trialq/test']['decisions'] == 6 + 3 + 1       # 6 answerable + 3 yes-type answers + 1 na-bool answer
    assert m['mimic_deid/train']['states'] > 0 and m['mimic_deid/calib']['states'] > 0
    # 3 base notes x 2 kept variants x 1 inserted name = 6 positive windows
    assert m['mimic_deid/test']['labels'].get('true', 0) == 6 and m['mimic_deid/polysemy']['states'] > 0
    for line in (out / 'mimic_trialq/test.jsonl').read_text().splitlines():
        r = json.loads(line)
        assert 'hadm_id' not in json.dumps(r['meta']) and r['id'].startswith('mimic_trialq:test:')
    print('PASS clinical converters on synthetic fixtures')
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == '__main__':
    main()
