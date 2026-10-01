"""Audit historical public splits and write separate document-disjoint rerun inputs.

Historical inputs and results are immutable. Test masks remove exact repeated inputs
seen in training/calibration; they do NOT remove calibration bias in old checkpoints.
New rerun splits reserve test, then dev, then calibration, then training documents
globally across public task families. IDs remain stable for traceability.
"""
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s1.split_integrity import fingerprint, input_key, state_key


def main():
    source = ROOT / 'prepared'
    dest = ROOT / 'prepared_clean'
    dest.mkdir(exist_ok=True)
    def records(path):
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                yield json.loads(line)
    data = {}
    for f in sorted(source.glob('*/*.jsonl')):
        if f.parent.name.startswith('_'):
            continue
        data[(f.parent.name, f.stem)] = f
    train_inputs = {input_key(r) for (t, s), path in data.items() if s == 'train' for r in records(path)}
    calib_inputs = {input_key(r) for (t, s), path in data.items() if s == 'calib' for r in records(path)}
    report, exclusions, clusters = [], {}, {}
    for (task, split), path in data.items():
        if split not in ('calib', 'test'):
            continue
        rows = list(records(path))
        tr = sum(input_key(r) in train_inputs for r in rows)
        train_states = {state_key(r) for r in records(data[(task, 'train')])} if (task,'train') in data else set()
        excluded, seen = [], set()
        for r in rows:
            k = input_key(r)
            if split == 'test' and (k in train_inputs or k in calib_inputs or k in seen):
                excluded.append(r['id'])
            seen.add(k)
        report.append(dict(task=task, split=split, records=len(rows),
                           exact_inputs_in_train=tr,
                           same_state_in_train=sum(state_key(r) in train_states for r in rows),
                           sensitivity_excluded=len(excluded)))
        if split == 'test':
            exclusions[task] = excluded
            clusters[task] = {r['id']: state_key(r) for r in rows}
    # All evaluations share first priority. Multiple tasks can ask about the same
    # evaluation document; only lower-priority splits are excluded.
    higher, clean_manifest = set(), {}
    for split in ('test', 'dev', 'calib', 'train'):
        this_states = set()
        for (task, sp), path in data.items():
            if sp != split:
                continue
            rows = list(records(path))
            variants = defaultdict(set)
            for r in rows:
                variants[input_key(r)].add(fingerprint(r['gold']))
            kept, seen = [], set()
            for r in rows:
                st, k = state_key(r), input_key(r)
                if st in higher or k in seen or len(variants[k]) > 1:
                    continue
                kept.append(r); seen.add(k); this_states.add(st)
            f = dest / task / (split + '.jsonl'); f.parent.mkdir(exist_ok=True)
            with f.open('w', encoding='utf-8') as stream:
                for r in kept:
                    stream.write(json.dumps(r, ensure_ascii=False) + '\n')
            import hashlib
            clean_manifest[f'{task}/{split}'] = dict(states=len(kept), decisions=sum(len(r['gold']) for r in kept),
                removed=len(rows)-len(kept), original_states=len(rows), sha256=hashlib.sha256(f.read_bytes()).hexdigest())
        higher |= this_states
    # Retrieval files are reference evaluation resources, not training records.
    if not (dest / '_retrieval').exists():
        shutil.copytree(source / '_retrieval', dest / '_retrieval')
    for name in ('manifest_clinical.json', 'eval_pmids.json'):
        if (source / name).exists(): shutil.copy2(source / name, dest / name)
    (dest / 'manifest.json').write_text(json.dumps(clean_manifest, indent=2), encoding='utf-8')
    result = dict(policy='Historical input audit; cleaned splits reserve test > dev > calib > train globally by normalised text.',
                  report=report, test_exclusions=exclusions, state_clusters=clusters, clean_manifest=clean_manifest)
    (ROOT / 'analysis/data_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    lines = ['# Public-data integrity audit', '',
             'Counts are records/states, not individual questions. Historical results are not retrained.', '',
             '| Task | Split | Records | Exact inputs in train | Same text in train | Excluded from test sensitivity |',
             '|---|---|---:|---:|---:|---:|']
    for r in report:
        lines.append('| ' + ' | '.join(str(r[k]) for k in ('task','split','records','exact_inputs_in_train','same_state_in_train','sensitivity_excluded')) + ' |')
    lines += ['', 'Same-text overlap with different questions is reported separately from exact input duplication.',
              'Sensitivity excludes exact inputs in train/calibration and repeated test inputs. Existing temperatures remain unchanged.',
              'prepared_clean is for NEW training and calibration; applying it cannot retroactively repair historical checkpoints.',
              'Document identities normalise whitespace; this is not a semantic near-duplicate or patient-level audit.']
    (ROOT / 'analysis/DATA_AUDIT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('Audited',len(data),'public splits; wrote clean rerun inputs and sensitivity masks.')


if __name__ == '__main__':
    main()
