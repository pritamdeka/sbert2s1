"""Patch-8 descriptive results: 2 heads x objectives on S-PubMedBERT (mean +- sd over 3 seeds).
Aggregation identical to the paper tables: metric averaged over question groups within a task, then over tasks.

    python analysis/patch8/describe_patch8.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[2]                       # .../system_one_biomedical_sbert
PUB = PROJ / 'results' / 's1_results_public' / 'runs'
P8 = PROJ / 'results' / 's1_results_patch8' / 'runs'
TRACKS = {'seen': ['pubmedqa', 'scifact', 'healthver', 'ddi', 'hoc', 'ade', 'druglib', 'medline_s1'],
          'heldout': ['pubhealth', 'biosses', 'mtsamples'], 'knowledge': ['medqa', 'medmcqa', 'mmlu_med'],
          'clinical': ['mednli', 'mimic_trialq', 'mimic_deid']}
CELLS = {  # (head, objective) -> run directories (3 seeds each)
    ('PFR', 'CE'): [PUB / f'rq3/ce/s{s}' for s in range(3)],
    ('PFR', 'Proper'): [PUB / f'rq3/proper/s{s}' for s in range(3)],
    ('PFR', 'PG (released)'): [PUB / f'main/spubmedbert/PFR/s{s}' for s in range(3)],
    ('PFR', 'PG w/o CE'): [PUB / f'rq3/pg_noce/s{s}' for s in range(3)],
    ('PFR', 'PG-LOO (unbiased)'): [P8 / f'rq3/pg_loo/s{s}' for s in range(3)],
    ('PFR', 'Reparam'): [PUB / f'rq3/reparam/s{s}' for s in range(3)],
    ('C', 'CE'): [P8 / f'rq3c/ce/s{s}' for s in range(3)],
    ('C', 'Proper'): [P8 / f'rq3c/proper/s{s}' for s in range(3)],
    ('C', 'PG (released)'): [PUB / f'main/spubmedbert/C/s{s}' for s in range(3)],
    ('C', 'PG-LOO (unbiased)'): [P8 / f'rq3c/pg_loo/s{s}' for s in range(3)],
    ('C', 'Reparam'): [P8 / f'rq3c/reparam/s{s}' for s in range(3)],
}


def track_of(task):
    return next((k for k, v in TRACKS.items() if task in v), None)


def run_metrics(run):
    m = json.loads((run / 'metrics_final.json').read_text())
    rows, perm = [], []
    for split, v in m.items():
        if split == 'retention' or v.get('missing'):
            continue
        task = split.split('/')[0]
        for g, x in v.get('groups', {}).items():
            rows.append(dict(task=task, track=track_of(task), acc=x['raw']['acc_norm'], ece=x['raw']['ece'],
                             ece_ts=(x.get('ts') or {}).get('ece'), nll_ts=(x.get('ts') or {}).get('nll'),
                             mae=x['raw'].get('mae')))
        if v.get('permutation') and track_of(task) == 'seen':
            perm.append(v['permutation']['flip_rate'])
    d = pd.DataFrame(rows)
    out = {}
    for tr in TRACKS:
        t = d[d.track == tr].groupby('task')[['acc', 'ece', 'ece_ts', 'nll_ts']].mean()
        out[f'{tr}_acc'] = t.acc.mean()
        if tr == 'seen':
            out.update(seen_ece=t.ece.mean(), seen_ece_ts=t.ece_ts.mean(), seen_nll_ts=t.nll_ts.mean())
    out['druglib_mae'] = d[d.task == 'druglib'].mae.mean()
    out['perm_flip'] = float(np.mean(perm)) if perm else np.nan
    ret = m.get('retention')
    out['scifact_ndcg10'] = ret['scifact']['ndcg10'] if ret else np.nan
    log = run / 'train_log.jsonl'
    gv = [json.loads(l).get('grad_var') for l in log.read_text().splitlines()] if log.exists() else []
    gv = [x for x in gv if x is not None]
    out['logged_grad_var'] = float(np.mean(gv)) if gv else np.nan
    return out


def main():
    rows = []
    for (head, obj), runs in CELLS.items():
        for r in runs:
            if not (r / 'metrics_final.json').exists():
                print('MISSING', r)
                continue
            rows.append(dict(head=head, objective=obj, run=str(r.relative_to(PROJ / 'results')), **run_metrics(r)))
    df = pd.DataFrame(rows)
    df.to_csv(HERE / 'per_run.csv', index=False)
    pct = ['seen_acc', 'heldout_acc', 'clinical_acc', 'knowledge_acc', 'seen_ece', 'seen_ece_ts', 'perm_flip']
    g = df.groupby(['head', 'objective'], sort=False)
    lines = ['| Head | Objective | Seen acc | Held-out | Clinical | Knowledge | ECE raw | ECE TS | NLL TS | MAE score | Perm flip % | n |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|']
    summ = []
    for (h, o), x in g:
        def f(c, scale=100, nd=1):
            v = x[c].dropna()
            return '--' if v.empty else f'{v.mean() * scale:.{nd}f} ± {v.std(ddof=1) * scale:.{nd}f}' if len(v) > 1 else f'{v.mean() * scale:.{nd}f}'
        lines.append(f'| {h} | {o} | {f("seen_acc")} | {f("heldout_acc")} | {f("clinical_acc")} | {f("knowledge_acc")} | '
                     f'{f("seen_ece")} | {f("seen_ece_ts")} | {f("seen_nll_ts", 1, 3)} | {f("druglib_mae", 1, 3)} | '
                     f'{f("perm_flip")} | {len(x)} |')
        summ.append(dict(head=h, objective=o, **{c: x[c].mean() for c in df.columns if c not in ('head', 'objective', 'run')},
                         seen_acc_sd=x.seen_acc.std(ddof=1)))
    pd.DataFrame(summ).to_csv(HERE / 'summary.csv', index=False)
    (HERE / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    r = df[df.scifact_ndcg10.notna()][['run', 'scifact_ndcg10']]
    print('\nretention (seed 0 runs with retention):'); print(r.to_string(index=False))


if __name__ == '__main__':
    main()
