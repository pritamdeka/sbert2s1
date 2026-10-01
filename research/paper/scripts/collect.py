"""Flatten every run's metrics into one CSV (one row per run x eval split x question group).

    python paper/scripts/collect.py results/s1_results_public   ->  paper/scripts/metrics_long.csv
"""
import json
import sys
from pathlib import Path

import pandas as pd

GROUNDED_SEEN = ['pubmedqa', 'scifact', 'healthver', 'ddi', 'hoc', 'ade', 'druglib', 'medline_s1']
HELD_OUT = ['pubhealth', 'biosses', 'mtsamples']
KNOWLEDGE = ['medqa', 'medmcqa', 'mmlu_med']
CLINICAL = ['mednli', 'mimic_trialq', 'mimic_deid']
GENERAL = ['typed_decisions']


def track(task):
    for name, tasks in (('seen', GROUNDED_SEEN), ('heldout', HELD_OUT), ('knowledge', KNOWLEDGE),
                        ('clinical', CLINICAL), ('general', GENERAL)):
        if task in tasks:
            return name
    return 'other'


def parse_run(run_dir, root):
    rid = str(run_dir.relative_to(root / 'runs')).replace('\\', '/')
    cfg = json.loads((run_dir / 'config.json').read_text()).get('config', {}) if (run_dir / 'config.json').exists() else {}
    base = dict(run=rid, encoder=cfg.get('encoder', rid.split('/')[1] if rid.startswith('llm') else None),
                arch=cfg.get('arch', 'LLM' if rid.startswith('llm') else None), objective=cfg.get('objective'),
                seed=cfg.get('seed'), frac=cfg.get('frac', 1.0), prior=cfg.get('prior'), w_ce=cfg.get('w_ce'),
                family=rid.split('/')[0])
    rows = []
    for tag in ('final', 'init'):
        f = run_dir / f'metrics_{tag}.json'
        if not f.exists():
            continue
        m = json.loads(f.read_text())
        for split, v in m.items():
            if split == 'retention':
                for r, x in v.items():
                    rows.append(dict(base, tag=tag, split=f'retention/{r}', task=f'ret_{r}', metric_ndcg10=x['ndcg10']))
                continue
            if v.get('missing'):
                continue
            task = split.split('/')[0]
            groups = [('std', v.get('groups', {}))] + [(f'long_{k}', g) for k, g in (v.get('long') or {}).items()]
            for variant, gs in groups:
                for g, x in gs.items():
                    row = dict(base, tag=tag, split=split, task=task, track=track(task), variant=variant, group=g,
                               qtype=x['qtype'], T=x.get('T'))
                    row.update({f'raw_{k}': val for k, val in x['raw'].items()})
                    row.update({f'ts_{k}': val for k, val in (x.get('ts') or {}).items()})
                    rows.append(row)
            if v.get('permutation') and tag == 'final':
                rows.append(dict(base, tag=tag, split=split, task=task, track=track(task), variant='perm',
                                 perm_flip=v['permutation']['flip_rate'], perm_js=v['permutation']['mean_js']))
            if 'items_per_s' in v:
                rows.append(dict(base, tag=tag, split=split, task=task, track=track(task), variant='speed',
                                 items_per_s=v['items_per_s']))
    return rows


def main():
    root = Path(sys.argv[1])
    rows = []
    for cfg in sorted((root / 'runs').rglob('metrics_final.json')):
        rows += parse_run(cfg.parent, root)
    df = pd.DataFrame(rows)
    out = Path(__file__).with_name('metrics_long.csv')
    df.to_csv(out, index=False)
    print(len(df), 'rows,', df.run.nunique(), 'runs ->', out)


if __name__ == '__main__':
    main()
