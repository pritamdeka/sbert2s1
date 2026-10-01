import gzip
import json
import sys
from functools import lru_cache
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from s1.temperature_lookup import temperature_for

SEEN = ['pubmedqa','scifact','healthver','ddi','hoc','ade','druglib','medline_s1']
HELD = ['pubhealth','biosses','mtsamples']


@lru_cache(maxsize=1)
def audit():
    return json.loads((ROOT/'analysis/data_audit.json').read_text(encoding='utf-8'))


@lru_cache(maxsize=4)
def predictions(run, task, filtered=False):
    f = Path(run)/'preds/final'/f'{task}.test.jsonl.gz'
    if not f.exists(): return None
    excluded = set(audit()['test_exclusions'].get(task, [])) if filtered else set()
    rows = {}
    with gzip.open(f, 'rt', encoding='utf-8') as stream:
        for r in map(json.loads, stream):
            if r['id'] not in excluded:
                rows[(r['id'],r['qid'])] = r
    return rows


def macro_accuracy(rows, correct):
    """Mean over (question ID, K) groups in task, then tasks, identical to tables."""
    groups = {}
    for r, c in zip(rows, correct):
        k = len(r['target']); task = r['task']
        groups.setdefault((task,r['qid'],k), []).append((float(c)-1/k)/(1-1/k))
    tasks = {}
    for (task,_,_), values in groups.items():
        tasks.setdefault(task, []).append(np.mean(values))
    return float(np.mean([np.mean(v) for v in tasks.values()]))
