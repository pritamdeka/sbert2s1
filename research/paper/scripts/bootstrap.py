"""Paired bootstrap CIs for the paper's key contrasts, from saved prediction logits (public tasks only).

For a configuration, the per-item correctness is averaged over its seeds. The metric is chance-normalised
accuracy averaged over (task, question, K) groups within a task, then over tasks. Resampling draws states
with replacement within each task (all questions of a state move together), identically for both sides
of a contrast.

    python paper/scripts/bootstrap.py results/s1_results_public/runs > paper/scripts/contrasts.txt
"""
import gzip
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from analysis_common import ROOT, audit

SEEN = ['pubmedqa', 'scifact', 'healthver', 'ddi', 'hoc', 'ade', 'druglib', 'medline_s1']
HELD = ['pubhealth', 'biosses', 'mtsamples']
B = 2000
RNG = np.random.default_rng(0)
RESULTS = []


def holm(pvalues):
    p = np.asarray(pvalues, float)
    out = np.empty(len(p)); running = 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (len(p)-rank)*p[i])
        out[i] = min(1., running)
    return out.tolist()


@lru_cache(maxsize=96)
def load_items(run_dir, task):
    f = run_dir / 'preds' / 'final' / f'{task}.test.jsonl.gz'
    if not f.exists():
        return None
    out = {}
    for line in gzip.open(f, 'rt'):
        r = json.loads(line)
        z, t = np.array(r['logits']), np.array(r['target'])
        out[(r['id'], r['qid'])] = (float(z.argmax() == t.argmax()), len(t))
    return out


def config_items(runs, root, task):
    """Mean correctness over seeds per item; items must be identical across seeds."""
    per = [load_items(root / r, task) for r in runs]
    per = [p for p in per if p]
    if not per:
        return None
    keys = sorted(per[0])
    return {k: (np.mean([p[k][0] for p in per]), per[0][k][1]) for k in keys}


def state_matrices(items_a, items_b, task):
    """Per task: arrays [n_states, n_groups] of item counts and correct sums for sides a and b."""
    if set(items_a) != set(items_b):
        raise ValueError('Paired configurations have different prediction item sets')
    keys = sorted(items_a)
    clusters = audit()['state_clusters'][task]
    states = sorted({clusters[k[0]] for k in keys})
    groups = sorted({(k[1], items_a[k][1]) for k in keys})
    si = {s: i for i, s in enumerate(states)}
    gi = {g: j for j, g in enumerate(groups)}
    cnt = np.zeros((len(states), len(groups)))
    ca = np.zeros_like(cnt)
    cb = np.zeros_like(cnt)
    for k in keys:
        i, j = si[clusters[k[0]]], gi[(k[1], items_a[k][1])]
        cnt[i, j] += 1
        ca[i, j] += items_a[k][0]
        cb[i, j] += items_b[k][0]
    Ks = np.array([g[1] for g in groups], float)
    return cnt, ca, cb, Ks


def acc_cn(cnt_sum, c_sum, Ks):
    ok = cnt_sum > 0
    acc = np.where(ok, c_sum / np.maximum(cnt_sum, 1), np.nan)
    return np.nanmean((acc - 1 / Ks) / (1 - 1 / Ks), axis=-1)


def contrast(root, a_runs, b_runs, tasks, label):
    mats = {}
    for t in tasks:
        xa, xb = config_items(a_runs, root, t), config_items(b_runs, root, t)
        if xa is not None and xb is not None:
            mats[t] = state_matrices(xa, xb, t)
    tasks = [t for t in tasks if t in mats]
    if not tasks:
        print(f'{label:55s} (no predictions)')
        return
    d0 = np.mean([acc_cn(m[0].sum(0), m[1].sum(0), m[3]) - acc_cn(m[0].sum(0), m[2].sum(0), m[3]) for m in mats.values()])
    per_task = []
    for t in tasks:
        cnt, ca, cb, Ks = mats[t]
        draws = []
        n = cnt.shape[0]
        for start in range(0, B, 128):
            w = RNG.multinomial(n, np.full(n, 1/n), size=min(128, B-start))
            counts = w @ cnt
            draws.extend(acc_cn(counts, w @ ca, Ks) - acc_cn(counts, w @ cb, Ks))
        per_task.append(draws)
    ds = np.mean(per_task, axis=0)
    lo, hi = np.percentile(ds, [2.5, 97.5])
    p = min(1.0, 2 * min((np.sum(ds <= 0)+1)/(B+1), (np.sum(ds >= 0)+1)/(B+1)))
    RESULTS.append(dict(label=label, rq=label.split(':')[0].split()[0], track='seen' if tasks == SEEN else 'heldout',
                        delta=float(d0*100), lo=float(lo*100), hi=float(hi*100), p=float(p),
                        seeds_a=len(a_runs), seeds_b=len(b_runs)))
    print(f'{label:55s} delta {d0 * 100:+6.2f}  95% CI [{lo * 100:+6.2f}, {hi * 100:+6.2f}]  p~{p:.3f}  ({len(tasks)} tasks)', flush=True)


def seeds(root, prefix):
    return [str(p.relative_to(root)) for p in sorted(root.glob(prefix)) if (p / 'preds').exists()]


PAIRS = [('spubmedbert', 'pubmedbert', 'S-PubMedBERT - PubMedBERT'), ('allmpnet', 'mpnet', 'all-MPNet - MPNet'),
         ('mbembed', 'modernbert', 'Nomic-embed - ModernBERT'), ('gtemb', 'modernbert', 'GTE-ModernBERT - ModernBERT'),
         ('bcmbembed', 'bcmb', 'BioClinical-MB-emb - BioClinical-MB')]
# (family, label, side a, side b) -- run-directory globs relative to the runs root; a - b is reported
PFR_CELLS = {'CE': 'rq3/ce/s*', 'Proper': 'rq3/proper/s*', 'PG': 'main/spubmedbert/PFR/s*',
             'PG w/o CE': 'rq3/pg_noce/s*', 'LOO': 'rq3/pg_loo/s*', 'Reparam': 'rq3/reparam/s*'}
C_CELLS = {'CE': 'rq3c/ce/s*', 'Proper': 'rq3c/proper/s*', 'PG': 'main/spubmedbert/C/s*',
           'LOO': 'rq3c/pg_loo/s*', 'Reparam': 'rq3c/reparam/s*'}


def contrast_list():
    out = []
    for arch in ('C', 'PFR'):
        for a, b, n in PAIRS:
            out.append(('RQ1', f'{arch} full: {n}', f'main/{a}/{arch}/s*', f'main/{b}/{arch}/s*'))
    for frac in ('0.02', '0.1'):
        for arch in ('C', 'PFR'):
            for a, b, n in PAIRS:
                out.append(('RQ1', f'{arch} {float(frac) * 100:g}%: {n}', f'lc/{a}/{arch}/f{frac}/s*', f'lc/{b}/{arch}/f{frac}/s*'))
    for obj in ('CE', 'Proper', 'PG', 'LOO', 'Reparam'):
        out.append(('RQ2', f'C - PFR ({obj})', C_CELLS[obj], PFR_CELLS[obj]))
    out.append(('RQ2', 'B - C (PG)', 'bi/spubmedbert/B/s*', 'main/spubmedbert/C/s*'))
    out.append(('RQ2', 'PFR frozen - shared prior (PG)', 'frozen/spubmedbert/PFR/s*', 'main/spubmedbert/PFR/s*'))
    for head, cells in (('PFR', PFR_CELLS), ('C', C_CELLS)):
        pairs = [('CE', 'PG'), ('Reparam', 'PG'), ('LOO', 'PG'), ('LOO', 'Reparam'), ('LOO', 'CE'), ('Proper', 'CE')]
        if head == 'PFR':
            pairs.append(('PG w/o CE', 'PG'))
        for a, b in pairs:
            out.append(('RQ3', f'{head}: {a} - {b}', cells[a], cells[b]))
    out += [('LLM', 'Laya-large FT - S-PubMedBERT C (PG)', 'laya_ft/s*', 'main/spubmedbert/C/s*'),
            ('LLM', 'Laya-large FT - S-PubMedBERT C (CE)', 'laya_ft/s*', 'rq3c/ce/s*'),
            ('LLM', 'Gemma-4-31B - Laya-large FT', 'llm/gemma4_31b', 'laya_ft/s*'),
            ('LLM', 'Gemma-4-31B - S-PubMedBERT C (CE)', 'llm/gemma4_31b', 'rq3c/ce/s*')]
    return out


def main():
    """Resumable: python bootstrap.py RUNS_ROOT [--draws 10000] [--budget SECONDS]."""
    import argparse
    import hashlib
    import time
    global B, RNG
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--draws', type=int, default=10000)
    ap.add_argument('--budget', type=float, default=1e12, help='stop after this many seconds (rerun to resume)')
    a = ap.parse_args()
    root, B, t0 = Path(a.root), a.draws, time.time()
    cache = ROOT / 'analysis' / f'contrasts_cache_{B}.jsonl'
    done = {}
    if cache.exists():
        for line in cache.read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            done[(r['track'], r['label'])] = r
    todo = [(track, tasks, c) for track, tasks in (('seen', SEEN), ('heldout', HELD)) for c in contrast_list()]
    for track, tasks, (fam, label, pa, pb) in todo:
        if (track, label) in done:
            continue
        if time.time() - t0 > a.budget:
            print(f'budget reached: {len(done)}/{len(todo)} contrasts cached; rerun to continue')
            return
        ra, rb = seeds(root, pa), seeds(root, pb)
        if not ra or not rb:
            done[(track, label)] = dict(label=label, rq=fam, track=track, missing=True)
        else:
            RNG = np.random.default_rng(int(hashlib.sha256(f'{track}|{label}'.encode()).hexdigest()[:8], 16))
            n0 = len(RESULTS)
            contrast(root, ra, rb, tasks, label)
            if len(RESULTS) == n0:
                done[(track, label)] = dict(label=label, rq=fam, track=track, missing=True)
            else:
                done[(track, label)] = dict(RESULTS[-1], rq=fam, track=track, label=label)
        with cache.open('a', encoding='utf-8') as f:
            f.write(json.dumps(done[(track, label)]) + '\n')
    rows = [done[(t, c[1])] for t in ('seen', 'heldout') for c in contrast_list() if not done[(t, c[1])].get('missing')]
    for rq in sorted({r['rq'] for r in rows}):
        fam = [r for r in rows if r['rq'] == rq]
        for row, p in zip(fam, holm([r['p'] for r in fam])):
            row['p_holm'] = p
    (ROOT / 'analysis/contrasts.json').write_text(json.dumps(dict(
        draws=B, min_attainable_p=2 / (B + 1), clusters='normalised document text',
        conditional_on='observed tasks and seed-averaged correctness',
        holm_families='within RQ1, RQ2, RQ3 and LLM, across both tracks', results=rows), indent=2), encoding='utf-8')
    lines = [f"{r['rq']} {r['label']} [{r['track']}]: {r['delta']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}], "
             f"p={r['p']:.4f}, Holm={r['p_holm']:.4f}, seeds {r['seeds_a']}/{r['seeds_b']}" for r in rows]
    Path(__file__).with_name('contrasts.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'wrote analysis/contrasts.json ({len(rows)} contrasts, {B} draws)')


if __name__ == '__main__':
    main()
