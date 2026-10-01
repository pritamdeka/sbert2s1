"""Paired, document-clustered bootstrap for the patch-8 contrasts (reuses paper/scripts/bootstrap.py).
10,000 draws (fixed RNG seed); Holm adjustment within each family (RQ2 head, RQ3 objective) across both tracks.

    python analysis/patch8/contrasts_patch8.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V2 = HERE.parents[1]
PROJ = V2.parent
sys.path.insert(0, str(V2 / 'paper' / 'scripts'))
import bootstrap as BS  # noqa: E402

BS.B = 10000
BS.RNG = np.random.default_rng(20261001)
PUB = PROJ / 'results' / 's1_results_public' / 'runs'
P8 = PROJ / 'results' / 's1_results_patch8' / 'runs'
S = lambda base, rel: [str(base / rel / f's{i}') for i in range(3)]  # noqa: E731
PFR = {'CE': S(PUB, 'rq3/ce'), 'Proper': S(PUB, 'rq3/proper'), 'PG': S(PUB, 'main/spubmedbert/PFR'),
       'LOO': S(P8, 'rq3/pg_loo'), 'Reparam': S(PUB, 'rq3/reparam')}
C = {'CE': S(P8, 'rq3c/ce'), 'Proper': S(P8, 'rq3c/proper'), 'PG': S(PUB, 'main/spubmedbert/C'),
     'LOO': S(P8, 'rq3c/pg_loo'), 'Reparam': S(P8, 'rq3c/reparam')}
CONTRASTS = [('RQ2', f'C - PFR | {o}', C[o], PFR[o]) for o in ('CE', 'Proper', 'PG', 'LOO', 'Reparam')]
for head, d in (('PFR', PFR), ('C', C)):
    CONTRASTS += [('RQ3', f'{head}: LOO - PG', d['LOO'], d['PG']),
                  ('RQ3', f'{head}: LOO - Reparam', d['LOO'], d['Reparam']),
                  ('RQ3', f'{head}: LOO - CE', d['LOO'], d['CE']),
                  ('RQ3', f'{head}: CE - PG', d['CE'], d['PG']),
                  ('RQ3', f'{head}: Reparam - PG', d['Reparam'], d['PG']),
                  ('RQ3', f'{head}: Proper - CE', d['Proper'], d['CE'])]


def main(budget_s=float(sys.argv[1]) if len(sys.argv) > 1 else 1e9):
    """Resumable: each finished contrast is cached in partial.jsonl; pass a time budget in seconds."""
    import time
    t0 = time.time()
    cache = HERE / 'partial.jsonl'
    done = {}
    if cache.exists():
        for l in cache.read_text().splitlines():
            r = json.loads(l)
            done[(r['track'], r['label'])] = r
    for i, (track, tasks) in enumerate((('seen', BS.SEEN), ('heldout', BS.HELD))):
        for j, (fam, label, a, b) in enumerate(CONTRASTS):
            if (track, label) in done:
                continue
            if time.time() - t0 > budget_s:
                print(f'time budget reached; {len(done)} of {2 * len(CONTRASTS)} contrasts done')
                return
            BS.RNG = np.random.default_rng([20261001, i, j])      # per-contrast stream: resumable, reproducible
            n0 = len(BS.RESULTS)
            BS.contrast(Path('/'), a, b, tasks, f'{fam} {label}')
            if len(BS.RESULTS) > n0:
                r = dict(BS.RESULTS[-1], family=fam, track=track, label=label)
                done[(track, label)] = r
                with cache.open('a', encoding='utf-8') as f:
                    f.write(json.dumps(r) + '\n')
    out = [done[(t, lab)] for t in ('seen', 'heldout') for _, lab, _, _ in CONTRASTS if (t, lab) in done]
    for fam in ('RQ2', 'RQ3'):
        rows = [r for r in out if r['family'] == fam]
        for r, p in zip(rows, BS.holm([r['p'] for r in rows])):
            r['p_holm'] = p
    (HERE / 'contrasts_patch8.json').write_text(json.dumps(dict(draws=BS.B, rng_seed=20261001,
        clusters='normalised document text', conditional_on='observed tasks and seed-averaged correctness',
        min_attainable_p=2 / (BS.B + 1), results=out), indent=2), encoding='utf-8')
    lines = [f"{r['family']} {r['label']:24s} [{r['track']:7s}] {r['delta']:+6.2f} [{r['lo']:+6.2f}, {r['hi']:+6.2f}]  p={r['p']:.4f}  Holm={r['p_holm']:.4f}" for r in out]
    (HERE / 'contrasts_patch8.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
