"""Patch 8: revision controls for RQ2/RQ3 (+ optional RQ1 learning-curve seeds). Idempotent.

All runs use the ORIGINAL prepared/ data and the existing tokenizer cache, and copy the exact base
configuration of the historical grid (scripts/enqueue_v1.py), so every new number slots into the existing
tables. Run IDs are new; nothing existing is touched or re-run.

    python scripts/enqueue_patch8.py --dry-run          # list what would be enqueued
    python scripts/enqueue_patch8.py                    # P0 + P1 (default, 15 trainings + 1 probe)
    python scripts/enqueue_patch8.py --max-priority 0   # only the must-have P0 block (9 trainings + probe)
    python scripts/enqueue_patch8.py --max-priority 2   # also the optional learning-curve seeds (+50 learning-curve runs)

P0  must-have (9 trainings, S-PubMedBERT, seeds 0-2)
    rq3/pg_loo/s*      PFR + unbiased leave-one-out PG (+CE)      -> matched estimator control vs rq3/reparam
    rq3c/ce/s*         C   + CE           (save_model: release candidates; s0 also retrieval retention)
    rq3c/reparam/s*    C   + Reparam (+CE)                         -> head x objective (vs main/spubmedbert/C = C+PG)
    probe/gradvar      matched gradient-estimator probe on saved checkpoints (no training, ~15 min)
P1  completes the 2-head x 5-objective grid (6 trainings)
    rq3c/proper/s*, rq3c/pg_loo/s*
P2  optional RQ1 data-efficiency seeds for the extension pairs (50 runs, frac 0.1 / 0.02;
    evaluation dominates, so ~1.5-2.5 h each under packing, as measured for the existing lc runs)
    lc/{modernbert,mbembed,gtemb,bcmb,bcmbembed}/{C,PFR}/f0.1/s{1,2} and f0.02/s{0,1,2}

Existing cells reused as-is: rq3/{ce,proper,reparam,pg_noce} (PFR), main/spubmedbert/PFR (PFR+PG),
main/spubmedbert/C (C+PG), lc/<ext>/<arch>/f0.1/s0.
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from enqueue_v1 import FAMILY, LONG, TRAIN, VRAM, eval_splits  # noqa: E402  (identical base config)
from s1.common import atomic_json  # noqa: E402

EXT_LC = ['modernbert', 'mbembed', 'gtemb', 'bcmb', 'bcmbembed']
PROBE_RUNS = ['runs/main/spubmedbert/PFR/s0',      # historical checkpoint, trained with released PG
              'runs/rq3/pg_loo/s0',                # trained with the unbiased LOO estimator
              'runs/rq3c/ce/s0']                   # C head trained with CE


def grid(clinical):
    ev = eval_splits(clinical)
    base = dict(train_tasks=TRAIN, eval_splits=ev, long_eval={k: v for k, v in LONG.items() if k in ev},
                objective='rlcd_pg', steps=8000, batch=32)
    tasks = []

    def add(run_id, prio, enc, est=2.0, vram=VRAM['base'], **kw):
        # Same task layout as enqueue_v1.train_task, but input_config.json is written only for tasks that are
        # actually enqueued (not on --dry-run, not for priorities that are filtered out).
        cfg = dict(base, encoder=enc, **kw)
        run_dir = f'runs/{run_id}'
        tasks.append(dict(run_id=run_id, kind='train', priority=prio, run_dir=run_dir,
                          argv=['-m', 's1.train', '--run-dir', run_dir, '--config', f'{run_dir}/input_config.json'],
                          vram_gb=vram, slots=1, est_hours=est, after=[f'pretok/{FAMILY[enc]}'],
                          vram_key=f"{cfg['encoder']}:{cfg['arch']}", _cfg=cfg))

    for s in (0, 1, 2):
        # P0: matched estimator control (PFR head; compare with rq3/reparam and main/spubmedbert/PFR)
        add(f'rq3/pg_loo/s{s}', 0, 'spubmedbert', arch='PFR', seed=s, objective='rlcd_pg_loo',
            save_model=(s == 0))
        # P0: head x objective (C head; C+PG already exists as main/spubmedbert/C)
        add(f'rq3c/ce/s{s}', 0, 'spubmedbert', arch='C', seed=s, objective='ce', save_model=True,
            retention=['scifact', 'nfcorpus'] if s == 0 else [])
        add(f'rq3c/reparam/s{s}', 0, 'spubmedbert', arch='C', seed=s, objective='rlcd_reparam')
        # P1: complete the grid
        add(f'rq3c/proper/s{s}', 1, 'spubmedbert', arch='C', seed=s, objective='proper')
        add(f'rq3c/pg_loo/s{s}', 1, 'spubmedbert', arch='C', seed=s, objective='rlcd_pg_loo')

    tasks.append(dict(run_id='probe/gradvar', kind='cmd', priority=0, run_dir='runs/probe/gradvar',
                      argv=['-m', 's1.grad_probe', '--out', 'runs/probe/gradvar', '--runs', *PROBE_RUNS],
                      vram_gb=VRAM['base'], slots=1, est_hours=0.5,
                      after=['rq3/pg_loo/s0', 'rq3c/ce/s0'], vram_key='probe'))

    # P2: optional learning-curve seeds for the extension pairs (same settings as grid_ext's lc runs)
    for enc in EXT_LC:
        for arch in ('C', 'PFR'):
            for frac, seeds in ((0.1, (1, 2)), (0.02, (0, 1, 2))):
                for s in seeds:
                    add(f'lc/{enc}/{arch}/f{frac}/s{s}', 2, enc, est=2.5 if frac == 0.1 else 1.5, vram=24, arch=arch, seed=s, frac=frac)
    return tasks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--max-priority', type=int, default=1)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    clinical = (ROOT / 'prepared' / 'mednli' / 'test.jsonl').exists()
    if not clinical:
        sys.exit('prepared/mednli/test.jsonl missing: the historical runs were evaluated WITH the clinical '
                 'track, so these runs must be too (their configs would otherwise not match). Aborting.')
    tasks = [t for t in grid(clinical) if t['priority'] <= a.max_priority]
    by_p = {}
    for t in tasks:
        by_p[t['priority']] = by_p.get(t['priority'], 0) + 1
    print(f'tasks per priority: {by_p}  (queue: {os.environ.get("S1_QUEUE", ROOT / "queue")})')
    if a.dry_run:
        for t in tasks:
            c = t.get('_cfg', {})
            print(f"  P{t['priority']}  {t['run_id']:32s} {c.get('arch', ''):4s} {c.get('objective', ''):13s} "
                  f"frac={c.get('frac', 1.0)} save_model={c.get('save_model', False)}")
        return
    from s1.queue import enqueue
    for t in tasks:
        cfg = t.pop('_cfg', None)
        if cfg is not None and not (ROOT / t['run_dir'] / 'config.json').exists():
            atomic_json(ROOT / t['run_dir'] / 'input_config.json', cfg)
    print('enqueued', enqueue(tasks), 'new tasks')


if __name__ == '__main__':
    main()
