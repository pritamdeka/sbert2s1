"""Measure how many concurrent trainings one MI300X should run (sets S1_MAX_SLOTS).

Inside ONE GPU allocation, for N in 1,2,4,6,8: start N identical PFR/S-PubMedBert trainings for
--minutes each (a private queue per N, packer with max-slots N), then read each run's throughput.
Writes PACKING_CALIBRATION.md. Requires the pubmedbert pre-tokenised cache (the smoke stage builds it).

    python scripts/packing_calibration.py --minutes 8
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s1.common import atomic_json, read_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--minutes', type=float, default=8)
    ap.add_argument('--levels', type=int, nargs='+', default=[1, 2, 4, 6, 8])
    a = ap.parse_args()
    rows = []
    for n in a.levels:
        qdir = ROOT / 'queue_pack' / f'n{n}'
        tasks = []
        for i in range(n):
            run_dir = f'runs_pack/n{n}/r{i}'
            cfg = dict(encoder='spubmedbert', arch='PFR', objective='rlcd_pg', seed=i, steps=100000,
                       train_tasks=['pubmedqa_art', 'ade', 'hoc', 'medline_s1'], eval_splits=[],
                       max_minutes=a.minutes, eval_perms=0)
            atomic_json(ROOT / run_dir / 'input_config.json', cfg)
            tasks.append(dict(run_id=f'pack/n{n}/r{i}', priority=0, run_dir=run_dir, vram_gb=1, slots=1,
                              est_hours=0.3, argv=['-m', 's1.train', '--run-dir', run_dir, '--config', f'{run_dir}/input_config.json']))
        env = dict(os.environ, S1_QUEUE=str(qdir))
        subprocess.run([sys.executable, '-c', f'from s1.queue import enqueue; import json; enqueue(json.loads({json.dumps(json.dumps(tasks))}))'],
                       cwd=ROOT, env=env, check=True)
        subprocess.run([sys.executable, '-m', 's1.queue', 'pack', '--vram-gb', '180', '--max-slots', str(n)],
                       cwd=ROOT, env=env, check=True)
        per = []
        peak = []
        for i in range(n):
            tp = ROOT / f'runs_pack/n{n}/r{i}/throughput.json'
            pm = ROOT / f'runs_pack/n{n}/r{i}/peak_mem.json'
            if tp.exists():
                per.append(read_json(tp)['items_per_s'])
            if pm.exists():
                peak.append(read_json(pm)['peak_gb'])
        rows.append(dict(n=n, aggregate=sum(per), per_run=(sum(per) / len(per)) if per else 0, peak_gb=max(peak or [0]),
                         completed=len(per)))
        print(rows[-1], flush=True)
    base = rows[0]['aggregate'] or 1
    lines = ['# Packing calibration (MI300X, PFR on S-PubMedBert, batch 32, bf16)', '',
             '| concurrent runs | aggregate items/s | per-run items/s | speed-up vs 1 | peak VRAM per run (GB) |',
             '|---|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['n']} | {r['aggregate']:.0f} | {r['per_run']:.0f} | {r['aggregate'] / base:.2f}x | {r['peak_gb']:.1f} |")
    best = max(rows, key=lambda r: r['aggregate'])
    knee = next((r for r in rows if r['aggregate'] >= 0.9 * best['aggregate']), best)
    lines += ['', f"Recommended S1_MAX_SLOTS = {knee['n']} (smallest level within 10% of the best aggregate throughput)."]
    (ROOT / 'PACKING_CALIBRATION.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
