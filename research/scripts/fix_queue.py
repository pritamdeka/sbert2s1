"""Patch 4 queue repair. Run AFTER all workers have stopped (squeue shows none):
  * failed tasks go back to pending with retries reset (they failed from GPU OOM caused by a neighbour);
  * pending tasks get the new VRAM budgets (Laya 44 GB, Qwen 100 GB, Gemma 110 GB), which each child now
    enforces as a hard cap.

    python scripts/fix_queue.py            # dry run
    python scripts/fix_queue.py --apply
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s1.common import atomic_json  # noqa: E402
from s1.queue import DIRS  # noqa: E402

BUDGET = {'llm_qwen38_27b': 100, 'llm_gemma4_31b': 110}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--apply', action='store_true')
    a = ap.parse_args()
    running = list(DIRS['running'].glob('*.json'))
    if running:
        sys.exit(f'{len(running)} tasks still in running/: stop the workers first (or run: python -m s1.queue requeue-stale)')
    for p in list(DIRS['failed'].glob('*.json')):
        t = json.loads(p.read_text())
        print('failed -> pending:', p.stem, '|', (t.get('last_error') or '').strip().splitlines()[-1:] or '')
        if a.apply:
            t['retries'] = 0
            t.pop('last_error', None)
            atomic_json(p, t)
            os.replace(p, DIRS['pending'] / p.name)
    for p in DIRS['pending'].glob('*.json'):
        t = json.loads(p.read_text())
        new = BUDGET.get(t.get('vram_key'), 44 if str(t.get('vram_key', '')).startswith('laya:') else None)
        if new and t['vram_gb'] != new:
            print(f'vram {p.stem}: {t["vram_gb"]} -> {new}')
            if a.apply:
                t['vram_gb'] = new
                atomic_json(p, t)
    print('applied' if a.apply else 'dry run: add --apply')


if __name__ == '__main__':
    main()
