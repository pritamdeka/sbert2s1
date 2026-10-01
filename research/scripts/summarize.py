"""Compact metrics table for finished runs (numbers only, no text; safe to paste anywhere).

    python scripts/summarize.py                 # every run under runs/
    python scripts/summarize.py runs/smoke      # a subtree
Columns per (run, eval split, question group): n, K, accuracy, ECE before / after temperature
scaling, AUROC (binary) or MAE (ordinal). Also prints permutation flip rate, retention nDCG@10,
step-0 (init) accuracy when present, and training throughput / peak memory.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fmt(v, nd=3):
    return '-' if v is None or v != v else f'{v:.{nd}f}'


def rows(metrics):
    for split, m in sorted(metrics.items()):
        if split == 'retention':
            for r, v in m.items():
                yield f'  retention {r}: nDCG@10 {fmt(v["ndcg10"])} ({v["n_queries"]} q)'
            continue
        if m.get('missing'):
            yield f'  {split}: MISSING (not prepared)'
            continue
        extra = f' ({m["items_per_s"]:.1f} items/s)' if 'items_per_s' in m else ''
        for g, v in sorted(m['groups'].items()):
            raw, ts = v['raw'], v.get('ts') or {}
            tail = f'auroc {fmt(raw.get("auroc"))}' if 'auroc' in raw else f'mae {fmt(raw.get("mae"))}' if 'mae' in raw else ''
            yield (f'  {split:24s} {g[:28]:28s} n={raw["n"]:5d} K={raw["K"]:2d} acc {fmt(raw["acc"])} '
                   f'ece {fmt(raw["ece"])}->{fmt(ts.get("ece"))} (T={fmt(v.get("T"), 2)}) {tail}{extra}')
        if m.get('permutation'):
            p = m['permutation']
            yield f'  {split:24s} permutation: flip {fmt(p["flip_rate"])} JS {fmt(p["mean_js"], 4)} over {p["n"]} items'
        for strat, g in (m.get('long') or {}).items():
            accs = [v['raw']['acc'] for v in g.values()]
            yield f'  {split:24s} long[{strat}] mean acc {fmt(sum(accs) / len(accs))}'


def main():
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'runs'
    for mf in sorted(base.rglob('metrics_final.json')):
        run = mf.parent
        head = str(run.relative_to(ROOT)) if run.is_relative_to(ROOT) else str(run)
        info = []
        if (run / 'throughput.json').exists():
            info.append(f'{json.loads((run / "throughput.json").read_text())["items_per_s"]:.0f} items/s')
        if (run / 'peak_mem.json').exists():
            info.append(f'peak {json.loads((run / "peak_mem.json").read_text())["peak_gb"]:.1f} GB')
        if (run / 'state.json').exists():
            st = json.loads((run / 'state.json').read_text())
            if 'alpha0' in st:
                info.append('alpha0 ' + ' '.join(f'{k[0]}={v:.2f}' for k, v in st['alpha0'].items()))
        print(f'\n== {head} {"(DONE)" if (run / "DONE").exists() else "(incomplete)"} {" | ".join(info)}')
        for line in rows(json.loads(mf.read_text())):
            print(line)
        init = run / 'metrics_init.json'
        if init.exists():
            for split, m in sorted(json.loads(init.read_text()).items()):
                if 'groups' in m:
                    accs = [v['raw']['acc'] for v in m['groups'].values()]
                    print(f'  [step 0] {split:24s} mean acc {fmt(sum(accs) / len(accs))}')
        log = run / 'train_log.jsonl'
        if log.exists():
            lines = log.read_text().splitlines()
            if lines:
                last = json.loads(lines[-1])
                print(f'  train: step {last["step"]} loss {fmt(last["loss"])} ce {fmt(last.get("ce"))} '
                      f'items/s {fmt(last.get("items_per_s"), 0)} alpha {[round(a, 2) for a in last.get("alpha", [])]}')


if __name__ == '__main__':
    main()
