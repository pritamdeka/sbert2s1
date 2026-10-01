"""Latency / throughput benchmark (run it with nothing else on the GPU).

For each architecture, batch-1 requests with N in {1, 5, 10} questions over one state from the
test sets: p50/p95 latency and questions/s, on GPU (and CPU with --cpu). Untrained weights have the
same cost as trained ones, so no checkpoint is needed. The LLM row uses option-letter scoring.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from . import data as D
from .common import atomic_json
from .model import S1Model, load_encoder
from .models_io import spec_with_path

CFG = dict(max_len=512, head_max_len=192, bi_state_len=256, amp='bf16')


def bench_encoder(model, split, device, n_questions, reps=60):
    # States with >= n questions (HoC abstracts carry 10 questions each).
    by_rec = {}
    for i, it in enumerate(split['items']):
        by_rec.setdefault(it['rec'], []).append(i)
    groups = [v[:n_questions] for v in by_rec.values() if len(v) >= n_questions][:reps + 10]
    times = []
    model.eval()
    with torch.no_grad():
        for g in groups:
            b = D.to_device(D.collate(split, g, cross=model.uses_cross(), bi=model.uses_bi(), **{k: CFG[k] for k in
                            ('max_len', 'head_max_len', 'bi_state_len')}), device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                model(b)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times[10:])            # drop warm-up
    return dict(p50_ms=float(np.percentile(t, 50)), p95_ms=float(np.percentile(t, 95)),
                questions_per_s=float(n_questions * 1000 / t.mean()), n=len(t))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--cache', default='cache/pretok')
    a = ap.parse_args()
    device = torch.device('cpu' if a.cpu or not torch.cuda.is_available() else 'cuda')
    res = {}
    sp = D.load_split(a.cache, 'pubmedbert', 'hoc', 'test')
    spec = spec_with_path('pubmedbert')
    for arch in ('Z', 'B', 'C', 'PFR'):
        m = S1Model(load_encoder(spec['path']), arch).to(device)
        res[f'pubmedbert/{arch}'] = {n: bench_encoder(m, sp, device, n) for n in (1, 5, 10)}
        print(arch, res[f'pubmedbert/{arch}'], flush=True)
        del m
    try:
        from .train import build_model
        m, _ = build_model(dict(arch='C', prior='shared', init_from=None), spec_with_path('laya'), device)
        spl = D.load_split(a.cache, 'laya_en', 'hoc', 'test')
        res['laya/C'] = {n: bench_encoder(m, spl, device, n) for n in (1, 5, 10)}
        print('laya', res['laya/C'], flush=True)
    except Exception as e:                       # ModernBERT on this stack is itself a smoke-test question
        res['laya/C'] = {'error': str(e)[:500]}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    atomic_json(Path(a.out) / f'latency_{device.type}.json', dict(device=torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu', results=res))


if __name__ == '__main__':
    main()
