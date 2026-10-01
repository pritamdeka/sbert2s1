"""Post-hoc temperature scaling per (question type, option-count bucket), Laya's bucket scheme.

Fitted on a calibration slice carved from train *before* training (never on dev/test). Buckets with
fewer than MIN_ITEMS items fall back to the per-type temperature, then to 1.0.
"""
import numpy as np
import torch

from .common import QTYPE_NAMES, temp_bucket
from .temperature_lookup import temperature_for  # noqa: F401  (re-exported; patch 8)

MIN_ITEMS = 10
T_MIN, T_MAX = 0.05, 20.0


def _fit(rows):
    """rows: list of (logits[K], target[K]). Minimise soft NLL over log T with LBFGS."""
    kmax = max(len(z) for z, _ in rows)
    Z = torch.full((len(rows), kmax), -1e4, dtype=torch.float64)
    Tg = torch.zeros((len(rows), kmax), dtype=torch.float64)
    for i, (z, t) in enumerate(rows):
        Z[i, :len(z)] = torch.tensor(z, dtype=torch.float64)
        Tg[i, :len(t)] = torch.tensor(t, dtype=torch.float64)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        l = -(Tg * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        l.backward()
        return l
    try:
        opt.step(closure)
        t = float(log_t.exp())
    except RuntimeError:
        return 1.0
    return float(min(T_MAX, max(T_MIN, t))) if np.isfinite(t) else 1.0


def fit_temperatures(entries):
    """entries: iterable of dict(qtype, logits, target). Returns {'type': {name: T}, 'bucket': {b: T}}."""
    by_type, by_bucket = {}, {}
    for e in entries:
        k = len(e['logits'])
        by_type.setdefault(QTYPE_NAMES[e['qtype']], []).append((e['logits'], e['target']))
        by_bucket.setdefault(temp_bucket(e['qtype'], k), []).append((e['logits'], e['target']))
    types = {name: (_fit(r) if len(r) >= MIN_ITEMS else 1.0) for name, r in by_type.items()}
    buckets = {b: _fit(r) for b, r in by_bucket.items() if len(r) >= MIN_ITEMS}
    return {'type': types, 'bucket': buckets,
            'counts': {b: len(r) for b, r in by_bucket.items()}}
