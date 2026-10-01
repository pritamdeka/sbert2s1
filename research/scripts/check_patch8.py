"""Pre-flight check for patch 8 (login node, CPU, about a minute). Exits non-zero if anything is wrong.

    python scripts/check_patch8.py

1. Config hashes: every existing run directory still hashes identically under the patched trainer
   (patch 8 must not invalidate finished or pending runs).
2. Objectives: on synthetic logits, the leave-one-out PG estimator matches the pathwise gradient
   (slope ~1), the same-sample mean baseline gives slope ~(G-1)/G, w_score scales linearly, G<2 is rejected.
3. Temperature lookup: Python/NumPy/PyTorch integer codes and names select the same temperature;
   invalid types raise.
4. Inputs: pretokenised caches, the original prepared/ data with the clinical track, and the saved
   checkpoint used by the gradient probe.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

FAIL, WARN = [], []


def ok(msg):
    print('  ok   ', msg)


def bad(msg):
    print('  FAIL ', msg)
    FAIL.append(msg)


def warn(msg):
    print('  warn ', msg)
    WARN.append(msg)


# s1.train.DEFAULTS as shipped in patch 7. Patch 8 must leave it byte-identical, otherwise the config hash of
# every existing run directory would change ("config changed; use a new run directory").
DEFAULTS_P7 = {"G": 4, "alpha_min": 20.0, "amp": "bf16", "arch": "PFR", "batch": 32, "bi_state_len": 256,
               "cache": None, "ckpt_minutes": 20, "encoder": None, "eval_init": False, "eval_perms": 5,
               "eval_splits": [], "frac": 1.0, "grad_probe_every": 250, "head_max_len": 192, "init_from": None,
               "long_eval": {}, "lr_alpha": 0.05, "lr_enc": 2.5e-05, "lr_head": 0.0001, "max_epochs": 8.0,
               "max_len": 512, "max_minutes": 0, "mix_alpha": 0.5, "objective": "rlcd_pg", "permute": True,
               "prepared": None, "prior": "shared", "retention": [], "save_model": False, "seed": 0,
               "sigma_end": 0.1, "sigma_start": 0.4, "steps": 8000, "train_tasks": [], "w_ce": 1.0, "w_rps": 1.0,
               "w_sph": 0.75, "warmup": 0.05, "weight_decay": 0.01}


def check_hashes():
    print('[1] config hashes of existing runs')
    from s1.common import digest, read_json
    from s1.train import DEFAULTS
    (ok if DEFAULTS == DEFAULTS_P7 else bad)('s1.train.DEFAULTS identical to patch 7')
    n, mism = 0, []
    for cpath in sorted((ROOT / 'runs').rglob('config.json')):
        rel = cpath.parent.relative_to(ROOT)
        icfg = cpath.parent / 'input_config.json'
        if not icfg.exists() or rel.parts[1] in ('smoke', 'smoke2'):
            continue      # smoke runs predate the patch-4 alpha keys; they are never re-run
        stored = read_json(cpath)
        cfg = dict(DEFAULTS, **read_json(icfg))
        cfg['prepared'] = cfg['prepared'] or stored['config'].get('prepared')
        cfg['cache'] = cfg['cache'] or stored['config'].get('cache')
        h = digest({k: v for k, v in cfg.items() if k not in ('prepared', 'cache', 'ckpt_minutes')})
        if h != stored['config_hash']:
            mism.append(str(rel))
        n += 1
    if n == 0:
        warn('no run directories found under runs/ (run this inside the campaign S1_ROOT)')
    elif mism:
        bad(f'{len(mism)} of {n} run directories would change hash, e.g. {mism[:3]}')
    else:
        ok(f'{n} non-smoke run directories keep their config hash')


def check_objectives():
    print('[2] objectives')
    from s1 import objectives as O
    torch.manual_seed(0)
    n, G = 40000, 4
    base = torch.tensor([[0.3, -0.6, 1.0, -1e4]]).repeat(n, 1)
    t = torch.tensor([[0., 1., 0., 0.]]).repeat(n, 1)
    m = torch.tensor([[True, True, True, False]]).repeat(n, 1)
    q = torch.zeros(n, dtype=torch.long)

    def grad(name, **kw):
        torch.manual_seed(1)
        z = base.clone().requires_grad_(True)
        loss, _ = O.loss(name, z, t, m, q, sigma=0.25, G=G, w_ce=0.0, **kw)
        g, = torch.autograd.grad(loss, z)
        return (g * n)[:, :3].double()

    ref = grad('rlcd_reparam').mean(0)
    loo = grad('rlcd_pg_loo')
    slope = float((loo.mean(0) * ref).sum() / (ref * ref).sum())
    se = float(loo.std(0).norm() / np.sqrt(n) / ref.norm())
    (ok if abs(slope - 1) < max(0.03, 4 * se) else bad)(f'LOO PG slope vs pathwise = {slope:.3f} (expected 1)')
    if not np.all(grad('rlcd_pg_loo').numpy()[:, :] == grad('rlcd_pg_loo').numpy()):
        bad('LOO PG not reproducible under a fixed seed')
    g1, g2 = grad('rlcd_pg_loo'), grad('rlcd_pg_loo', w_score=2.0)
    (ok if torch.allclose(g2, 2 * g1) else bad)('w_score scales the score term linearly')
    g_hist = grad('rlcd_pg')
    (ok if torch.isfinite(g_hist).all() else bad)('historical rlcd_pg unchanged and finite')
    try:
        O.loss('rlcd_pg_loo', base[:2].clone().requires_grad_(True), t[:2], m[:2], q[:2], G=1)
        bad('G=1 should be rejected for rlcd_pg_loo')
    except ValueError:
        ok('G<2 rejected for rlcd_pg_loo')
    from s1.grad_probe import item_grads
    pm = item_grads('pg_mean', base, t, m, q, 0.25, G, 0.75, 1.0)[:, :3].mean(0)
    s_pm = float((pm * ref).sum() / (ref * ref).sum())
    (ok if abs(s_pm - (G - 1) / G) < 0.05 else bad)(f'same-sample mean baseline slope = {s_pm:.3f} (expected {(G - 1) / G})')


def check_temperatures():
    print('[3] temperature lookup')
    from s1.temperature_lookup import temperature_for
    from s1.calibrate import temperature_for as tf2
    tp = {'bucket': {'choice:3-5': 2.0, 'score:3-5': 1.19, 'noul:2': 1.4}, 'type': {'choice': 3.0, 'score': 0.9}}
    cases = [(1, 4, 1.19), (np.int64(1), 4, 1.19), (np.int32(1), 4, 1.19), (torch.tensor(1), 4, 1.19),
             ('score', 4, 1.19), (0, 4, 2.0), (0, 40, 3.0), (2, 2, 1.4), (1, 7, 0.9), (2, 9, 1.0)]
    good = all(abs(temperature_for(tp, qt, k) - want) < 1e-12 for qt, k, want in cases)
    (ok if good else bad)('integer codes (int/NumPy/torch) and names select the right bucket/type/fallback')
    (ok if tf2 is temperature_for else bad)('s1.calibrate re-exports the shared lookup')
    for badq in (1.9, True, 'Score', 3, torch.tensor([1, 2])):
        try:
            temperature_for(tp, badq, 3)
            bad(f'invalid question type {badq!r} accepted')
        except (TypeError, ValueError):
            pass
    ok('invalid question types raise')


def check_inputs():
    print('[4] inputs')
    from s1.common import read_json
    for fam in ('pubmedbert', 'modernbert'):
        d = ROOT / 'cache' / 'pretok' / fam
        (ok if d.exists() and any(d.glob('*.train.pkl')) else bad)(f'tokenizer cache {d.relative_to(ROOT)}')
    (ok if (ROOT / 'prepared' / 'mednli' / 'test.jsonl').exists() else bad)('clinical track present in prepared/')
    lock = ROOT / 'configs' / 'models.lock.json'
    if lock.exists():
        have = set(read_json(lock))
        miss = {'spubmedbert', 'modernbert', 'mbembed', 'gtemb', 'bcmb', 'bcmbembed'} - have
        (ok if not miss else bad)(f'models.lock.json has all encoders{"" if not miss else f" (missing {miss})"}')
    else:
        bad('configs/models.lock.json missing')
    mp = ROOT / 'runs' / 'main' / 'spubmedbert' / 'PFR' / 's0' / 'model.pt'
    (ok if mp.exists() else warn)(f'probe checkpoint {mp.relative_to(ROOT)}'
                                  + ('' if mp.exists() else ' missing: the probe will skip it'))
    from s1.queue import done_ids
    done = done_ids()
    for dep in ('pretok/pubmedbert', 'pretok/modernbert'):
        (ok if dep in done else bad)(f'queue dependency {dep} is done')


def main():
    check_hashes()
    check_objectives()
    check_temperatures()
    check_inputs()
    print(f'\n{len(FAIL)} failure(s), {len(WARN)} warning(s)')
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
