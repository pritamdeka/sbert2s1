"""Matched gradient-estimator probe for RQ3 (patch 8). No training; one GPU for a few minutes.

The training-log `grad_var` values compare estimators measured on DIFFERENT models at different steps and
on different scales, so they cannot isolate estimator variance. This probe fixes all of that: for each
saved checkpoint it computes the logits of the SAME training batches once, then, at fixed noise levels,
draws many independent gradient estimates (w.r.t. the logits, smoothed-score term only, w_ce = 0) from

    rlcd_pg        released recipe: same-sample mean baseline + batch-wide std normalisation
    pg_mean        same-sample mean baseline, no normalisation  (expected slope (G-1)/G)
    rlcd_pg_loo    leave-one-out baseline, no normalisation      (unbiased, expected slope 1)
    rlcd_reparam   pathwise / reparameterised                    (unbiased, expected slope 1)

and compares each with a high-sample pathwise reference g* of the true smoothed-score gradient:

    slope        <E[g], g*> / <g*, g*>      effective scale of the estimator (1 = unbiased scale)
    cosine       cos(E[g], g*)              direction agreement
    var          per-item gradient variance (sum over options, mean over items) for ONE estimate
    var_matched  var / slope^2              variance after removing the scale difference (fair comparison)
    snr          |g*|^2 / var_matched       signal-to-noise per item
    score_vs_ce  slope * |g*| / |grad CE|   relative weight of the score term against the CE term

    python -m s1.grad_probe --runs runs/main/spubmedbert/PFR/s0 runs/rq3/pg_loo/s0 --out runs/probe/gradvar
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

from . import data as D
from . import evaluate as E
from . import objectives as O
from .common import ROOT, atomic_json, read_json

ESTIMATORS = ('rlcd_pg', 'pg_mean', 'rlcd_pg_loo', 'rlcd_reparam')


def _pg_mean_unnormalised(logits, target, mask, qtype, sigma, G, w_sph, w_rps):
    """Released PG with the std normalisation removed (isolates the (G-1)/G baseline effect)."""
    maskf = mask.float()
    eps = O._noise((G,) + tuple(logits.shape), maskf, sigma, logits.device)
    z = logits.detach().unsqueeze(0) + eps
    with torch.no_grad():
        q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
        r = O.proper_reward(q, target.unsqueeze(0), qtype, mask, w_sph, w_rps)
        adv = r - r.mean(0, keepdim=True)
    logp = -(((z - logits.unsqueeze(0)) ** 2) * maskf).sum(-1) / (2 * sigma ** 2)
    return -(adv * logp).mean()


def item_grads(name, z0, target, mask, qtype, sigma, G, w_sph, w_rps):
    """One gradient estimate per item, [N, K] (the loss averages over items, so multiply by N)."""
    z = z0.detach().clone().requires_grad_(True)
    if name == 'pg_mean':
        loss = _pg_mean_unnormalised(z, target, mask, qtype, sigma, G, w_sph, w_rps)
    else:
        loss, _ = O.loss(name, z, target, mask, qtype, sigma=sigma, G=G, w_ce=0.0, w_sph=w_sph, w_rps=w_rps)
    g, = torch.autograd.grad(loss, z)
    return (g * mask).double() * z.shape[0]


def probe_batch(z, target, mask, qtype, sigma, G, draws, ref_draws, w_sph, w_rps, seed):
    torch.manual_seed(seed)
    ref = torch.stack([item_grads('rlcd_reparam', z, target, mask, qtype, sigma, 256, w_sph, w_rps)
                       for _ in range(ref_draws)]).mean(0)
    zc = z.detach().clone().requires_grad_(True)
    ce = O.soft_ce(zc, target, mask)
    gce, = torch.autograd.grad(ce, zc)
    gce = (gce * mask).double() * z.shape[0]
    ref_sq = float((ref ** 2).sum(-1).mean())
    ce_norm = float(gce.norm(dim=-1).mean())
    ref_norm = float(ref.norm(dim=-1).mean())
    out = {}
    for name in ESTIMATORS:
        torch.manual_seed(seed + 1 + ESTIMATORS.index(name))
        g = torch.stack([item_grads(name, z, target, mask, qtype, sigma, G, w_sph, w_rps) for _ in range(draws)])
        m = g.mean(0)
        var = float(g.var(0, unbiased=True).sum(-1).mean())
        slope = float((m * ref).sum() / (ref * ref).sum().clamp_min(1e-30))
        cos = float((m * ref).sum() / (m.norm() * ref.norm()).clamp_min(1e-30))
        vm = var / max(slope ** 2, 1e-30)
        out[name] = dict(slope=slope, cosine=cos, var=var, var_matched=vm, snr=ref_sq / max(vm, 1e-30),
                         score_vs_ce=slope * ref_norm / max(ce_norm, 1e-30))
    out['_ref'] = dict(ref_norm=ref_norm, ce_norm=ce_norm)
    return out


def load_run(run_dir, device):
    from .models_io import spec_with_path
    from .train import build_model, load_splits
    cfg = read_json(run_dir / 'config.json')['config']
    if not (run_dir / 'model.pt').exists():
        return None, cfg, None, None
    spec = spec_with_path(cfg['encoder'])
    model, _ = build_model(dict(cfg, init_from=None), spec, device)
    model.load_state_dict(torch.load(run_dir / 'model.pt', map_location='cpu'))
    model.eval()
    tok = None
    if spec.get('query_prefix') or spec.get('doc_prefix'):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(os.path.join(spec['path'], spec['tokenizer_subfolder'])
                                            if spec.get('tokenizer_subfolder') else spec['path'])
    for key, field in (('_q_prefix', 'query_prefix'), ('_d_prefix', 'doc_prefix')):
        cfg[key] = tok(spec[field], add_special_tokens=False)['input_ids'] if spec.get(field) else []
    family = spec['tok_family']
    train = D.merge(load_splits(cfg, family, [t if '/' in t else t + '/train' for t in cfg['train_tasks']]))
    return model, cfg, train, family


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', nargs='+', required=True, help='run directories with model.pt (relative to S1_ROOT)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--batches', type=int, default=40)
    ap.add_argument('--draws', type=int, default=32, help='independent estimates per estimator and batch')
    ap.add_argument('--ref-draws', type=int, default=8, help='x256-sample pathwise draws for the reference')
    ap.add_argument('--sigmas', type=float, nargs='+', default=[0.4, 0.25, 0.1])
    ap.add_argument('--seed', type=int, default=20261001, help='batch schedule seed, shared by all runs')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    device = torch.device(a.device)
    if device.type == 'cuda':
        from .common import apply_vram_cap
        apply_vram_cap()
    out_dir = ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    result = dict(settings=dict(vars(a), estimators=ESTIMATORS), runs={})
    for rd in a.runs:
        run_dir = ROOT / rd
        t0 = time.time()
        model, cfg, train, family = load_run(run_dir, device)
        if model is None:
            print(f'{rd}: no model.pt (run without save_model?) -- skipped', flush=True)
            result['runs'][rd] = dict(missing='model.pt')
            continue
        # Same batch schedule for every run of the same tokenizer family (fixed probe seed, frac=1).
        mix = D.Mixture(train, frac=1.0, alpha=cfg['mix_alpha'], seed=a.seed)
        per = {s: {n: [] for n in ESTIMATORS + ('_ref',)} for s in a.sigmas}
        for bi in range(a.batches):
            idxs, rng = mix.batch(bi, cfg['batch'])
            orders = D.random_orders(train, idxs, rng, cfg['permute'])
            b = D.to_device(D.collate(train, idxs, orders, cross=model.uses_cross(), bi=model.uses_bi(),
                                      max_len=cfg['max_len'], head_max_len=cfg['head_max_len'],
                                      bi_state_len=cfg['bi_state_len'], q_prefix=cfg['_q_prefix'],
                                      d_prefix=cfg['_d_prefix']), device)
            with torch.no_grad(), E._autocast(device, cfg['amp']):
                z, _ = model(b)
            z = z.float()
            for s in a.sigmas:
                r = probe_batch(z, b['target'], b['marker_mask'], b['qtype'], s, cfg['G'], a.draws, a.ref_draws,
                                cfg['w_sph'], cfg['w_rps'], seed=a.seed * 1000 + bi)
                for n, v in r.items():
                    per[s][n].append(v)
        summary = {}
        for s in a.sigmas:
            summary[str(s)] = {n: {k: float(np.mean([row[k] for row in rows])) for k in rows[0]}
                               for n, rows in per[s].items()}
        result['runs'][rd] = dict(objective=cfg['objective'], arch=cfg['arch'], seed=cfg['seed'],
                                  family=family, batches=a.batches, minutes=(time.time() - t0) / 60, sigma=summary)
        atomic_json(out_dir / 'grad_probe.json', result)
        print(f'\n{rd}  (trained with {cfg["objective"]}, {cfg["arch"]})', flush=True)
        for s in a.sigmas:
            print(f'  sigma={s}', flush=True)
            for n in ESTIMATORS:
                v = summary[str(s)][n]
                print(f'    {n:13s} slope {v["slope"]:7.3f}  cos {v["cosine"]:6.3f}  var {v["var"]:.3e}  '
                      f'var_matched {v["var_matched"]:.3e}  snr {v["snr"]:.3e}  score/ce {v["score_vs_ce"]:.3f}',
                      flush=True)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    atomic_json(out_dir / 'grad_probe.json', result)
    (out_dir / 'DONE').write_text('ok\n')
    print('wrote', out_dir / 'grad_probe.json', flush=True)


if __name__ == '__main__':
    sys.exit(main())
