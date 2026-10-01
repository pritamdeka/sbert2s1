"""One run = (optional) training -> temperature calibration -> evaluation. Resumable at every stage.

    python -m s1.train --run-dir RUN --config CONFIG.json

* Training checkpoints every `ckpt_minutes` and on SIGTERM (then exits with code 3 = incomplete);
  rerunning the same command continues from the checkpoint.
* Calibration writes temps.json; evaluation writes one prediction file per eval split and skips files
  that already exist, so an interrupted evaluation also resumes.
* The run directory is keyed by the config; a different config in the same directory is refused.
"""
import argparse
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import data as D
from . import evaluate as E
from . import objectives as O
from .calibrate import fit_temperatures
from .common import QTYPE_NAMES, QTYPES, ROOT, apply_vram_cap, atomic_json, digest, read_json, seed_everything
from .model import S1Model, build_laya, load_encoder

INCOMPLETE = 3
DEFAULTS = dict(
    encoder=None, arch='PFR', objective='rlcd_pg', prior='shared', seed=0,
    train_tasks=[], eval_splits=[], frac=1.0, mix_alpha=0.5,
    steps=8000, max_epochs=8.0, batch=32, lr_enc=2.5e-5, lr_head=1e-4, lr_alpha=0.05, alpha_min=20.0,
    warmup=0.05, weight_decay=0.01,
    max_len=512, head_max_len=192, bi_state_len=256,
    G=4, sigma_start=0.4, sigma_end=0.1, w_ce=1.0, w_sph=0.75, w_rps=1.0,
    permute=True, eval_init=False, eval_perms=5, long_eval={}, retention=[], save_model=False,
    init_from=None, amp='bf16', ckpt_minutes=20, grad_probe_every=250, max_minutes=0,
    prepared=None, cache=None)

# Optional keys such as w_score are read with cfg.get(...) and are deliberately NOT added to
# DEFAULTS, so the config hash of every existing run directory is unchanged.

STOP = {'flag': False}


def _on_term(signum, frame):
    STOP['flag'] = True


def lr_at(step, total, base, warmup):
    w = max(1, int(total * warmup))
    if step < w:
        return base * (step + 1) / w
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - w) / max(1, total - w))))


def load_splits(cfg, family, names):
    out = {}
    for name in names:
        task, split = name.split('/')
        out[name] = D.load_split(cfg['cache'], family, task, split)
    return out


def build_model(cfg, spec, device):
    if spec.get('laya'):
        model, laya_cfg = build_laya(spec['path'])
    else:
        enc = load_encoder(spec['path'])
        model = S1Model(enc, cfg['arch'], pooling=spec.get('pooling', 'mean'), prior=cfg['prior'])
        laya_cfg = None
        if cfg.get('init_from'):
            model.load_state_dict(torch.load(Path(cfg['init_from']) / 'model.pt', map_location='cpu'), strict=True)
    return model.to(device), laya_cfg


def zero_shot_alpha(model, calib, cfg, device):
    """a_t = max(1/T_t, alpha_min), with T_t the per-type temperature fitted on raw cosines (alpha=1).
    For PFR the zero-initialised residual is exactly 0 here, so the fit sees the pure bi-encoder prior.

    The floor matters: where the prior is uninformative (options that differ only by a label word,
    e.g. yes/no), the fit returns a huge T, i.e. a near-zero scale, and a near-zero scale can never be
    trained back up. alpha_min=20 is the usual sentence-transformers similarity scale; with cosine gaps
    of ~0.002 it still gives near-uniform step-0 predictions, so the step-0 model stays the calibrated
    zero-shot one to within that gap, but the term remains trainable."""
    with torch.no_grad():
        model.alpha.fill_(1.0)
    temps = fit_temperatures(E.calib_entries(model, calib, cfg, device))
    return {name: max(1.0 / t, cfg['alpha_min']) for name, t in temps['type'].items()}


def set_alpha(model, alpha_by_type):
    with torch.no_grad():
        for name, a in alpha_by_type.items():
            model.alpha[QTYPES[name]] = a


def evaluate_all(model, cfg, family, run, device, temps, tag, tok=None, laya_temps=None):
    out_dir = run / 'preds' / tag
    metrics_path = run / f'metrics_{tag}.json'
    metrics = read_json(metrics_path) if metrics_path.exists() else {}
    for name in cfg['eval_splits']:
        if name in metrics:
            continue
        if STOP['flag']:
            return False
        try:
            sp = D.load_split(cfg['cache'], family, *name.split('/'))
        except FileNotFoundError:
            metrics[name] = {'missing': True}
            atomic_json(metrics_path, metrics)
            continue
        idxs = list(range(len(sp['items'])))
        logits = E.predict(model, sp, idxs, cfg, device)
        E.write_preds(out_dir / f"{name.replace('/', '.')}.jsonl.gz", sp, idxs, logits)
        m = {'groups': E.metrics_for(sp, idxs, logits, temps)}
        if laya_temps:
            m['groups_laya_shipped'] = E.metrics_for(sp, idxs, logits, laya_temps)
        if tag == 'final' and cfg['eval_perms']:
            m['permutation'] = E.permutation_probe(model, sp, cfg, device, cfg['eval_perms'], cfg['seed'])
        if tag == 'final' and name in cfg['long_eval']:
            m['long'] = {}
            for strat in cfg['long_eval'][name]:
                lz = E.long_state(model, sp, cfg, device, strat, sp['special'])
                E.write_preds(out_dir / f"{name.replace('/', '.')}.{strat}.jsonl.gz", sp, idxs, lz)
                m['long'][strat] = E.metrics_for(sp, idxs, lz, temps)
        metrics[name] = m
        atomic_json(metrics_path, metrics)
    if tag == 'final' and cfg['retention'] and tok is not None and 'retention' not in metrics:
        metrics['retention'] = {r: E.retrieval_ndcg(model, tok, device, Path(cfg['prepared']) / '_retrieval' / r,
                                                    q_prefix=cfg.get('_q_prefix_text', ''),
                                                    d_prefix=cfg.get('_d_prefix_text', ''))
                                for r in cfg['retention']}
        atomic_json(metrics_path, metrics)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, _on_term)
    cfg = dict(DEFAULTS, **read_json(a.config))
    cfg['prepared'] = cfg['prepared'] or str(ROOT / 'prepared')
    cfg['cache'] = cfg['cache'] or str(ROOT / 'cache' / 'pretok')
    run = Path(a.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    chash = digest({k: v for k, v in cfg.items() if k not in ('prepared', 'cache', 'ckpt_minutes')})
    cpath = run / 'config.json'
    if cpath.exists() and read_json(cpath)['config_hash'] != chash:
        sys.exit(f'{run}: config changed; use a new run directory')
    atomic_json(cpath, dict(config_hash=chash, config=cfg))
    if (run / 'DONE').exists():
        print('already complete', flush=True)
        return

    from .models_io import spec_with_path
    spec = spec_with_path(cfg['encoder'])
    family = spec['tok_family']
    device = torch.device(a.device)
    if device.type == 'cuda':
        apply_vram_cap()
    seed_everything(cfg['seed'])
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
    model, laya_cfg = build_model(cfg, spec, device)
    laya_temps = None
    if laya_cfg:
        laya_temps = {'type': {QTYPE_NAMES[i]: t for i, t in enumerate(laya_cfg.get('temperature', [1, 1, 1]))},
                      'bucket': laya_cfg.get('temperature_by_options', {})}
    tok = None
    if cfg['retention'] or spec.get('query_prefix') or spec.get('doc_prefix'):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(os.path.join(spec['path'], spec['tokenizer_subfolder'])
                                            if spec.get('tokenizer_subfolder') else spec['path'])
    # Model-specific retrieval prefixes (e.g. Nomic "search_query: "), applied to bi-encoder rows only.
    # Runtime keys (underscore): derived from the pinned model spec, not part of the run config hash.
    for key, field in (('_q_prefix', 'query_prefix'), ('_d_prefix', 'doc_prefix')):
        cfg[key] = tok(spec[field], add_special_tokens=False)['input_ids'] if spec.get(field) else []
        cfg[key + '_text'] = spec.get(field, '')

    calib_names = [t.split('/')[0] + '/calib' for t in cfg['train_tasks']]
    calib = load_splits(cfg, family, calib_names)
    state_path = run / 'state.json'
    state = read_json(state_path) if state_path.exists() else {}
    ckpt = run / 'ckpt' / 'latest.pt'
    train_steps = 0

    # Step-0 prior scale. Trained values (checkpoint / model.pt) are loaded later and override this.
    if model.uses_bi():
        if 'alpha0' not in state:
            state['alpha0'] = zero_shot_alpha(model, calib, cfg, device)
            atomic_json(state_path, state)
        set_alpha(model, state['alpha0'])

    if cfg['eval_init'] and not state.get('init_done'):
        init_temps = fit_temperatures(E.calib_entries(model, calib, cfg, device))
        if not evaluate_all(model, cfg, family, run, device, init_temps, 'init', laya_temps=laya_temps):
            sys.exit(INCOMPLETE)
        state['init_done'] = True
        atomic_json(state_path, state)

    trainable = cfg['steps'] > 0 and model.arch != 'Z'
    if trainable and not state.get('train_done'):
        train = D.merge(load_splits(cfg, family, [t if '/' in t else t + '/train' for t in cfg['train_tasks']]))
        mix = D.Mixture(train, frac=cfg['frac'], alpha=cfg['mix_alpha'], seed=cfg['seed'])
        total = int(min(cfg['steps'], math.ceil(cfg['max_epochs'] * mix.total() / cfg['batch'])))
        state.update(pool_sizes=mix.sizes(), total_steps=total)
        opt = torch.optim.AdamW(model.trainable_groups(cfg['lr_enc'], cfg['lr_head'], cfg['lr_alpha']),
                                weight_decay=cfg['weight_decay'])
        base_lrs = [g['lr'] for g in opt.param_groups]
        step = 0
        resumed = ckpt.exists()
        if resumed:
            ck = torch.load(ckpt, map_location='cpu', weights_only=False)
            model.load_state_dict(ck['model'])
            opt.load_state_dict(ck['opt'])
            step = ck['step']
            random.setstate(ck['py_rng'])
            np.random.set_state(ck['np_rng'])
            torch.set_rng_state(ck['torch_rng'])
            if device.type == 'cuda' and ck.get('cuda_rng') is not None:
                torch.cuda.set_rng_state(ck['cuda_rng'])
            print(f'resumed at step {step}/{total}', flush=True)
        log = open(run / 'train_log.jsonl', 'a', encoding='utf-8')
        t_ckpt = t_start = time.monotonic()
        seen = 0
        model.train()

        def save_ckpt():
            ckpt.parent.mkdir(exist_ok=True)
            tmp = ckpt.with_suffix('.tmp')
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, py_rng=random.getstate(),
                            np_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state() if device.type == 'cuda' else None), tmp)
            os.replace(tmp, ckpt)

        # Tests / smoke: simulate a scheduler's TERM signal once, on the first attempt only, to exercise resume.
        stop_at = -1 if resumed else int(os.environ.get('S1_TEST_STOP_AT', cfg.get('test_stop_at', -1)))
        while step < total:
            if step == stop_at:
                STOP['flag'] = True
            if STOP['flag']:
                save_ckpt()
                atomic_json(state_path, state)
                print(f'SIGTERM: checkpointed at step {step}', flush=True)
                sys.exit(INCOMPLETE)
            progress = step / max(1, total)
            for g, base in zip(opt.param_groups, base_lrs):
                g['lr'] = lr_at(step, total, base, cfg['warmup'])
            sigma = O.sigma_at(progress, cfg['sigma_start'], cfg['sigma_end'])
            idxs, rng = mix.batch(step, cfg['batch'])
            orders = D.random_orders(train, idxs, rng, cfg['permute'])
            b = D.to_device(D.collate(train, idxs, orders, cross=model.uses_cross(), bi=model.uses_bi(),
                                      max_len=cfg['max_len'], head_max_len=cfg['head_max_len'],
                                      bi_state_len=cfg['bi_state_len'], q_prefix=cfg['_q_prefix'],
                                      d_prefix=cfg['_d_prefix']), device)
            with E._autocast(device, cfg['amp']):
                z, _ = model(b)
            z = z.float()
            loss, stats = O.loss(cfg['objective'], z, b['target'], b['marker_mask'], b['qtype'], sigma=sigma,
                                 G=cfg['G'], w_ce=cfg['w_ce'], w_sph=cfg['w_sph'], w_rps=cfg['w_rps'],
                                 w_score=cfg.get('w_score', 1.0))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']], 1.0)
            # Never apply a non-finite update: skip it, and fail the run loudly if it keeps happening, so a
            # diverged run cannot finish and look like a (degenerate) result.
            if not (torch.isfinite(loss) and torch.isfinite(gn)):
                bad = state.get('nonfinite_steps', 0) + 1
                state['nonfinite_steps'] = bad
                opt.zero_grad(set_to_none=True)
                if bad >= 20:
                    atomic_json(state_path, state)
                    sys.exit(f'aborting: {bad} non-finite training steps (loss={float(loss)}, grad_norm={float(gn)})')
                step += 1
                continue
            opt.step()
            step += 1
            seen += len(idxs)
            if step % 50 == 0 or step == total:
                el = time.monotonic() - t_start
                row = dict(step=step, loss=float(loss.detach()), grad_norm=float(gn), lr=opt.param_groups[0]['lr'],
                           sigma=sigma, items_per_s=seen / max(el, 1e-9), alpha=model.alpha.detach().tolist(), **stats)
                if cfg['grad_probe_every'] and step % cfg['grad_probe_every'] == 0:
                    row['grad_var'] = O.grad_variance_probe(cfg['objective'], z.detach(), b['target'], b['marker_mask'],
                                                            b['qtype'], sigma, G=cfg['G'], w_sph=cfg['w_sph'], w_rps=cfg['w_rps'],
                                                            w_score=cfg.get('w_score', 1.0))
                if device.type == 'cuda':
                    row['peak_gb'] = torch.cuda.max_memory_reserved() / 2 ** 30
                    atomic_json(run / 'peak_mem.json', dict(peak_gb=row['peak_gb'], step=step))
                log.write(json.dumps(row) + '\n')
                log.flush()
                atomic_json(run / 'throughput.json', dict(items_per_s=row['items_per_s'], step=step, elapsed_s=el))
            if time.monotonic() - t_ckpt > cfg['ckpt_minutes'] * 60:
                save_ckpt()
                t_ckpt = time.monotonic()
            if cfg['max_minutes'] and time.monotonic() - t_start > cfg['max_minutes'] * 60:
                print('max_minutes reached (throughput probe); stopping without evaluation', flush=True)
                (run / 'DONE').write_text('probe\n')
                return
        log.close()
        train_steps = step
        state['train_done'] = True
        # Always keep the trained weights until evaluation finishes, so an interrupted evaluation
        # resumes without retraining; deleted at the end unless save_model is set.
        torch.save(model.state_dict(), run / 'model.pt.tmp')
        os.replace(run / 'model.pt.tmp', run / 'model.pt')
        if ckpt.exists():
            ckpt.unlink()
        atomic_json(state_path, state)
    elif state.get('train_done') and (run / 'model.pt').exists():
        model.load_state_dict(torch.load(run / 'model.pt', map_location='cpu'))

    tpath = run / 'temps.json'
    if tpath.exists():
        temps = read_json(tpath)
    else:
        temps = fit_temperatures(E.calib_entries(model, calib, cfg, device))
        atomic_json(tpath, temps)
    if not evaluate_all(model, cfg, family, run, device, temps, 'final', tok=tok, laya_temps=laya_temps):
        sys.exit(INCOMPLETE)
    if not cfg['save_model'] and (run / 'model.pt').exists():
        (run / 'model.pt').unlink()            # disk: keep weights only for runs marked save_model
    (run / 'DONE').write_text(f'train_steps={train_steps}\n')
    print('done', run, flush=True)


if __name__ == '__main__':
    main()
