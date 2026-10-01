"""Patch 8 end-to-end CPU test on a tiny random encoder (login node, a few minutes, no GPU):

    python tests/test_patch8.py

Trains PFR with the new leave-one-out objective and C with CE / Reparam for a few steps (save_model on),
runs the matched gradient probe on the saved checkpoints, and dry-runs the patch-8 grid.
The tokenizer is the staged PubMedBERT one (configs/models.lock.json), or S1_TEST_TOKENIZER if set.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TMP = Path(tempfile.mkdtemp(prefix='s1p8_'))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

OK = []


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    OK.append(msg)
    print('  ok', msg.splitlines()[0], flush=True)


def make_tiny():
    from transformers import AutoTokenizer, BertConfig, BertModel
    tok_src = os.environ.get('S1_TEST_TOKENIZER')
    if not tok_src:
        from s1.models_io import spec_with_path
        tok_src = spec_with_path('pubmedbert')['path']
    tok = AutoTokenizer.from_pretrained(tok_src)
    cfg = BertConfig(vocab_size=tok.vocab_size, hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=128, max_position_embeddings=512)
    torch.manual_seed(0)
    BertModel(cfg).save_pretrained(TMP / 'tiny')
    tok.save_pretrained(TMP / 'tiny')
    (TMP / 'models.json').write_text(json.dumps(
        {'tiny': {'local_path': str(TMP / 'tiny'), 'pooling': 'mean', 'tok_family': 'tiny'}}))


def make_prepared():
    src = ROOT / 'prepared'
    for task in ('pubmedqa', 'ade', 'hoc', 'druglib', 'medline_s1'):
        for split, n in (('train', 60), ('calib', 30), ('test', 30)):
            lines = (src / task / f'{split}.jsonl').read_text(encoding='utf-8').splitlines()[:n]
            (TMP / 'prepared' / task).mkdir(parents=True, exist_ok=True)
            (TMP / 'prepared' / task / f'{split}.jsonl').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def env():
    return dict(os.environ, S1_MODELS=str(TMP / 'models.json'), PYTHONPATH=str(ROOT))


def run_train(name, **cfg):
    base = dict(encoder='tiny', train_tasks=['pubmedqa', 'ade', 'hoc', 'druglib', 'medline_s1'],
                eval_splits=['pubmedqa/test', 'hoc/test', 'druglib/test'], steps=12, batch=8, amp='none',
                ckpt_minutes=0, eval_perms=0, grad_probe_every=4, save_model=True,
                prepared=str(TMP / 'prepared'), cache=str(TMP / 'cache'))
    base.update(cfg)
    run = TMP / 'runs' / name
    run.mkdir(parents=True, exist_ok=True)
    (run / 'input.json').write_text(json.dumps(base))
    p = subprocess.run([sys.executable, '-m', 's1.train', '--run-dir', str(run), '--config', str(run / 'input.json'),
                        '--device', 'cpu'], cwd=ROOT, env=env(), capture_output=True, text=True)
    check(p.returncode == 0, f'{name}: training exit 0\n{p.stdout[-1500:]}\n{p.stderr[-3000:]}')
    check((run / 'metrics_final.json').exists(), f'{name}: metrics written')
    check((run / 'model.pt').exists() == bool(base['save_model']), f'{name}: model.pt kept iff save_model')
    return run


def main():
    print('patch 8 CPU test in', TMP, flush=True)
    make_tiny()
    make_prepared()
    os.environ['S1_MODELS'] = str(TMP / 'models.json')
    from s1.data import build_cache
    build_cache(str(TMP / 'tiny'), 'tiny', TMP / 'prepared', TMP / 'cache')
    r_loo = run_train('pfr_loo', arch='PFR', objective='rlcd_pg_loo')
    log = [json.loads(l) for l in open(r_loo / 'train_log.jsonl')]
    check(any('grad_var' in r for r in log), 'LOO run logs gradient variance')
    r_ce = run_train('c_ce', arch='C', objective='ce')
    run_train('c_reparam', arch='C', objective='rlcd_reparam', save_model=False)
    out = TMP / 'probe'
    p = subprocess.run([sys.executable, '-m', 's1.grad_probe', '--device', 'cpu', '--out', str(out), '--batches', '2',
                        '--draws', '4', '--ref-draws', '1', '--sigmas', '0.4', '0.1',
                        '--runs', str(r_loo), str(r_ce), str(TMP / 'runs' / 'c_reparam')],
                       cwd=ROOT, env=env(), capture_output=True, text=True)
    check(p.returncode == 0, f'grad_probe exit 0\n{p.stdout[-2000:]}\n{p.stderr[-3000:]}')
    res = json.loads((out / 'grad_probe.json').read_text())
    check(res['runs'][str(TMP / 'runs' / 'c_reparam')].get('missing') == 'model.pt', 'probe skips runs without model.pt')
    est = res['runs'][str(r_loo)]['sigma']['0.4']
    check(set(est) >= {'rlcd_pg', 'pg_mean', 'rlcd_pg_loo', 'rlcd_reparam'}, 'probe reports all four estimators')
    check(all(abs(v) < 1e9 for e in est.values() for v in e.values()), 'probe values finite')
    p = subprocess.run([sys.executable, 'scripts/enqueue_patch8.py', '--dry-run', '--max-priority', '2'],
                       cwd=ROOT, env=env(), capture_output=True, text=True)
    if 'prepared/mednli/test.jsonl missing' in p.stdout + p.stderr:
        print('  (grid dry-run skipped: no clinical track in this copy of prepared/)')
    else:
        check(p.returncode == 0 and 'rq3c/ce/s0' in p.stdout and 'probe/gradvar' in p.stdout,
              f'grid dry-run\n{p.stdout[-1500:]}{p.stderr[-1500:]}')
    print(f'\nALL {len(OK)} CHECKS PASSED')


if __name__ == '__main__':
    main()
