"""Every run in the paper as a plain list of experiments, plus a simple local runner.

    python experiments.py list                         # all runs, grouped
    python experiments.py list --group rq3c            # one group
    python experiments.py run --match main/spubmedbert # run matching experiments one after another
    python experiments.py commands --group lc > jobs.txt   # one shell command per line, for any scheduler

Groups: zeroshot, main, bi, rq3 (PFR x objective), rq3c (C x objective), frozen, clinical, laya, lc
(learning curves), llm, probe, latency. Run IDs match the paper's run directories under runs/, which the
analysis and table scripts read.

`run` executes on whatever device PyTorch finds (CUDA, ROCm or CPU) and skips experiments that already
finished (runs/<id>/DONE). Training resumes from its last checkpoint if interrupted, so re-running the
same command is safe. `commands` prints the exact commands instead, so you can hand them to SLURM, a job
array, GNU parallel or anything else; run the printed pre-tokenisation commands first.

`--smoke` writes a tiny version of each training run (200 steps, two tasks) under runs_smoke/ to check an
installation end to end.
"""
import argparse
import fnmatch
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

TRAIN = ['pubmedqa', 'pubmedqa_art', 'scifact', 'healthver', 'ddi', 'hoc', 'ade', 'druglib', 'medqa', 'medmcqa',
         'medline_s1']
EVAL_PUBLIC = ['pubmedqa', 'scifact', 'healthver', 'pubhealth', 'ddi', 'hoc', 'ade', 'druglib', 'mtsamples', 'biosses',
               'medqa', 'medmcqa', 'mmlu_med', 'medline_s1', 'typed_decisions']
EVAL_CLINICAL = ['mednli/test', 'mimic_trialq/test', 'mimic_deid/test', 'mimic_deid/polysemy']
CLINICAL_TRAIN = ['mednli', 'mimic_deid']
LONG = {'mimic_trialq/test': ['truncate', 'window_mean', 'window_conf', 'rtd']}

# Encoders (slugs from configs/models.json) and their tokenizer family (one pre-tokenised cache each).
FAMILY = {'pubmedbert': 'pubmedbert', 'spubmedbert': 'pubmedbert', 'medcpt_q': 'pubmedbert',
          'mpnet': 'mpnet', 'allmpnet': 'mpnet',
          'modernbert': 'modernbert', 'mbembed': 'modernbert', 'gtemb': 'modernbert',
          'bcmb': 'modernbert', 'bcmbembed': 'modernbert', 'bge': 'bert_uncased', 'laya': 'laya_en'}
ENCODERS = ['pubmedbert', 'spubmedbert', 'medcpt_q', 'mpnet', 'allmpnet', 'modernbert', 'mbembed', 'gtemb',
            'bcmb', 'bcmbembed', 'bge']
PAIRED = [e for e in ENCODERS if e != 'bge']          # parent-retriever pairs get learning curves
SEEDS = (0, 1, 2)
RETENTION = ['scifact', 'nfcorpus']

# Objectives of the head x objective grid on S-PubMedBERT-MS-MARCO.
OBJECTIVES = {'ce': dict(objective='ce'), 'proper': dict(objective='proper'),
              'reparam': dict(objective='rlcd_reparam'), 'pg_loo': dict(objective='rlcd_pg_loo')}
LLMS = ['qwen38_27b', 'gemma4_31b']


def clinical_available():
    return (ROOT / 'prepared' / 'mednli' / 'test.jsonl').exists()


def base_config(clinical):
    ev = [f'{t}/test' for t in EVAL_PUBLIC] + (EVAL_CLINICAL if clinical else [])
    return dict(train_tasks=TRAIN, eval_splits=ev, long_eval={k: v for k, v in LONG.items() if k in ev},
                objective='rlcd_pg', steps=8000, batch=32)


def experiments(clinical):
    """Return a list of dicts: id, group, kind ('train' | 'llm' | 'cmd'), family, and config or argv."""
    base, out = base_config(clinical), []

    def train(group, run_id, enc, **kw):
        out.append(dict(id=run_id, group=group, kind='train', family=FAMILY[enc], config=dict(base, encoder=enc, **kw)))

    for enc in ENCODERS:
        train('zeroshot', f'zeroshot/{enc}/Z', enc, arch='Z', steps=0, retention=RETENTION)
    train('zeroshot', 'zeroshot/laya/C', 'laya', arch='C', steps=0)
    for enc in ENCODERS:
        for arch in ('C', 'PFR'):
            for s in SEEDS:
                extra = dict(eval_init=True, retention=RETENTION) if s == 0 else {}
                train('main', f'main/{enc}/{arch}/s{s}', enc, arch=arch, seed=s, **extra)
    for enc in ('pubmedbert', 'spubmedbert'):
        for s in SEEDS:
            train('bi', f'bi/{enc}/B/s{s}', enc, arch='B', seed=s, **(dict(retention=RETENTION) if s == 0 else {}))
    for s in SEEDS:
        for name, kw in OBJECTIVES.items():
            train('rq3', f'rq3/{name}/s{s}', 'spubmedbert', arch='PFR', seed=s,
                  save_model=(name == 'pg_loo' and s == 0), **kw)
            train('rq3c', f'rq3c/{name}/s{s}', 'spubmedbert', arch='C', seed=s, save_model=(name == 'ce'),
                  **(dict(retention=RETENTION) if name == 'ce' and s == 0 else {}), **kw)
        train('rq3', f'rq3/pg_noce/s{s}', 'spubmedbert', arch='PFR', seed=s, objective='rlcd_pg', w_ce=0.0)
        train('frozen', f'frozen/spubmedbert/PFR/s{s}', 'spubmedbert', arch='PFR', prior='frozen', seed=s,
              **(dict(retention=RETENTION) if s == 0 else {}))
        if clinical:
            train('clinical', f'clinical/spubmedbert/PFR/s{s}', 'spubmedbert', arch='PFR', seed=s,
                  train_tasks=TRAIN + CLINICAL_TRAIN)
        train('laya', f'laya_ft/s{s}', 'laya', arch='C', seed=s)
    for enc in PAIRED:
        for arch in ('C', 'PFR'):
            for frac in (0.02, 0.1):
                for s in SEEDS:
                    train('lc', f'lc/{enc}/{arch}/f{frac}/s{s}', enc, arch=arch, seed=s, frac=frac)
    calib = TRAIN + (CLINICAL_TRAIN if clinical else [])
    for slug in LLMS:
        out.append(dict(id=f'llm/{slug}', group='llm', kind='llm', family=None,
                        argv=['-m', 's1.llm_baseline', '--model', slug, '--run-dir', f'runs/llm/{slug}',
                              '--splits', *base['eval_splits'], '--calib-tasks', *calib]))
    out.append(dict(id='probe/gradvar', group='probe', kind='cmd', family=None,
                    argv=['-m', 's1.grad_probe', '--out', 'runs/probe/gradvar', '--runs',
                          'runs/main/spubmedbert/PFR/s0', 'runs/rq3/pg_loo/s0', 'runs/rq3c/ce/s0'],
                    needs=['main/spubmedbert/PFR/s0', 'rq3/pg_loo/s0', 'rq3c/ce/s0']))
    out.append(dict(id='latency', group='latency', kind='cmd', family=None,
                    argv=['-m', 's1.latency', '--out', 'runs/latency']))
    # The gradient probe needs the PFR + PG checkpoint of main/spubmedbert/PFR/s0.
    for e in out:
        if e['id'] == 'main/spubmedbert/PFR/s0':
            e['config']['save_model'] = True
    return out


def smoke_version(e):
    e = json.loads(json.dumps(e))
    if e['kind'] == 'train':
        e['config'].update(train_tasks=['pubmedqa', 'ade'], eval_splits=['pubmedqa/test', 'ade/test'],
                           long_eval={}, retention=[], eval_perms=2,
                           steps=0 if e['config'].get('steps') == 0 else 200)
    return e


def select(exps, groups, patterns):
    if groups:
        exps = [e for e in exps if e['group'] in groups]
    if patterns:
        exps = [e for e in exps if any(fnmatch.fnmatch(e['id'], p) or p in e['id'] for p in patterns)]
    return exps


def run_dir(e, runs):
    return f'{runs}/{e["id"]}'


def command(e, runs, exe=sys.executable):
    if e['kind'] == 'train':
        d = run_dir(e, runs)
        return [exe, '-m', 's1.train', '--run-dir', d, '--config', f'{d}/input_config.json']
    argv = [a.replace('runs/', f'{runs}/', 1) if a.startswith('runs/') else a for a in e['argv']]
    return [exe, *argv]


def pretok_command(family, exe=sys.executable):
    return [exe, '-m', 's1.data', '--family', family]


def write_config(e, runs):
    if e['kind'] != 'train':
        return
    d = ROOT / run_dir(e, runs)
    d.mkdir(parents=True, exist_ok=True)
    (d / 'input_config.json').write_text(json.dumps(e['config'], indent=1))


def pretok_done(family):
    d = ROOT / 'cache' / 'pretok' / family
    return d.is_dir() and any(d.iterdir())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('action', choices=['list', 'run', 'commands'])
    ap.add_argument('--group', nargs='*', default=[], help='restrict to these groups')
    ap.add_argument('--match', nargs='*', default=[], help='restrict to IDs matching these patterns (glob or substring)')
    ap.add_argument('--smoke', action='store_true', help='tiny runs under runs_smoke/ to test an installation')
    ap.add_argument('--dry-run', action='store_true', help='with run: print what would run')
    a = ap.parse_args()

    clinical = clinical_available()
    exps = select(experiments(clinical), a.group, a.match)
    runs = 'runs_smoke' if a.smoke else 'runs'
    if a.smoke:
        exps = [smoke_version(e) for e in exps if e['kind'] == 'train']

    if a.action == 'list':
        groups = {}
        for e in exps:
            groups.setdefault(e['group'], []).append(e['id'])
        print(f'clinical track: {"available" if clinical else "not built (public tasks only)"}')
        for g, ids in groups.items():
            print(f'\n[{g}] {len(ids)} runs')
            for i in ids:
                done = (ROOT / runs / i / 'DONE').exists()
                print(f'  {"done " if done else "     "}{i}', flush=True)
        print(f'\n{len(exps)} experiments')
        return

    families = sorted({e['family'] for e in exps if e['family']})
    if a.action == 'commands':
        for f in families:
            print(shlex.join(pretok_command(f, 'python')))
        for e in exps:
            write_config(e, runs)
            print(shlex.join(command(e, runs, 'python')))
        return

    for f in families:
        if not pretok_done(f):
            print('pre-tokenising', f, flush=True)
            if not a.dry_run:
                subprocess.run(pretok_command(f), cwd=ROOT, check=True)
    for e in exps:
        if (ROOT / runs / e['id'] / 'DONE').exists():
            print('done   ', e['id'])
            continue
        missing = [n for n in e.get('needs', []) if not (ROOT / runs / n / 'DONE').exists()]
        if missing:
            print('skip   ', e['id'], '(needs', ', '.join(missing) + ')')
            continue
        print('run    ', e['id'], flush=True)
        if a.dry_run:
            continue
        write_config(e, runs)
        subprocess.run(command(e, runs), cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
