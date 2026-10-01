"""Expand the v1 (arXiv) experiment grid into queue tasks. Idempotent: existing tasks are kept.

    python scripts/enqueue_v1.py            # everything, priorities P0..P3
    python scripts/enqueue_v1.py --smoke    # tiny smoke grid in queue_smoke/ (set S1_QUEUE accordingly)

Priorities make a truncated campaign still produce a paper:
  P0  pre-tokenisation; S-PubMedBert pair x {PFR, C} seed 0; all zero-shot rows; Laya zero-shot; Qwen baseline
  P1  remaining encoders seed 0; bi-encoder B; RLCD decomposition; Laya fine-tuned; learning curves seed 0; Gemma
  P2  seeds 1-2 of every P0/P1 training row except learning curves
  P3  clinical variant; frozen prior; learning-curve seeds 1-2; latency
Run build_clinical.py BEFORE this script if the clinical track should be evaluated (eval splits are part
of each run's config hash).
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s1.common import atomic_json  # noqa: E402

TRAIN = ['pubmedqa', 'pubmedqa_art', 'scifact', 'healthver', 'ddi', 'hoc', 'ade', 'druglib', 'medqa', 'medmcqa', 'medline_s1']
EVAL_PUBLIC = ['pubmedqa', 'scifact', 'healthver', 'pubhealth', 'ddi', 'hoc', 'ade', 'druglib', 'mtsamples', 'biosses',
               'medqa', 'medmcqa', 'mmlu_med', 'medline_s1', 'typed_decisions']
EVAL_CLINICAL = ['mednli/test', 'mimic_trialq/test', 'mimic_deid/test', 'mimic_deid/polysemy']
CLINICAL_TRAIN = ['mednli', 'mimic_deid']
LONG = {'mimic_trialq/test': ['truncate', 'window_mean', 'window_conf', 'rtd']}
ENCODERS = ['spubmedbert', 'pubmedbert', 'medcpt_q', 'allmpnet', 'mpnet']
FAMILY = {'spubmedbert': 'pubmedbert', 'pubmedbert': 'pubmedbert', 'medcpt_q': 'pubmedbert',
          'mpnet': 'mpnet', 'allmpnet': 'mpnet', 'laya': 'laya_en',
          'modernbert': 'modernbert', 'mbembed': 'modernbert', 'gtemb': 'modernbert',
          'bcmb': 'modernbert', 'bcmbembed': 'modernbert', 'bge': 'bert_uncased'}
# RQ1 extension: modern matched pairs (ModernBERT -> two embedders; BioClinical-ModernBERT -> embedder)
# plus BGE-base as an unpaired modern embedder.
EXT_ENCODERS = ['mbembed', 'gtemb', 'modernbert', 'bcmbembed', 'bcmb', 'bge']
# Enforced per process (s1.common.apply_vram_cap). Measured peaks: base 6-16 GB, Laya-large 36 GB;
# LLMs = bf16 weights (54 / 62 GB) + activations for a 12k-token batch.
VRAM = {'base': 22, 'laya': 44, 'llm_qwen38_27b': 100, 'llm_gemma4_31b': 110}


def eval_splits(clinical):
    s = [f'{t}/test' for t in EVAL_PUBLIC]
    return s + (EVAL_CLINICAL if clinical else [])


def train_task(run_id, prio, cfg, after, est=2.0, vram=VRAM['base'], slots=1, key=None):
    run_dir = f'runs/{run_id}'
    atomic_json(ROOT / run_dir / 'input_config.json', cfg)
    return dict(run_id=run_id, kind='train', priority=prio, run_dir=run_dir,
                argv=['-m', 's1.train', '--run-dir', run_dir, '--config', f'{run_dir}/input_config.json'],
                vram_gb=vram, slots=slots, est_hours=est, after=after, vram_key=key or f"{cfg['encoder']}:{cfg['arch']}")


def grid(clinical, smoke=False):
    ev = eval_splits(clinical)
    base = dict(train_tasks=TRAIN, eval_splits=ev, long_eval={k: v for k, v in LONG.items() if k in ev},
                objective='rlcd_pg', steps=8000, batch=32)
    if smoke:
        base.update(train_tasks=['pubmedqa', 'ade'], eval_splits=['pubmedqa/test', 'ade/test'], long_eval={},
                    steps=200, eval_perms=2)
    tasks = []
    fams = sorted({FAMILY[e] for e in ENCODERS + ['laya']}) if not smoke else ['pubmedbert', 'laya_en']
    for fam in fams:
        tasks.append(dict(run_id=f'pretok/{fam}', kind='cmd', priority=0, argv=['-m', 's1.data', '--family', fam],
                          vram_gb=0, slots=0.5, est_hours=0.5, vram_key='pretok'))
    dep = lambda enc: [f'pretok/{FAMILY[enc]}']  # noqa: E731

    def add(run_id, prio, enc, **kw):
        cfg = dict(base, encoder=enc, **kw)
        vram = VRAM['laya'] if enc == 'laya' else VRAM['base']
        # Wall-time estimate under packing; the packer only starts a task if est*1.3+0.25 h still fits.
        est = 0.3 if smoke else 0.5 if cfg.get('frac', 1.0) < 0.2 or cfg.get('steps') == 0 else 2.0
        tasks.append(train_task(run_id, prio, cfg, dep(enc), est=est, vram=vram,
                                slots=2 if enc == 'laya' else 1))

    if smoke:
        add('smoke/spubmedbert/PFR/s0', 0, 'spubmedbert', arch='PFR', seed=0, eval_init=True, retention=['scifact'],
            test_stop_at=100)                          # forced TERM at step 100 -> requeue -> resume
        add('smoke/spubmedbert/C/s0', 0, 'spubmedbert', arch='C', seed=0, eval_init=True)
        add('smoke/spubmedbert/Z', 0, 'spubmedbert', arch='Z', steps=0)
        # smoke 2: after the alpha floor / alpha learning-rate fix (alpha must move; B must learn)
        add('smoke2/spubmedbert/B/s0', 0, 'spubmedbert', arch='B', seed=0)
        add('smoke2/spubmedbert/PFR/s0', 0, 'spubmedbert', arch='PFR', seed=0, eval_init=True)
        # Parity check on the real stack: Laya's README reports 0.362 accuracy zero-shot on typed-decisions.
        add('smoke/laya/C', 0, 'laya', arch='C', steps=0, eval_splits=['pubmedqa/test', 'typed_decisions/test'])
        add('smoke/laya/C_ft', 0, 'laya', arch='C', seed=0)
        tasks.append(dict(run_id='smoke/llm/qwen38_27b', kind='llm', priority=0, run_dir='runs/smoke/llm/qwen38_27b',
                          argv=['-m', 's1.llm_baseline', '--model', 'qwen38_27b', '--run-dir', 'runs/smoke/llm/qwen38_27b',
                                '--splits', 'pubmedqa/test', '--calib-tasks', 'pubmedqa'],
                          vram_gb=VRAM['llm_qwen38_27b'], slots=3, est_hours=0.5, after=[], vram_key='llm_qwen38_27b'))
        return tasks

    # P0 -------------------------------------------------------------------------------------
    for enc in ('spubmedbert', 'pubmedbert'):
        for arch in ('PFR', 'C'):
            add(f'main/{enc}/{arch}/s0', 0, enc, arch=arch, seed=0, eval_init=True, retention=['scifact', 'nfcorpus'],
                save_model=(enc == 'spubmedbert' and arch == 'PFR'))
    for enc in ENCODERS:
        add(f'zeroshot/{enc}/Z', 0, enc, arch='Z', steps=0, retention=['scifact', 'nfcorpus'])
    add('zeroshot/laya/C', 0, 'laya', arch='C', steps=0)
    llm_calib = TRAIN + (CLINICAL_TRAIN if clinical else [])
    for prio, slug in ((0, 'qwen38_27b'), (1, 'gemma4_31b')):
        tasks.append(dict(run_id=f'llm/{slug}', kind='llm', priority=prio, run_dir=f'runs/llm/{slug}',
                          argv=['-m', 's1.llm_baseline', '--model', slug, '--run-dir', f'runs/llm/{slug}',
                                '--splits', *ev, '--calib-tasks', *llm_calib],
                          vram_gb=VRAM[f'llm_{slug}'], slots=3, est_hours=6.0, after=[], vram_key=f'llm_{slug}'))
    # P1 -------------------------------------------------------------------------------------
    for enc in ('medcpt_q', 'allmpnet', 'mpnet'):
        for arch in ('PFR', 'C'):
            add(f'main/{enc}/{arch}/s0', 1, enc, arch=arch, seed=0, eval_init=True, retention=['scifact', 'nfcorpus'])
    for enc in ('spubmedbert', 'pubmedbert'):
        add(f'bi/{enc}/B/s0', 1, enc, arch='B', seed=0, retention=['scifact', 'nfcorpus'])
    rq3 = {'ce': dict(objective='ce'), 'proper': dict(objective='proper'),
           'reparam': dict(objective='rlcd_reparam'), 'pg_noce': dict(objective='rlcd_pg', w_ce=0.0)}
    for name, kw in rq3.items():
        add(f'rq3/{name}/s0', 1, 'spubmedbert', arch='PFR', seed=0, **kw)
    add('laya_ft/s0', 1, 'laya', arch='C', seed=0)
    for enc in ENCODERS:
        for arch in ('PFR', 'C'):
            for frac in (0.02, 0.1):
                add(f'lc/{enc}/{arch}/f{frac}/s0', 1, enc, arch=arch, seed=0, frac=frac)
    # P2 -------------------------------------------------------------------------------------
    for s in (1, 2):
        for enc in ENCODERS:
            for arch in ('PFR', 'C'):
                add(f'main/{enc}/{arch}/s{s}', 2, enc, arch=arch, seed=s)
        for enc in ('spubmedbert', 'pubmedbert'):
            add(f'bi/{enc}/B/s{s}', 2, enc, arch='B', seed=s)
        for name, kw in rq3.items():
            add(f'rq3/{name}/s{s}', 2, 'spubmedbert', arch='PFR', seed=s, **kw)
        add(f'laya_ft/s{s}', 2, 'laya', arch='C', seed=s)
    # P3 -------------------------------------------------------------------------------------
    for s in (0, 1, 2):
        if clinical:
            add(f'clinical/spubmedbert/PFR/s{s}', 3, 'spubmedbert', arch='PFR', seed=s, train_tasks=TRAIN + CLINICAL_TRAIN,
                save_model=False)
        add(f'frozen/spubmedbert/PFR/s{s}', 3, 'spubmedbert', arch='PFR', prior='frozen', seed=s,
            retention=['scifact', 'nfcorpus'] if s == 0 else [])
    for s in (1, 2):
        for enc in ENCODERS:
            for arch in ('PFR', 'C'):
                for frac in (0.02, 0.1):
                    add(f'lc/{enc}/{arch}/f{frac}/s{s}', 3, enc, arch=arch, seed=s, frac=frac)
    tasks.append(dict(run_id='latency', kind='cmd', priority=3, exclusive=True, run_dir='runs/latency',
                      argv=['-m', 's1.latency', '--out', 'runs/latency'], vram_gb=100, slots=6, est_hours=1.0,
                      after=['pretok/pubmedbert', 'pretok/laya_en'], vram_key='latency'))
    return tasks


def grid_ext(clinical):
    """RQ1 extension. A 200-step smoke run per new architecture family gates everything else: if it
    fails, the dependent runs never become ready and the workers exit instead of wasting GPU time."""
    ev = eval_splits(clinical)
    base = dict(train_tasks=TRAIN, eval_splits=ev, long_eval={k: v for k, v in LONG.items() if k in ev},
                objective='rlcd_pg', steps=8000, batch=32)
    tasks = [dict(run_id=f'pretok/{fam}', kind='cmd', priority=0, argv=['-m', 's1.data', '--family', fam],
                  vram_gb=0, slots=0.5, est_hours=0.5, vram_key='pretok') for fam in ('modernbert', 'bert_uncased')]

    def add(run_id, prio, enc, after=(), est=2.0, **kw):
        cfg = dict(base, encoder=enc, **kw)
        tasks.append(train_task(run_id, prio, cfg, [f'pretok/{FAMILY[enc]}', *after], est=est, vram=24))

    smoke = dict(train_tasks=['pubmedqa', 'ade'], eval_splits=['pubmedqa/test', 'ade/test'], long_eval={},
                 steps=200, eval_perms=0)
    for enc in ('mbembed', 'bge'):
        for arch in ('C', 'PFR'):
            add(f'ext_smoke/{enc}/{arch}', 0, enc, est=0.3, arch=arch, seed=0, **smoke)
    gate = lambda enc: [f'ext_smoke/{"bge" if enc == "bge" else "mbembed"}/{a}' for a in ('C', 'PFR')]  # noqa: E731
    for enc in EXT_ENCODERS:
        add(f'zeroshot/{enc}/Z', 0, enc, after=gate(enc), est=0.5, arch='Z', steps=0, retention=['scifact', 'nfcorpus'])
        for arch in ('C', 'PFR'):
            add(f'main/{enc}/{arch}/s0', 0, enc, after=gate(enc), arch=arch, seed=0, eval_init=True,
                retention=['scifact', 'nfcorpus'])
            for s in (1, 2):
                add(f'main/{enc}/{arch}/s{s}', 1, enc, after=gate(enc), arch=arch, seed=s)
            if enc != 'bge':
                add(f'lc/{enc}/{arch}/f0.1/s0', 1, enc, after=gate(enc), est=0.5, arch=arch, seed=0, frac=0.1)
    return tasks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--ext', action='store_true', help='RQ1 extension grid (ModernBERT pairs + BGE)')
    ap.add_argument('--max-priority', type=int, default=3)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    clinical = (ROOT / 'prepared' / 'mednli' / 'test.jsonl').exists()
    tasks = grid_ext(clinical) if a.ext else grid(clinical, a.smoke)
    tasks = [t for t in tasks if t['priority'] <= a.max_priority]
    by_p = {}
    for t in tasks:
        by_p[t['priority']] = by_p.get(t['priority'], 0) + 1
    print(f'clinical track: {"yes" if clinical else "NO (prepared/mednli missing)"}; tasks per priority: {by_p}')
    if a.dry_run:
        print(json.dumps([t['run_id'] for t in tasks], indent=0)[:3000])
        return
    from s1.queue import enqueue
    print('enqueued', enqueue(tasks), 'new tasks into', os.environ.get('S1_QUEUE', ROOT / 'queue'))


if __name__ == '__main__':
    main()
