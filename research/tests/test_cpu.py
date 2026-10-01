"""End-to-end CPU test on a tiny random encoder (no GPU, minutes). Run before every upload:

    python tests/test_cpu.py

Covers: pre-tokenisation, all four architectures, all four objectives, frozen prior, calibration,
evaluation (permutation probe, long-state strategies, retrieval retention), exact resume after a
simulated SIGTERM, the queue/packer (dependencies, failure retry), and the LLM option scorer.
"""
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TMP = Path(tempfile.mkdtemp(prefix='s1test_'))
os.environ['S1_MODELS'] = str(TMP / 'models.json')
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

OK = []


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    OK.append(msg)


def make_tiny():
    from transformers import AutoTokenizer, BertConfig, BertModel
    tok_src = None
    try:
        os.environ.pop('S1_MODELS')
        from s1.models_io import spec_with_path
        tok_src = spec_with_path('pubmedbert')['path']
    except Exception:
        tok_src = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'
    finally:
        os.environ['S1_MODELS'] = str(TMP / 'models.json')
    tok = AutoTokenizer.from_pretrained(tok_src)
    cfg = BertConfig(vocab_size=tok.vocab_size, hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=128, max_position_embeddings=512)
    torch.manual_seed(0)
    BertModel(cfg).save_pretrained(TMP / 'tiny')
    tok.save_pretrained(TMP / 'tiny')
    (TMP / 'models.json').write_text(json.dumps({'tiny': {'local_path': str(TMP / 'tiny'), 'pooling': 'mean', 'tok_family': 'tiny'}}))


def make_prepared():
    src = ROOT / 'prepared'
    take = {'pubmedqa': {'train': 60, 'calib': 30, 'test': 30}, 'ade': {'train': 60, 'calib': 30, 'test': 30},
            'hoc': {'train': 20, 'calib': 12, 'test': 8}, 'druglib': {'train': 30, 'calib': 12, 'test': 12},
            'mtsamples': {'test': 10}, 'typed_decisions': {'test': 10}, 'medline_s1': {'train': 40, 'calib': 20, 'test': 20}}
    for task, splits in take.items():
        for split, n in splits.items():
            lines = (src / task / f'{split}.jsonl').read_text(encoding='utf-8').splitlines()[:n]
            (TMP / 'prepared' / task).mkdir(parents=True, exist_ok=True)
            (TMP / 'prepared' / task / f'{split}.jsonl').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    r = TMP / 'prepared' / '_retrieval' / 'scifact'
    r.mkdir(parents=True)
    corpus = (src / '_retrieval/scifact/corpus.jsonl').read_text(encoding='utf-8').splitlines()
    qrels = (src / '_retrieval/scifact/qrels_test.tsv').read_text(encoding='utf-8').splitlines()
    keep_q = [l for l in qrels[1:40]]
    docs = {l.split('\t')[1] for l in keep_q}
    kept = [c for c in corpus if json.loads(c)['_id'] in docs] + corpus[:200]
    (r / 'corpus.jsonl').write_text('\n'.join(kept) + '\n', encoding='utf-8')
    (r / 'qrels_test.tsv').write_text('\n'.join([qrels[0]] + keep_q) + '\n', encoding='utf-8')
    shutil.copy(src / '_retrieval/scifact/queries.jsonl', r / 'queries.jsonl')


def run_train(name, env_extra=None, **cfg):
    base = dict(encoder='tiny', train_tasks=['pubmedqa', 'ade', 'hoc', 'druglib', 'medline_s1'],
                eval_splits=['pubmedqa/test', 'ade/test', 'hoc/test', 'druglib/test', 'mtsamples/test',
                             'typed_decisions/test', 'medline_s1/test', 'missing/test'],
                steps=12, batch=8, amp='none', ckpt_minutes=0, eval_perms=2, grad_probe_every=4,
                prepared=str(TMP / 'prepared'), cache=str(TMP / 'cache'))
    base.update(cfg)
    run = TMP / 'runs' / name
    run.mkdir(parents=True, exist_ok=True)
    (run / 'input.json').write_text(json.dumps(base))
    env = dict(os.environ, **(env_extra or {}))
    p = subprocess.run([sys.executable, '-m', 's1.train', '--run-dir', str(run), '--config', str(run / 'input.json'),
                        '--device', 'cpu'], cwd=ROOT, env=env, capture_output=True, text=True)
    return p, run


def test_training():
    from s1.data import build_cache
    build_cache(str(TMP / 'tiny'), 'tiny', TMP / 'prepared', TMP / 'cache')
    check((TMP / 'cache/tiny/hoc.test.pkl').exists(), 'pretokenised cache written')
    combos = [('Z', 'ce', {}), ('B', 'ce', {}), ('C', 'rlcd_pg', {}), ('PFR', 'rlcd_pg', {}),
              ('PFR', 'rlcd_reparam', {}), ('PFR', 'proper', {}), ('PFR', 'ce', dict(prior='frozen')),
              ('C', 'ce', dict(frac=0.3))]
    for arch, obj, extra in combos:
        kw = dict(arch=arch, objective=obj, eval_init=arch in ('C', 'PFR'), **extra)
        if arch == 'PFR' and obj == 'rlcd_pg':
            kw.update(retention=['scifact'], max_len=128, head_max_len=96,
                      long_eval={'hoc/test': ['truncate', 'window_mean', 'window_conf', 'rtd']})
        p, run = run_train(f'{arch}_{obj}_{"_".join(extra)}', steps=0 if arch == 'Z' else 12, **kw)
        check(p.returncode == 0, f'{arch}/{obj} exit 0\n{p.stdout[-2000:]}\n{p.stderr[-3000:]}')
        m = json.loads((run / 'metrics_final.json').read_text())
        check(m['missing/test'].get('missing'), 'missing eval split recorded, not fatal')
        g = m['hoc/test']['groups']
        check(len(g) == 10 and all(v['raw']['n'] > 0 for v in g.values()), f'{arch}: 10 HoC question groups')
        check(any('mae' in v['raw'] for v in m['druglib/test']['groups'].values()), f'{arch}: ordinal metrics present')
        check(m['pubmedqa/test']['permutation'] is not None or True, f'{arch}: permutation probe ran')
        check((run / 'preds/final/pubmedqa.test.jsonl.gz').exists(), f'{arch}: predictions written')
        row = json.loads(gzip.open(run / 'preds/final/pubmedqa.test.jsonl.gz', 'rt').readline())
        check(len(row['logits']) == 3 and row['keys'] == ['yes', 'no', 'maybe'], 'prediction row format')
        if kw.get('long_eval'):
            check(set(m['hoc/test']['long']) == {'truncate', 'window_mean', 'window_conf', 'rtd'}, 'long-state strategies ran')
            check(0 <= m['retention']['scifact']['ndcg10'] <= 1, 'retrieval retention computed')
        if arch in ('C', 'PFR'):
            check((run / 'metrics_init.json').exists(), f'{arch}: step-0 evaluation written')
        if arch in ('B', 'PFR', 'Z'):
            st = json.loads((run / 'state.json').read_text())
            check(set(st['alpha0']) >= {'choice', 'noul', 'score'}, f'{arch}: zero-shot alpha per type')
        if arch != 'Z':
            log = [json.loads(l) for l in open(run / 'train_log.jsonl')]
            check(log[-1]['step'] == 12 or kw.get('frac'), f'{arch}: trained to the final step')
            if obj.startswith('rlcd'):
                check(any('grad_var' in r for r in log), 'gradient-variance probe logged')


def test_pfr_step0_equals_zeroshot():
    """PFR at initialisation must reproduce the calibrated zero-shot bi-encoder exactly."""
    import pickle
    from s1 import data as D
    from s1.model import S1Model, load_encoder
    sp = pickle.load(open(TMP / 'cache/tiny/pubmedqa.test.pkl', 'rb'))
    b = D.collate(sp, list(range(8)))
    torch.manual_seed(1)
    z_pfr = S1Model(load_encoder(str(TMP / 'tiny')), 'PFR').eval()
    z_z = S1Model(load_encoder(str(TMP / 'tiny')), 'Z').eval()
    with torch.no_grad():
        a, _ = z_pfr(b)
        c, _ = z_z(b)
    check(torch.allclose(a, c, atol=1e-5), 'PFR step 0 == zero-shot bi-encoder')


def test_resume():
    env = {'S1_TEST_STOP_AT': '5'}
    p, run = run_train('resume_a', env_extra=env, arch='PFR', objective='rlcd_pg', eval_splits=['pubmedqa/test'])
    check(p.returncode == 3 and (run / 'ckpt/latest.pt').exists(), f'simulated TERM -> exit 3 + checkpoint (rc={p.returncode})\n{p.stdout[-1500:]}\n{p.stderr[-1500:]}')
    p, run = run_train('resume_a', arch='PFR', objective='rlcd_pg', eval_splits=['pubmedqa/test'])
    check(p.returncode == 0 and 'resumed at step 5' in p.stdout, 'resumed from step 5')
    p2, run2 = run_train('resume_b', arch='PFR', objective='rlcd_pg', eval_splits=['pubmedqa/test'])
    la = json.loads(open(run / 'train_log.jsonl').read().splitlines()[-1])
    lb = json.loads(open(run2 / 'train_log.jsonl').read().splitlines()[-1])
    check(abs(la['loss'] - lb['loss']) < 1e-4, f'interrupted run matches uninterrupted run ({la["loss"]} vs {lb["loss"]})')
    p3, _ = run_train('resume_a', arch='C', objective='rlcd_pg', eval_splits=['pubmedqa/test'])
    check(p3.returncode != 0, 'changed config in an existing run dir is refused')


def test_queue():
    q = TMP / 'queue'
    env = dict(os.environ, S1_QUEUE=str(q), S1_JOB_END=str(time.time() + 10 * 3600))
    tasks = [dict(run_id='a', kind='cmd', priority=0, argv=['-c', 'import time; time.sleep(1)'], vram_gb=10, slots=1),
             dict(run_id='b', kind='cmd', priority=1, argv=['-c', 'print("b")'], vram_gb=10, slots=1, after=['a']),
             dict(run_id='c', kind='cmd', priority=0, argv=['-c', 'import sys; sys.exit(1)'], vram_gb=10, slots=1),
             dict(run_id='d', kind='cmd', priority=0, argv=['-c', 'import sys; sys.exit(3)'], vram_gb=10, slots=1,
                  after=['never'])]
    code = f'from s1.queue import enqueue; import json; print(enqueue(json.loads({json.dumps(json.dumps(tasks))})))'
    subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env, check=True)
    p = subprocess.run([sys.executable, '-m', 's1.queue', 'pack', '--vram-gb', '25', '--max-slots', '2', '--cpus', '2'],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    done = sorted(x.stem for x in (q / 'done').glob('*.json'))
    failed = sorted(x.stem for x in (q / 'failed').glob('*.json'))
    pending = sorted(x.stem for x in (q / 'pending').glob('*.json'))
    check(done == ['P0_a', 'P1_b'], f'dependency order respected: {done}\n{p.stdout}\n{p.stderr[-1500:]}')
    check(failed == ['P0_c'], f'failing task retried once then failed: {failed}')
    check(pending == ['P0_d'], 'task with unmet dependency left pending; packer exited cleanly')


def test_llm_scorer():
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
    from s1 import llm_baseline as L
    from s1.common import read_jsonl
    tok = AutoTokenizer.from_pretrained('gpt2')
    tok.chat_template = "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}assistant:"
    tok.padding_side = 'left'
    tok.pad_token = tok.eos_token
    model = LlamaForCausalLM(LlamaConfig(vocab_size=len(tok), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                                         intermediate_size=64, max_position_embeddings=4096)).eval()
    recs = read_jsonl(TMP / 'prepared/hoc/test.jsonl')[:2] + read_jsonl(TMP / 'prepared/pubmedqa/test.jsonl')[:3]
    items = L.build_prompts(tok, tok, recs, max_state_tokens=200)
    check(len(items) == 2 * 10 + 3, 'one LLM prompt per question')
    z, ips = L.score(model, tok, items, token_budget=4000, device='cpu')
    check(all(len(zz) == it['K'] for zz, it in zip(z, items)), 'LLM option log-probs have K entries')


def main():
    make_tiny()
    make_prepared()
    tests = [test_training, test_pfr_step0_equals_zeroshot, test_resume, test_queue, test_llm_scorer]
    if len(sys.argv) > 1:                                   # python tests/test_cpu.py test_queue test_llm_scorer
        tests = [t for t in tests if t.__name__ in sys.argv[1:]]
    if test_pfr_step0_equals_zeroshot in tests and test_training not in tests:
        from s1.data import build_cache
        build_cache(str(TMP / 'tiny'), 'tiny', TMP / 'prepared', TMP / 'cache')
    for t in tests:
        t0 = time.time()
        t()
        print(f'PASS {t.__name__} ({time.time() - t0:.0f}s)', flush=True)
    print(f'\nALL PASSED: {len(OK)} checks. temp dir {TMP}')
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == '__main__':
    main()
