"""Generative-LLM baseline in the same typed-decision harness: one forward pass per question, option
distribution = softmax over the next-token logits of the option letters (no sampling, no parsing).

    python -m s1.llm_baseline --model qwen38_27b --run-dir runs/llm/qwen38_27b --splits pubmedqa/test ... \
        --calib-tasks pubmedqa scifact ...

Temperatures are fitted on the calibration slices exactly as for the encoders, so pre/post-TS metrics
are comparable. Loading follows our Kelvin2 loader (ROCm, bf16, SDPA, HF offline). Credentialed
(PhysioNet) splits are only ever processed locally on the cluster by this script.
"""
import argparse
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import schema
from .calibrate import fit_temperatures
from .common import QTYPES, ROOT, atomic_json, read_json, read_jsonl
from .evaluate import metrics_for, write_preds

LETTERS = [chr(c) for c in range(ord('A'), ord('Z') + 1)] + [chr(c) for c in range(ord('a'), ord('z') + 1)]
STOP = {'flag': False}
PROMPT = ('Read the text and answer the question.\n\nText:\n{state}\n\nQuestion: {ins}\n\nOptions:\n{opts}\n\n'
          'Respond with the letter of the correct option only.')


def load(slug):
    import transformers
    from .models_io import spec_with_path
    spec = spec_with_path(slug)
    path = spec['path']
    cfg = transformers.AutoConfig.from_pretrained(path)
    classes = [getattr(transformers, n) for n in (getattr(cfg, 'architectures', None) or []) if hasattr(transformers, n)]
    for n in ('AutoModelForCausalLM', 'AutoModelForMultimodalLM', 'AutoModelForImageTextToText'):
        if hasattr(transformers, n):
            classes.append(getattr(transformers, n))
    errors, model = [], None
    for cls in classes:
        try:
            model = cls.from_pretrained(path, dtype=torch.bfloat16, attn_implementation='sdpa',
                                        low_cpu_mem_usage=True, device_map={'': 'cuda:0'})
            break
        except (ValueError, KeyError, TypeError) as e:
            errors.append(f'{cls.__name__}: {str(e)[:300]}')
    if model is None:
        raise RuntimeError('No loader worked:\n' + '\n'.join(errors))
    try:
        proc = transformers.AutoProcessor.from_pretrained(path)
    except Exception:
        proc = transformers.AutoTokenizer.from_pretrained(path)
    tok = getattr(proc, 'tokenizer', proc)
    tok.padding_side = 'left'
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model.eval()
    return model, proc, tok, spec


def letter_ids(tok, k):
    ids = []
    for L in LETTERS[:k]:
        cands = {tuple(tok.encode(L, add_special_tokens=False)), tuple(tok.encode(' ' + L, add_special_tokens=False))}
        cands = [c[0] for c in cands if len(c) == 1]
        if not cands:
            raise ValueError(f'letter {L!r} is not a single token')
        ids.append(cands)
    return ids


def build_prompts(proc, tok, records, max_state_tokens):
    items = []
    for r in records:
        state = schema.serialize_state(r['state'])
        ids = tok.encode(state, add_special_tokens=False)
        if len(ids) > max_state_tokens:
            state = tok.decode(ids[:max_state_tokens]) + ' [...]'
        for qid, q in r['questions'].items():
            if qid not in r['gold']:
                continue
            opts = schema.render_options(q)
            text = PROMPT.format(state=state, ins=q['instructions'],
                                 opts='\n'.join(f'{LETTERS[i]}. {o}' for i, o in enumerate(opts)))
            msgs = [{'role': 'user', 'content': [{'type': 'text', 'text': text}]}]
            try:
                prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except Exception:
                prompt = tok.apply_chat_template([{'role': 'user', 'content': text}], tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False)
            items.append(dict(prompt=prompt, K=len(opts), qid=qid, qtype=QTYPES[q['type']], id=r['id'],
                              keys=schema.option_keys(q), target=schema.target(q, r['gold'][qid])))
    return items


def _last_logits(model, enc):
    """Next-token logits of the last position only. Computing the full [batch, seq, vocab] logits for
    long prompts costs tens of GB with 150k-260k vocabularies; logits_to_keep=1 avoids that."""
    try:
        return model(**enc, logits_to_keep=1).logits[:, -1].float()
    except TypeError:                                   # model class without logits_to_keep
        return model(**enc).logits[:, -1].float()


@torch.no_grad()
def score(model, tok, items, token_budget=12000, device='cuda'):
    lens = [len(tok.encode(it['prompt'], add_special_tokens=False)) for it in items]
    order = sorted(range(len(items)), key=lambda i: lens[i])
    out, t0, i = [None] * len(items), time.time(), 0
    lid_cache = {}
    budget = token_budget
    while i < len(order):
        if STOP['flag']:
            return None, None
        L = lens[order[min(len(order) - 1, i)]]
        bs = max(1, min(32, budget // max(1, L)))
        js = order[i:i + bs]
        L = max(lens[j] for j in js)
        bs = max(1, min(len(js), budget // max(1, L)))
        js = js[:bs]
        enc = tok([items[j]['prompt'] for j in js], return_tensors='pt', padding=True, add_special_tokens=False).to(device)
        try:
            logits = _last_logits(model, enc)
        except torch.OutOfMemoryError:
            del enc
            torch.cuda.empty_cache()
            if len(js) == 1:
                raise                                   # a single prompt does not fit the budget
            budget = max(L, budget // 2)                # halve the batch and retry the same items
            print(f'OOM: token budget -> {budget}', flush=True)
            continue
        logp = torch.log_softmax(logits, -1).cpu()
        for r, j in enumerate(js):
            k = items[j]['K']
            if k not in lid_cache:
                lid_cache[k] = letter_ids(tok, k)
            out[j] = [float(torch.logsumexp(logp[r, ids], 0)) for ids in lid_cache[k]]
        i += len(js)
    return out, len(items) / max(1e-9, time.time() - t0)


class _Split:
    """Minimal adapter so evaluate.write_preds / metrics_for work on LLM items."""

    def __init__(self, items):
        self.items = items
        self.d = dict(items=[dict(qid=it['qid'], qtype=it['qtype'], K=it['K'], keys=it['keys'], target=it['target'], rec=n)
                             for n, it in enumerate(items)], ids=[it['id'] for it in items])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--splits', nargs='+', required=True)
    ap.add_argument('--calib-tasks', nargs='*', default=[])
    ap.add_argument('--prepared', default=str(ROOT / 'prepared'))
    ap.add_argument('--max-state-tokens', type=int, default=6000)
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: STOP.update(flag=True))
    run = Path(a.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    from .common import apply_vram_cap
    apply_vram_cap()
    model, proc, tok, spec = load(a.model)
    atomic_json(run / 'config.json', dict(model=a.model, revision=spec.get('revision'), splits=a.splits,
                                          calib=a.calib_tasks, max_state_tokens=a.max_state_tokens))
    tpath = run / 'temps.json'
    if not tpath.exists():
        entries = []
        for t in a.calib_tasks:
            f = Path(a.prepared) / t / 'calib.jsonl'
            if not f.exists():
                continue
            items = build_prompts(proc, tok, read_jsonl(f), a.max_state_tokens)
            z, _ = score(model, tok, items)
            if z is None:
                sys.exit(3)
            entries += [dict(qtype=it['qtype'], logits=zz, target=it['target']) for it, zz in zip(items, z)]
        atomic_json(tpath, fit_temperatures(entries) if entries else {'type': {}, 'bucket': {}})
    temps = read_json(tpath)
    mpath = run / 'metrics_final.json'
    metrics = read_json(mpath) if mpath.exists() else {}
    for name in a.splits:
        if name in metrics:
            continue
        f = Path(a.prepared) / (name + '.jsonl')
        if not f.exists():
            metrics[name] = {'missing': True}
            continue
        items = build_prompts(proc, tok, read_jsonl(f), a.max_state_tokens)
        z, ips = score(model, tok, items)
        if z is None:
            sys.exit(3)
        sp = _Split(items).d
        idxs = list(range(len(items)))
        write_preds(run / 'preds' / 'final' / f"{name.replace('/', '.')}.jsonl.gz", sp, idxs, z)
        metrics[name] = dict(groups=metrics_for(sp, idxs, z, temps), items_per_s=ips)
        atomic_json(mpath, metrics)
        print(name, len(items), f'{ips:.1f} items/s', flush=True)
    (run / 'DONE').write_text('ok\n')


if __name__ == '__main__':
    main()
