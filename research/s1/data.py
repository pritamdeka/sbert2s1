"""Pre-tokenisation (done once per tokenizer family, on CPU) and batch assembly (cheap, per step).

With 12 CPUs shared by up to 6 packed trainers, nothing is tokenised inside the training loop:
states, question headers, option spans and bi-encoder hypotheses are tokenised once and cached;
each step only slices and concatenates integer lists. Option order is chosen at assembly time,
so random option permutation during training costs nothing.
"""
import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import torch

from . import schema
from .common import QTYPES, ROOT, read_json, read_jsonl

OPT_TOKENS = 48          # per-option cap, as in laya.common.build_sequence
HYP_TOKENS = 64          # bi-encoder hypothesis cap


def cache_path(cache_dir, family, task, split):
    return Path(cache_dir) / family / f'{task}.{split}.pkl'


def pretokenize_records(tok, records):
    """Return {'ids','states','items'} with every text piece tokenised without special tokens."""
    mask = tok.mask_token

    def enc(texts, cap=None):
        texts = [t.replace(mask, ' ') for t in texts]
        out = tok(texts, add_special_tokens=False, truncation=cap is not None, max_length=cap)['input_ids']
        return out

    states = enc([schema.serialize_state(r['state']) for r in records])
    items = []
    for ri, r in enumerate(records):
        for qid, q in r['questions'].items():
            if qid not in r['gold']:
                continue
            opts = schema.render_options(q)
            opt_ids = [[tok.mask_token_id] + ids for ids in enc([' ' + o for o in opts], OPT_TOKENS)]
            items.append(dict(
                rec=ri, qid=qid, qtype=QTYPES[q['type']], K=len(opts),
                head=enc([schema.head_text(q)])[0],
                opts=opt_ids,
                hyps=enc(schema.hypotheses(q), HYP_TOKENS),
                target=schema.target(q, r['gold'][qid]),
                keys=schema.option_keys(q)))
    return dict(ids=[r['id'] for r in records], states=states, items=items,
                special=dict(cls=tok.cls_token_id, sep=tok.sep_token_id, pad=tok.pad_token_id,
                             mask=tok.mask_token_id))


def build_cache(tok_path, family, prepared, cache_dir, tasks=None):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_path)
    prepared = Path(prepared)
    for task_dir in sorted(p for p in prepared.iterdir() if p.is_dir()):
        if tasks and task_dir.name not in tasks:
            continue
        for f in sorted(task_dir.glob('*.jsonl')):
            out = cache_path(cache_dir, family, task_dir.name, f.stem)
            if out.exists() and out.stat().st_mtime >= f.stat().st_mtime:
                continue
            data = pretokenize_records(tok, read_jsonl(f))
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix('.tmp')
            with open(tmp, 'wb') as fh:
                pickle.dump(data, fh, protocol=5)
            tmp.replace(out)
            print(f'{family} {task_dir.name}/{f.stem}: {len(data["ids"])} states, {len(data["items"])} decisions', flush=True)


def load_split(cache_dir, family, task, split):
    with open(cache_path(cache_dir, family, task, split), 'rb') as fh:
        return pickle.load(fh)


# ---- batch assembly ----------------------------------------------------------------------

def cross_sequence(it, state, sp, order, max_len=512, head_max_len=192, truncate='right'):
    """laya.common.build_sequence on pre-tokenised parts:
    [CLS] <type> question: instructions [SEP] [MASK] opt_a [MASK] opt_b ... [SEP] state [SEP]"""
    opt_ids = [it['opts'][i] for i in order]
    budget = head_max_len - sum(len(o) for o in opt_ids)
    if budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        budget = head_max_len - sum(len(o) for o in opt_ids)
    ids = [sp['cls']] + it['head'][:max(8, budget)] + [sp['sep']]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(sp['sep'])
    room = max(0, max_len - len(ids) - 1)
    st = state[max(0, len(state) - room):] if truncate == 'left' else state[:room]
    ids = ids + st + [sp['sep']]
    return ids[:max_len], [m for m in markers if m < max_len]


def _pad(rows, pad):
    L = max(len(r) for r in rows)
    ids = torch.full((len(rows), L), pad, dtype=torch.long)
    att = torch.zeros((len(rows), L), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, :len(r)] = torch.tensor(r, dtype=torch.long)
        att[i, :len(r)] = 1
    return ids, att


def collate(split, idxs, orders=None, cross=True, bi=True, max_len=512, head_max_len=192,
            bi_state_len=256, state_override=None, q_prefix=(), d_prefix=()):
    """Assemble one batch. `orders[j]` permutes the options of item j (targets permuted to match).
    `state_override[j]` replaces the state token list (used for long-note windows).
    `q_prefix` / `d_prefix`: token ids some embedding models expect before queries (our hypotheses)
    and documents (our states), e.g. Nomic's "search_query: " / "search_document: ". Bi-encoder only."""
    q_prefix, d_prefix = list(q_prefix), list(d_prefix)
    sp = split['special']
    items = [split['items'][i] for i in idxs]
    orders = orders or [list(range(it['K'])) for it in items]
    kmax = max(it['K'] for it in items)
    n = len(items)
    b = dict(
        qtype=torch.tensor([it['qtype'] for it in items]),
        marker_mask=torch.zeros((n, kmax), dtype=torch.bool),
        target=torch.zeros((n, kmax)),
        order=orders)
    for j, (it, o) in enumerate(zip(items, orders)):
        b['marker_mask'][j, :it['K']] = True
        b['target'][j, :it['K']] = torch.tensor([it['target'][i] for i in o])
    states = [state_override[j] if state_override else split['states'][it['rec']] for j, it in enumerate(items)]
    if cross:
        seqs, mpos = [], torch.zeros((n, kmax), dtype=torch.long)
        for j, (it, o) in enumerate(zip(items, orders)):
            ids, markers = cross_sequence(it, states[j], sp, o, max_len, head_max_len)
            if len(markers) != it['K']:
                raise ValueError(f'item {idxs[j]}: options exceed head budget')
            seqs.append(ids)
            mpos[j, :len(markers)] = torch.tensor(markers)
        b['input_ids'], b['attention_mask'] = _pad(seqs, sp['pad'])
        b['marker_pos'] = mpos
    if bi:
        key_of, s_rows, item_state = {}, [], []
        for j, it in enumerate(items):
            key = ('o', j) if state_override else it['rec']
            if key not in key_of:
                key_of[key] = len(s_rows)
                s_rows.append([sp['cls']] + d_prefix + states[j][:max(1, bi_state_len - 2 - len(d_prefix))] + [sp['sep']])
            item_state.append(key_of[key])
        h_rows, h_index = [], torch.full((n, kmax), -1, dtype=torch.long)
        for j, (it, o) in enumerate(zip(items, orders)):
            for slot, i in enumerate(o):
                h_index[j, slot] = len(h_rows)
                h_rows.append([sp['cls']] + q_prefix + it['hyps'][i] + [sp['sep']])
        b['s_ids'], b['s_att'] = _pad(s_rows, sp['pad'])
        b['h_ids'], b['h_att'] = _pad(h_rows, sp['pad'])
        b['item_state'] = torch.tensor(item_state)
        b['h_index'] = h_index
    return b


def to_device(b, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


# ---- training mixture --------------------------------------------------------------------

def merge(splits):
    """Concatenate several pre-tokenised splits into one (items keep their task name)."""
    specials = {json_key(sp['special']) for sp in splits.values()}
    if len(specials) != 1:
        raise ValueError('splits come from different tokenizer families')
    m = dict(ids=[], states=[], items=[], special=next(iter(splits.values()))['special'])
    for name, sp in splits.items():
        off = len(m['states'])
        m['ids'] += sp['ids']
        m['states'] += sp['states']
        for it in sp['items']:
            m['items'].append(dict(it, rec=it['rec'] + off, task=name))
    return m


def json_key(d):
    return tuple(sorted(d.items()))


class Mixture:
    """Deterministic schedule over a merged training split: task t is drawn with probability
    proportional to n_t ** alpha (alpha=0.5 tempers large tasks), items uniformly within a task.
    `frac` subsamples each task by *state* (all questions of a kept state stay together).
    The schedule is a pure function of (seed, step), so a resumed run continues exactly."""

    def __init__(self, merged, frac=1.0, alpha=0.5, seed=0):
        by_task = {}
        for i, it in enumerate(merged['items']):
            by_task.setdefault(it['task'], []).append(i)
        rng = np.random.default_rng(seed + 7919)
        self.pools = {}
        for name, idx in by_task.items():
            recs = sorted({merged['items'][i]['rec'] for i in idx})
            n_keep = max(1, int(round(len(recs) * frac)))
            keep = set(rng.permutation(recs)[:n_keep].tolist()) if frac < 1 else set(recs)
            self.pools[name] = np.array([i for i in idx if merged['items'][i]['rec'] in keep])
        self.names = sorted(self.pools)
        sizes = np.array([len(self.pools[k]) for k in self.names], dtype=float)
        self.p = sizes ** alpha / (sizes ** alpha).sum()
        self.seed = seed

    def sizes(self):
        return {k: int(len(v)) for k, v in self.pools.items()}

    def total(self):
        return int(sum(len(v) for v in self.pools.values()))

    def batch(self, step, bsz):
        rng = np.random.default_rng([self.seed, step])
        tasks = rng.choice(len(self.names), size=bsz, p=self.p)
        idxs = [int(self.pools[self.names[t]][rng.integers(len(self.pools[self.names[t]]))]) for t in tasks]
        return idxs, rng


def random_orders(split, idxs, rng, permute=True):
    out = []
    for i in idxs:
        k = split['items'][i]['K']
        o = list(range(k))
        # Ordinal scales keep their order: permuting "level 0..4" would destroy the scale.
        if permute and split['items'][i]['qtype'] != QTYPES['score']:
            o = [int(x) for x in rng.permutation(k)]
        out.append(o)
    return out


def main():
    p = argparse.ArgumentParser(description='Pre-tokenise prepared tasks for one tokenizer family')
    p.add_argument('--family', required=True)
    p.add_argument('--prepared', default=str(ROOT / 'prepared'))
    p.add_argument('--cache', default=str(ROOT / 'cache' / 'pretok'))
    p.add_argument('--tasks', nargs='*')
    a = p.parse_args()
    from .models_io import spec_with_path, specs
    import os
    slug = next(s for s, m in specs().items() if m.get('tok_family') == a.family)
    spec = spec_with_path(slug)
    tok_path = os.path.join(spec['path'], spec['tokenizer_subfolder']) if spec.get('tokenizer_subfolder') else spec['path']
    build_cache(tok_path, a.family, a.prepared, a.cache, a.tasks)


if __name__ == '__main__':
    random.seed(0)
    main()
