"""Inference over pre-tokenised splits, robustness probes, long-state strategies and retrieval retention.

Predictions are stored as raw logits (canonical option order) plus targets, so every calibration and
metric can be recomputed offline without re-running a model. No record text is written, only IDs.
"""
import gzip
import json
import math
from pathlib import Path

import numpy as np
import torch

from . import data as D
from .calibrate import temperature_for
from .metrics import group_metrics, js_divergence, softmax_np


def _autocast(device, amp):
    if device.type == 'cuda' and amp == 'bf16':
        return torch.autocast('cuda', dtype=torch.bfloat16)
    return torch.autocast('cpu', enabled=False)


@torch.no_grad()
def predict(model, split, idxs, cfg, device, orders=None, state_override=None, batch=64):
    """Logits [len(idxs)] as lists in *canonical* option order (permutations are undone)."""
    model.eval()
    lens = [len(split['states'][split['items'][i]['rec']]) if state_override is None else len(state_override[j])
            for j, i in enumerate(idxs)]
    order_idx = sorted(range(len(idxs)), key=lambda j: lens[j])
    out = [None] * len(idxs)
    for s in range(0, len(order_idx), batch):
        js = order_idx[s:s + batch]
        b = D.collate(split, [idxs[j] for j in js], orders=[orders[j] for j in js] if orders else None,
                      cross=model.uses_cross(), bi=model.uses_bi(), max_len=cfg['max_len'],
                      head_max_len=cfg['head_max_len'], bi_state_len=cfg['bi_state_len'],
                      state_override=[state_override[j] for j in js] if state_override else None,
                      q_prefix=cfg.get('_q_prefix', ()), d_prefix=cfg.get('_d_prefix', ()))
        b = D.to_device(b, device)
        with _autocast(device, cfg['amp']):
            z, _ = model(b)
        z = z.float().cpu().numpy()
        for r, j in enumerate(js):
            k = split['items'][idxs[j]]['K']
            o = b['order'][r]
            canon = np.empty(k)
            canon[o] = z[r, :k]           # slot r holds option o[r]
            out[j] = canon.tolist()
    return out


def calib_entries(model, splits, cfg, device):
    entries = []
    for name, sp in splits.items():
        idxs = list(range(len(sp['items'])))
        for i, z in zip(idxs, predict(model, sp, idxs, cfg, device)):
            it = sp['items'][i]
            entries.append(dict(qtype=it['qtype'], logits=z, target=it['target']))
    return entries


def metrics_for(split, idxs, logits, temps=None):
    """Metrics per (qid-group). Items are grouped by question id and option count."""
    groups = {}
    for i, z in zip(idxs, logits):
        it = split['items'][i]
        groups.setdefault((it['qid'], it['qtype'], it['K']), []).append((z, it['target']))
    out = {}
    for (qid, qt, k), rows in groups.items():
        Z = np.array([z for z, _ in rows])
        Tg = np.array([t for _, t in rows])
        T = temperature_for(temps, qt, k) if temps else 1.0
        out[f'{qid}|K{k}'] = dict(raw=group_metrics(softmax_np(Z), Tg, qt),
                                   ts=group_metrics(softmax_np(Z, T), Tg, qt) if temps else None,
                                   qtype=qt, T=T)
    return out


def write_preds(path, split, idxs, logits, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with gzip.open(tmp, 'wt', encoding='utf-8') as f:
        for n, (i, z) in enumerate(zip(idxs, logits)):
            it = split['items'][i]
            row = dict(id=split['ids'][it['rec']], qid=it['qid'], qtype=it['qtype'], keys=it['keys'],
                       target=it['target'], logits=[round(v, 5) for v in z])
            if extra:
                row.update({k: v[n] for k, v in extra.items()})
            f.write(json.dumps(row) + '\n')
    tmp.replace(path)


def permutation_probe(model, split, cfg, device, n_perm, seed):
    """Choice items with K>=3: logits under n_perm random option orders (returned canonical)."""
    idxs = [i for i, it in enumerate(split['items']) if it['qtype'] == 0 and it['K'] >= 3]
    if not idxs or n_perm <= 0:
        return None
    rng = np.random.default_rng(seed)
    base = predict(model, split, idxs, cfg, device)
    perms = []
    for _ in range(n_perm):
        orders = [list(rng.permutation(split['items'][i]['K'])) for i in idxs]
        perms.append(predict(model, split, idxs, cfg, device, orders=orders))
    flips, jss = [], []
    for j in range(len(idxs)):
        p0 = softmax_np(np.array(base[j]))
        for pl in perms:
            p1 = softmax_np(np.array(pl[j]))
            flips.append(float(p0.argmax() != p1.argmax()))
            jss.append(js_divergence(p0[None], p1[None]))
    return dict(n=len(idxs), n_perm=n_perm, flip_rate=float(np.mean(flips)), mean_js=float(np.mean(jss)))


# ---- long states ---------------------------------------------------------------------------

def _windows(state, size, stride):
    if len(state) <= size:
        return [state]
    return [state[s:s + size] for s in range(0, max(1, len(state) - size + stride), stride)]


@torch.no_grad()
def long_state(model, split, cfg, device, strategy, tok_special, k_chunks=3, chunk=128):
    """Logits per item for one long-state strategy:
    truncate        first tokens that fit (the default everywhere else)
    window_mean     mean of logits over sliding windows
    window_conf     the window whose prediction is most confident
    rtd             retrieve-then-decide: the bi-encoder picks the k chunks most similar to the question
                    (its instructions) and the model decides on them, in document order"""
    idxs = list(range(len(split['items'])))
    if strategy == 'truncate':
        return predict(model, split, idxs, cfg, device)
    room = cfg['max_len'] - cfg['head_max_len'] - 8
    if strategy in ('window_mean', 'window_conf'):
        out = []
        for i in idxs:
            wins = _windows(split['states'][split['items'][i]['rec']], room, room // 2)
            zs = np.array(predict(model, split, [i] * len(wins), cfg, device, state_override=wins))
            if strategy == 'window_mean':
                out.append(zs.mean(0).tolist())
            else:
                conf = softmax_np(zs).max(-1)
                out.append(zs[int(conf.argmax())].tolist())
        return out
    if strategy == 'rtd':
        sp = tok_special
        chosen = []
        cache = {}
        for i in idxs:
            it = split['items'][i]
            rec = it['rec']
            if rec not in cache:
                chunks = _windows(split['states'][rec], chunk, chunk)
                dp = list(cfg.get('_d_prefix', ()))
                ids, att = D._pad([[sp['cls']] + dp + c + [sp['sep']] for c in chunks], sp['pad'])
                emb = torch.nn.functional.normalize(model.embed(ids.to(device), att.to(device)).float(), dim=-1)
                cache[rec] = (chunks, emb)
            chunks, emb = cache[rec]
            qp = list(cfg.get('_q_prefix', ()))
            q_ids, q_att = D._pad([[sp['cls']] + qp + it['head'][:126] + [sp['sep']]], sp['pad'])
            qv = torch.nn.functional.normalize(model.embed(q_ids.to(device), q_att.to(device)).float(), dim=-1)
            top = sorted(torch.topk(emb @ qv[0], min(k_chunks, len(chunks))).indices.tolist())
            chosen.append([t for c in top for t in chunks[c]])
        return predict(model, split, idxs, cfg, device, state_override=chosen)
    raise ValueError(strategy)


# ---- retrieval retention -------------------------------------------------------------------

@torch.no_grad()
def retrieval_ndcg(model, tok, device, ret_dir, max_len=256, batch=128, k=10, q_prefix='', d_prefix=''):
    ret_dir = Path(ret_dir)
    corpus = [json.loads(l) for l in open(ret_dir / 'corpus.jsonl', encoding='utf-8')]
    queries = {q['_id']: q['text'] for q in map(json.loads, open(ret_dir / 'queries.jsonl', encoding='utf-8'))}
    qrels = {}
    for line in open(ret_dir / 'qrels_test.tsv', encoding='utf-8').read().splitlines()[1:]:
        qid, did, s = line.split('\t')[:3]
        if int(s) > 0:
            qrels.setdefault(qid, {})[did] = int(s)
    qids = [q for q in qrels if q in queries]

    def enc(texts):
        vs = []
        for s in range(0, len(texts), batch):
            t = tok(texts[s:s + batch], padding=True, truncation=True, max_length=max_len, return_tensors='pt')
            vs.append(torch.nn.functional.normalize(model.embed(t['input_ids'].to(device), t['attention_mask'].to(device)).float(), dim=-1))
        return torch.cat(vs)
    model.eval()
    D_ = enc([d_prefix + (d.get('title', '') + ' ' + d['text']).strip() for d in corpus])
    Q = enc([q_prefix + queries[q] for q in qids])
    ids = [d['_id'] for d in corpus]
    scores = []
    for qi, q in enumerate(qids):
        top = torch.topk(D_ @ Q[qi], k).indices.tolist()
        dcg = sum((2 ** qrels[q].get(ids[t], 0) - 1) / math.log2(r + 2) for r, t in enumerate(top))
        ideal = sorted(qrels[q].values(), reverse=True)[:k]
        idcg = sum((2 ** g - 1) / math.log2(r + 2) for r, g in enumerate(ideal))
        scores.append(dcg / idcg if idcg else 0.0)
    return dict(ndcg10=float(np.mean(scores)), n_queries=len(qids), n_docs=len(ids))
