"""Inference for models exported by the sbert2s1 framework.

This module is self-contained (torch, transformers, safetensors, huggingface_hub) and identical in
behaviour to the research code's ``s1.predict``: the request rendering, cross-head sequence layout,
prior fusion and temperature lookup reproduce training exactly.

Requests use the Jev/Laya typed-decision schema:

* choice: ``{"type": "choice", "instructions": str, "criteria": {key: description, ...}}``
* score:  ``{"type": "score", "instructions": str, "criteria": [level_0, level_1, ...]}`` (ordinal, low to high)
* noul:   ``{"type": "noul", "instructions": str, "criteria": {"false": str, "true": str}}`` (criteria optional)

Answers: choice -> choice + probabilities; score -> expected level + probabilities; noul -> P(true).
Probabilities are temperature-scaled with temperatures fitted on held-out calibration data. The model
reads at most ``max_len`` (512) tokens, question block first; longer states are truncated and the
answer carries ``truncated: true``.
"""
import copy
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

QTYPES = {'choice': 0, 'score': 1, 'noul': 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}
NOUL_DEFAULT = {'false': 'no, the statement does not hold', 'true': 'yes, the statement holds'}
OPT_TOKENS = 48       # per-option token cap
HYP_TOKENS = 64       # bi-encoder hypothesis cap (PFR models only)


# ---- request schema (identical rendering to training) -------------------------------------------

def _serialize_state(state):
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def _render_criterion(v):
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, separators=(', ', ': '), default=str)


def option_keys(q):
    if q['type'] == 'choice':
        return [str(k) for k in q['criteria']]
    if q['type'] == 'score':
        return [str(i) for i in range(len(q['criteria']))]
    return ['false', 'true']


def render_options(q):
    t, crit = q['type'], q.get('criteria')
    if t == 'choice':
        return [str(k) if v is None or v == '' else f'{k}: {_render_criterion(v)}' for k, v in crit.items()]
    if t == 'score':
        return [f'level {i}: {_render_criterion(c)}' for i, c in enumerate(crit)]
    crit = crit or {}
    labels = q.get('labels') or {'false': 'false', 'true': 'true'}
    return [labels[s] + ': ' + (_render_criterion(crit.get(s)) if crit.get(s) not in (None, '') else NOUL_DEFAULT[s])
            for s in ('false', 'true')]


def _validate(qid, q):
    if q.get('type') not in QTYPES:
        raise ValueError(f'{qid}: type must be one of {list(QTYPES)}')
    if 'instructions' not in q:
        raise ValueError(f'{qid}: missing "instructions"')
    if q['type'] == 'choice' and (not isinstance(q.get('criteria'), dict) or len(q['criteria']) < 2):
        raise ValueError(f'{qid}: choice needs a dict of >= 2 criteria')
    if q['type'] == 'score' and (not isinstance(q.get('criteria'), list) or len(q['criteria']) < 2):
        raise ValueError(f'{qid}: score needs a list of >= 2 levels')


# ---- model (parameter names match the exported checkpoint) ---------------------------------------

def _pool(h, mask, mode):
    if mode == 'cls':
        return h[:, 0]
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp_min(1.0)


class CrossHead(nn.Module):
    def __init__(self, d, head_layers=2, n_act=2, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d, max(1, d // 64), 4 * d, dropout, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False)
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))   # unused at inference

    def forward(self, h, attention_mask, marker_pos, qtype):
        h = h + self.type_emb(qtype)[:, None, :]
        pad = ~attention_mask.bool()
        for layer in self.head.layers:
            h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        return self.scorer(torch.gather(h, 1, idx)).squeeze(-1).float()


class S1Model(nn.Module):
    def __init__(self, encoder, arch, pooling='mean', frozen_prior=False):
        super().__init__()
        self.arch, self.pooling = arch, pooling
        self.encoder = encoder
        d = encoder.config.hidden_size
        self.alpha = nn.Parameter(torch.ones(3) * 20.0)
        self.register_buffer('temperature', torch.ones(3))
        if arch in ('C', 'PFR'):
            self.cross = CrossHead(d)
        if frozen_prior:
            self.bi_encoder = copy.deepcopy(encoder)

    def uses_bi(self):
        return self.arch in ('Z', 'B', 'PFR')

    def uses_cross(self):
        return self.arch in ('C', 'PFR')

    def _embed(self, ids, att):
        enc = self.bi_encoder if hasattr(self, 'bi_encoder') else self.encoder
        return _pool(enc(input_ids=ids, attention_mask=att).last_hidden_state, att, self.pooling)

    def forward(self, b):
        z = 0.0
        if self.uses_cross():
            h = self.encoder(input_ids=b['input_ids'], attention_mask=b['attention_mask']).last_hidden_state
            z = z + self.cross(h, b['attention_mask'], b['marker_pos'], b['qtype'])
        if self.uses_bi():
            s = F.normalize(self._embed(b['s_ids'], b['s_att']).float(), dim=-1)
            v = F.normalize(self._embed(b['h_ids'], b['h_att']).float(), dim=-1)
            cos = (s[b['item_state']].unsqueeze(1) * v[b['h_index'].clamp(min=0)]).sum(-1)
            z = z + self.alpha[b['qtype']].unsqueeze(1) * cos.masked_fill(~b['marker_mask'], 0.0)
        return z.masked_fill(~b['marker_mask'], -1e4)


# ---- predictor -------------------------------------------------------------------------------------

def _temperature(temps, qtype, k):
    name = QTYPE_NAMES[int(qtype)]
    size = '2' if k <= 2 else '3-5' if k <= 5 else '6-10' if k <= 10 else '11+'
    b = f'{name}:{size}'
    return temps.get('bucket', {}).get(b, temps.get('type', {}).get(name, 1.0))


class Predictor:
    def __init__(self, model, tok, cfg, device):
        self.model, self.tok, self.cfg, self.device = model.eval(), tok, cfg, device
        self.temps = cfg['temperatures']

    def _enc(self, texts, cap=None):
        texts = [t.replace(self.tok.mask_token, ' ') for t in texts]
        return self.tok(texts, add_special_tokens=False, truncation=cap is not None, max_length=cap)['input_ids']

    def _items(self, state, questions):
        st = self._enc([_serialize_state(state)])[0]
        items = []
        for qid, q in questions.items():
            _validate(qid, q)
            opts = render_options(q)
            items.append(dict(qid=qid, qtype=QTYPES[q['type']], K=len(opts), keys=option_keys(q),
                              head=self._enc([f"{q['type']} question: {q['instructions']}"])[0],
                              opts=[[self.tok.mask_token_id] + o for o in self._enc([' ' + o for o in opts], OPT_TOKENS)],
                              hyps=self._enc([f"{q['instructions']} {o}" for o in opts], HYP_TOKENS)))
        return st, items

    def _cross_sequence(self, it, state):
        max_len, head_max = self.cfg['max_len'], self.cfg['head_max_len']
        cls, sep = self.tok.cls_token_id, self.tok.sep_token_id
        opt_ids = it['opts']
        budget = head_max - sum(len(o) for o in opt_ids)
        if budget < 16:
            per = max(4, (head_max - 16) // max(1, len(opt_ids)))
            opt_ids = [o[:per] for o in opt_ids]
            budget = head_max - sum(len(o) for o in opt_ids)
        ids = [cls] + it['head'][:max(8, budget)] + [sep]
        markers = []
        for o in opt_ids:
            markers.append(len(ids))
            ids.extend(o)
        ids.append(sep)
        room = max(0, max_len - len(ids) - 1)
        truncated = len(state) > room
        ids = ids + state[:room] + [sep]
        markers = [m for m in markers if m < max_len]
        if len(markers) != it['K']:
            raise ValueError(f"{it['qid']}: too many/long options for the {head_max}-token question budget")
        return ids[:max_len], markers, truncated

    @staticmethod
    def _pad(rows, pad):
        L = max(len(r) for r in rows)
        ids = torch.full((len(rows), L), pad, dtype=torch.long)
        att = torch.zeros((len(rows), L), dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, :len(r)] = torch.tensor(r)
            att[i, :len(r)] = 1
        return ids, att

    def _batch(self, entries):
        """entries: list of (state_ids, item)."""
        n, kmax = len(entries), max(it['K'] for _, it in entries)
        b = dict(qtype=torch.tensor([it['qtype'] for _, it in entries]),
                 marker_mask=torch.zeros((n, kmax), dtype=torch.bool))
        for j, (_, it) in enumerate(entries):
            b['marker_mask'][j, :it['K']] = True
        trunc = [False] * n
        pad, cls, sep = self.tok.pad_token_id, self.tok.cls_token_id, self.tok.sep_token_id
        if self.model.uses_cross():
            seqs, mpos = [], torch.zeros((n, kmax), dtype=torch.long)
            for j, (st, it) in enumerate(entries):
                ids, markers, trunc[j] = self._cross_sequence(it, st)
                seqs.append(ids)
                mpos[j, :len(markers)] = torch.tensor(markers)
            b['input_ids'], b['attention_mask'] = self._pad(seqs, pad)
            b['marker_pos'] = mpos
        if self.model.uses_bi():
            s_rows, item_state, h_rows = [], [], []
            h_index = torch.full((n, kmax), -1, dtype=torch.long)
            seen = {}
            for j, (st, it) in enumerate(entries):
                key = id(st)
                if key not in seen:
                    seen[key] = len(s_rows)
                    s_rows.append([cls] + st[:max(1, self.cfg['bi_state_len'] - 2)] + [sep])
                item_state.append(seen[key])
                for slot in range(it['K']):
                    h_index[j, slot] = len(h_rows)
                    h_rows.append([cls] + it['hyps'][slot] + [sep])
            b['s_ids'], b['s_att'] = self._pad(s_rows, pad)
            b['h_ids'], b['h_att'] = self._pad(h_rows, pad)
            b['item_state'], b['h_index'] = torch.tensor(item_state), h_index
        return {k: v.to(self.device) for k, v in b.items()}, trunc

    @torch.no_grad()
    def logits(self, requests, batch_size=32):
        """requests: list of (state, questions). Returns raw logits per request and question."""
        entries, owner = [], []
        for r, (state, questions) in enumerate(requests):
            st, items = self._items(state, questions)
            for it in items:
                entries.append((st, it))
                owner.append(r)
        out = [dict() for _ in requests]
        amp = self.device.type == 'cuda'
        for s in range(0, len(entries), batch_size):
            chunk = entries[s:s + batch_size]
            b, trunc = self._batch(chunk)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                z = self.model(b).float().cpu()
            for j, (st, it) in enumerate(chunk):
                out[owner[s + j]][it['qid']] = (z[j, :it['K']], it, trunc[j])
        return out

    def predict_batch(self, requests, batch_size=32):
        """``requests``: list of ``(state, questions)`` pairs. Returns one result dict per request."""
        results = []
        for per in self.logits(requests, batch_size):
            ans = {}
            for qid, (z, it, truncated) in per.items():
                p = torch.softmax(z / _temperature(self.temps, it['qtype'], it['K']), -1)
                conf = float(p.max())
                if it['qtype'] == QTYPES['choice']:
                    a = dict(type='choice', choice=it['keys'][int(p.argmax())], answer_confidence=round(conf, 4),
                             probabilities={k: round(float(v), 4) for k, v in zip(it['keys'], p)})
                elif it['qtype'] == QTYPES['score']:
                    a = dict(type='score', score=round(float((torch.arange(it['K']) * p).sum()), 4),
                             answer_confidence=round(conf, 4),
                             probabilities={str(i): round(float(v), 4) for i, v in enumerate(p)})
                else:
                    a = dict(type='noul', noul=round(float(p[1]), 4), answer_confidence=round(conf, 4))
                if truncated:
                    a['truncated'] = True
                ans[qid] = a
            results.append({'answers': ans})
        return results

    def predict(self, state, questions):
        """Answer every question in ``questions`` about one ``state`` (a string or JSON-serialisable
        object). Returns ``{"answers": {question_id: answer, ...}}``."""
        return self.predict_batch([(state, questions)])[0]


def load(path_or_repo, device=None, revision=None):
    """Load an exported sbert2s1 model from a local directory or a Hugging Face Hub repository.

    Args:
        path_or_repo: local directory with ``s1_config.json``, ``model.safetensors``, ``encoder/`` and
            ``tokenizer/``, or a Hub id such as ``"pritamdeka/S1-PubMedBERT"``.
        device: ``"cuda"``, ``"cpu"`` or a ``torch.device``; defaults to CUDA when available.
        revision: optional Hub revision (branch, tag or commit) for reproducible downloads.

    Returns:
        A :class:`Predictor`.
    """
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    p = Path(path_or_repo)
    if not p.exists():
        p = Path(snapshot_download(str(path_or_repo), revision=revision,
                                   allow_patterns=['*.json', '*.safetensors', 'tokenizer/*', 'encoder/*', '*.txt']))
    cfg = json.loads((p / 's1_config.json').read_text())
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    enc_cfg = AutoConfig.from_pretrained(p / 'encoder')
    try:
        enc = AutoModel.from_config(enc_cfg, attn_implementation='sdpa')
    except (ValueError, ImportError, TypeError):
        enc = AutoModel.from_config(enc_cfg)
    sd = load_file(str(p / 'model.safetensors'))
    model = S1Model(enc, cfg['arch'], pooling=cfg.get('pooling', 'mean'),
                    frozen_prior=any(k.startswith('bi_encoder.') for k in sd))
    model.load_state_dict(sd, strict=True)
    tok = AutoTokenizer.from_pretrained(p / 'tokenizer')
    return Predictor(model.to(device), tok, cfg, device)
