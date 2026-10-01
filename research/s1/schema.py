"""Typed-decision schema: records, option rendering, targets, and bi-encoder hypotheses.

Record (one per state, one JSONL line), identical in spirit to LocalLLaMA/typed-decisions:
  {"id": str, "task": str, "split": str, "state": str | dict,
   "questions": {qid: {"type": "choice"|"score"|"noul", "instructions": str,
                       "criteria": dict (choice) | list (score) | dict{false,true} (noul, optional),
                       "labels": {"false": str, "true": str} (noul, optional)}},
   "gold": {qid: {"probabilities": {option_key: p}}},
   "meta": {...}}
Option keys: choice -> criteria keys; score -> "0".."K-1"; noul -> "false","true".

Option rendering reproduces laya.common.render_options (Apache-2.0, Convai Innovations) so the
cross (mask-slot) architecture exported by this package reads the same token sequence Laya does.
"""
import json

from .common import QTYPES

NOUL_DEFAULT = {'false': 'no, the statement does not hold', 'true': 'yes, the statement holds'}


def serialize_state(state):
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def render_criterion(v):
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, separators=(', ', ': '), default=str)


def option_keys(q):
    t = q['type']
    if t == 'choice':
        return [str(k) for k in q['criteria']]
    if t == 'score':
        return [str(i) for i in range(len(q['criteria']))]
    return ['false', 'true']


def render_options(q):
    """Option texts in label-index order (laya.common.render_options)."""
    t, crit = q['type'], q.get('criteria')
    if t == 'choice':
        return [str(k) if v is None or v == '' else f'{k}: {render_criterion(v)}' for k, v in crit.items()]
    if t == 'score':
        return [f'level {i}: {render_criterion(c)}' for i, c in enumerate(crit)]
    crit = crit or {}
    labels = q.get('labels') or {'false': 'false', 'true': 'true'}
    out = []
    for side in ('false', 'true'):
        c = crit.get(side)
        out.append(labels[side] + ': ' + (render_criterion(c) if c not in (None, '') else NOUL_DEFAULT[side]))
    return out


def head_text(q):
    """Laya's question header: "<type> question: <instructions>"."""
    return f"{q['type']} question: {q['instructions']}"


def hypotheses(q):
    """Bi-encoder hypothesis per option: the question followed by the rendered option.

    For a retrieval-trained SBERT the hypothesis plays the query and the state plays the
    passage, which is the direction MS-MARCO training optimised.
    """
    return [f"{q['instructions']} {o}" for o in render_options(q)]


def target(q, gold_q):
    keys = option_keys(q)
    probs = (gold_q or {}).get('probabilities', {})
    t = [float(probs.get(k, 0.0)) for k in keys]
    s = sum(t)
    if s <= 0:
        raise ValueError('gold has no mass on any option')
    return [v / s for v in t]


def validate(rec):
    for k in ('id', 'task', 'split', 'state', 'questions', 'gold'):
        if k not in rec:
            raise ValueError(f'record missing {k!r}')
    for qid, q in rec['questions'].items():
        if q['type'] not in QTYPES:
            raise ValueError(f'{rec["id"]}/{qid}: bad type')
        if q['type'] == 'choice' and (not isinstance(q['criteria'], dict) or len(q['criteria']) < 2):
            raise ValueError(f'{rec["id"]}/{qid}: choice needs >=2 criteria')
        if q['type'] == 'score' and (not isinstance(q['criteria'], list) or len(q['criteria']) < 2):
            raise ValueError(f'{rec["id"]}/{qid}: score needs a list of >=2 levels')
        if qid in rec['gold']:
            target(q, rec['gold'][qid])
    return rec


def to_laya_questions(questions):
    """Our question dicts are already the public Laya/Jev request schema."""
    return {qid: {k: v for k, v in q.items() if k in ('type', 'instructions', 'criteria', 'labels')}
            for qid, q in questions.items()}
