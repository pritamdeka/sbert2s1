"""Inference API for an exported sbert2s1 model (the released artifact).

    from s1.predict import load
    m = load('path/or/hub-id')
    m.predict(state, {"urgent": {"type": "noul", "instructions": "Is this urgent?"}})

Returns Laya/Jev-shaped answers: choice -> {"choice", "probabilities", "answer_confidence"},
score -> {"score" (expected level), "probabilities"}, noul -> {"noul": P(true)}.
The request schema is the public Laya/Jev one, so the same client code works against Laya, Jev and this.
"""
import json
import os
from pathlib import Path

import numpy as np
import torch

from . import data as D
from . import schema
from .calibrate import temperature_for
from .common import QTYPES
from .metrics import softmax_np
from .model import S1Model


class Predictor:
    def __init__(self, model, tok, cfg, temps, device):
        self.model, self.tok, self.cfg, self.temps, self.device = model.eval(), tok, cfg, temps, device

    @torch.no_grad()
    def predict(self, state, questions, long=None):
        rec = dict(id='req', task='req', split='req', state=state, questions=questions,
                   gold={q: {'probabilities': {schema.option_keys(v)[0]: 1.0}} for q, v in questions.items()})
        schema.validate(rec)
        sp = D.pretokenize_records(self.tok, [rec])
        idxs = list(range(len(sp['items'])))
        from .evaluate import long_state, predict
        if long and len(sp['states'][0]) > self.cfg['max_len'] - self.cfg['head_max_len']:
            z = long_state(self.model, sp, self.cfg, self.device, long, sp['special'])
        else:
            z = predict(self.model, sp, idxs, self.cfg, self.device)
        out = {}
        for it, zz in zip(sp['items'], z):
            k = it['K']
            p = softmax_np(np.array(zz), temperature_for(self.temps, it['qtype'], k))
            keys = it['keys']
            conf = float(p.max())
            if it['qtype'] == QTYPES['choice']:
                out[it['qid']] = dict(type='choice', choice=keys[int(p.argmax())], answer_confidence=round(conf, 4),
                                      probabilities={kk: round(float(v), 4) for kk, v in zip(keys, p)})
            elif it['qtype'] == QTYPES['score']:
                out[it['qid']] = dict(type='score', score=round(float((np.arange(k) * p).sum()), 4),
                                      answer_confidence=round(conf, 4),
                                      probabilities={str(i): round(float(v), 4) for i, v in enumerate(p)})
            else:
                out[it['qid']] = dict(type='noul', noul=round(float(p[1]), 4), answer_confidence=round(conf, 4))
        return {'answers': out}


def load(path, device=None):
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    p = Path(path) if Path(path).exists() else Path(snapshot_download(path))
    cfg = json.loads((p / 's1_config.json').read_text())
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    enc = AutoModel.from_config(AutoConfig.from_pretrained(p / 'encoder'), attn_implementation='sdpa')
    model = S1Model(enc, cfg['arch'], pooling=cfg['pooling'], prior=cfg.get('prior', 'shared'))
    model.load_state_dict(load_file(str(p / 'model.safetensors')), strict=True)
    tok = AutoTokenizer.from_pretrained(p / 'tokenizer')
    run_cfg = dict(max_len=cfg['max_len'], head_max_len=cfg['head_max_len'], bi_state_len=cfg['bi_state_len'],
                   amp='bf16' if device.type == 'cuda' else 'none')
    return Predictor(model.to(device), tok, run_cfg, cfg['temperatures'], device)


def export(run_dir, out_dir, laya=False):
    """Write a self-contained model directory from a finished run with save_model=true.
    With laya=True (arch C only) also write the Laya layout, loadable by `laya.load(out_dir + '/laya')`."""
    from safetensors.torch import save_file
    from transformers import AutoTokenizer
    from .common import read_json
    from .models_io import spec_with_path
    run, out = Path(run_dir), Path(out_dir)
    cfg = read_json(run / 'config.json')['config']
    spec = spec_with_path(cfg['encoder'])
    sd = torch.load(run / 'model.pt', map_location='cpu')
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in sd.items()}, str(out / 'model.safetensors'))
    enc_dir = os.path.join(spec['path'], 'encoder') if spec.get('laya') else spec['path']
    from transformers import AutoConfig
    AutoConfig.from_pretrained(enc_dir).save_pretrained(out / 'encoder')
    tok_dir = os.path.join(spec['path'], spec['tokenizer_subfolder']) if spec.get('tokenizer_subfolder') else spec['path']
    AutoTokenizer.from_pretrained(tok_dir).save_pretrained(out / 'tokenizer')
    temps = read_json(run / 'temps.json')
    (out / 's1_config.json').write_text(json.dumps(dict(
        arch='C' if spec.get('laya') else cfg['arch'], pooling=spec.get('pooling', 'mean'), prior=cfg.get('prior', 'shared'),
        max_len=cfg['max_len'], head_max_len=cfg['head_max_len'], bi_state_len=cfg['bi_state_len'],
        temperatures=temps, base_model=spec['model_id'], base_revision=spec.get('revision'),
        objective=cfg['objective'], train_tasks=cfg['train_tasks'], seed=cfg['seed']), indent=1))
    if laya:
        if cfg['arch'] != 'C' and not spec.get('laya'):
            raise ValueError('Laya layout exists only for the cross (C) architecture')
        lay = out / 'laya'
        lay.mkdir(exist_ok=True)
        w = {('encoder.' + k[len('encoder.'):]) if k.startswith('encoder.') else k[len('cross.'):]: v.contiguous()
             for k, v in sd.items() if k.startswith(('encoder.', 'cross.'))}
        types = temps.get('type', {})
        w['temperature'] = torch.tensor([min(5.0, max(0.5, types.get(n, 1.0))) for n in ('choice', 'score', 'noul')])
        save_file(w, str(lay / 'model.safetensors'))
        AutoConfig.from_pretrained(enc_dir).save_pretrained(lay / 'encoder')
        AutoTokenizer.from_pretrained(tok_dir).save_pretrained(lay / 'tokenizer')
        (lay / 'rl_agent_config.json').write_text(json.dumps(dict(
            encoder=spec['model_id'], head_layers=2, max_len=cfg['max_len'], head_max_len=cfg['head_max_len'],
            act_costs={'escalate': 0.5}, amp_dtype='bf16', model_name='sbert2s1',
            temperature=w['temperature'].tolist(),
            temperature_by_options={b: min(5.0, max(0.5, t)) for b, t in temps.get('bucket', {}).items()}), indent=1))
    return out
