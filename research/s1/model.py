"""Decision models built from one sentence-transformer (or MLM) encoder.

Architectures:
  Z    zero-shot bi-encoder: z_i = a_t * cos(pool E(state), pool E(hypothesis_i)); only a_t (= 1/tau) is fitted
  B    trained bi-encoder with the same form; stays a usable embedding model
  C    Laya-compatible cross mask-slot head (module names match laya.common.DecisionModel, so a Laya
       checkpoint loads with strict=True and a C model exports to the Laya format)
  PFR  prior-fused residual: z_i = c_i + a_t * cos_i, scorer's last Linear zero-initialised, so the
       untrained model equals the calibrated zero-shot bi-encoder

The cross head reproduces laya.common.DecisionModel (Apache-2.0, Convai Innovations).
"""
import copy
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ARCHS = ('Z', 'B', 'C', 'PFR')


def load_encoder(path, revision=None):
    from transformers import AutoConfig, AutoModel
    kw = dict(revision=revision) if revision else {}
    cfg = AutoConfig.from_pretrained(path, **kw)
    if hasattr(cfg, 'reference_compile'):
        cfg.reference_compile = False                     # ModernBERT: skip torch.compile for portability
    for attn in ('sdpa', 'eager'):
        try:
            # Always fp32 master weights: transformers 5 otherwise keeps the checkpoint's stored dtype, and
            # an fp16 checkpoint (gte-modernbert-base) then trains in fp16 and overflows to NaN.
            # Mixed precision comes from bf16 autocast in the trainer.
            return AutoModel.from_pretrained(path, config=cfg, attn_implementation=attn, dtype=torch.float32, **kw)
        except (ValueError, ImportError, TypeError) as e:     # architecture without SDPA support
            err = e
    raise err


def pool(h, mask, mode):
    if mode == 'cls':
        return h[:, 0]
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp_min(1.0)


class CrossHead(nn.Module):
    """laya.common.DecisionModel without the encoder-owning wrapper; parameter names kept."""

    def __init__(self, d, head_layers=2, n_act=2, dropout=0.1):
        super().__init__()
        nhead = max(1, d // 64)
        layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False) if head_layers > 0 else None
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))

    def forward(self, h, attention_mask, marker_pos, marker_mask, qtype):
        h = h + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            pad = ~attention_mask.bool()
            for layer in self.head.layers:
                h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        return self.scorer(m).squeeze(-1).float()


class S1Model(nn.Module):
    def __init__(self, encoder, arch, pooling='mean', prior='shared', head_layers=2, n_act=2):
        super().__init__()
        if arch not in ARCHS:
            raise ValueError(f'arch must be one of {ARCHS}')
        self.arch, self.pooling, self.prior = arch, pooling, prior
        self.encoder = encoder
        d = encoder.config.hidden_size
        # a_t per question type (choice, score, noul); set from the zero-shot temperature before training
        self.alpha = nn.Parameter(torch.ones(3) * 20.0)
        self.register_buffer('temperature', torch.ones(3))       # Laya checkpoint compatibility
        if arch in ('C', 'PFR'):
            self.cross = CrossHead(d, head_layers, n_act)
        if arch == 'PFR':
            nn.init.zeros_(self.cross.scorer[-1].weight)
            nn.init.zeros_(self.cross.scorer[-1].bias)
            if prior == 'frozen':
                self.bi_encoder = copy.deepcopy(encoder)
                for p in self.bi_encoder.parameters():
                    p.requires_grad_(False)
        if arch == 'Z':
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    # ---- pieces -------------------------------------------------------------------------
    def uses_bi(self):
        return self.arch in ('Z', 'B', 'PFR')

    def uses_cross(self):
        return self.arch in ('C', 'PFR')

    def _bi_enc(self):
        return self.bi_encoder if hasattr(self, 'bi_encoder') else self.encoder

    def embed(self, input_ids, attention_mask, chunk=256):
        enc = self._bi_enc()
        outs = []
        for i in range(0, input_ids.size(0), chunk):
            h = enc(input_ids=input_ids[i:i + chunk], attention_mask=attention_mask[i:i + chunk]).last_hidden_state
            outs.append(pool(h, attention_mask[i:i + chunk], self.pooling))
        return torch.cat(outs)

    def bi_cos(self, b):
        """cos(state_i, hypothesis_ij) laid out as [n_items, Kmax]."""
        s = F.normalize(self.embed(b['s_ids'], b['s_att']).float(), dim=-1)
        v = F.normalize(self.embed(b['h_ids'], b['h_att']).float(), dim=-1)
        cos = (s[b['item_state']].unsqueeze(1) * v[b['h_index'].clamp(min=0)]).sum(-1)
        return cos.masked_fill(~b['marker_mask'], 0.0)

    def cross_logits(self, b):
        h = self.encoder(input_ids=b['input_ids'], attention_mask=b['attention_mask']).last_hidden_state
        return self.cross(h, b['attention_mask'], b['marker_pos'], b['marker_mask'], b['qtype'])

    def forward(self, b):
        """Raw logits [n_items, Kmax]; padded option slots are -1e4. Returns (logits, parts)."""
        parts = {}
        z = 0.0
        if self.uses_cross():
            parts['cross'] = self.cross_logits(b)
            z = z + parts['cross']
        if self.uses_bi():
            parts['cos'] = self.bi_cos(b)
            z = z + self.alpha[b['qtype']].unsqueeze(1) * parts['cos']
        z = z.masked_fill(~b['marker_mask'], -1e4)
        return z, parts

    # ---- persistence --------------------------------------------------------------------
    def trainable_groups(self, lr_enc, lr_head, lr_alpha=None):
        """Encoder, head, and the prior scale alpha. alpha is a scale of order 10-100, so it gets its
        own (larger) learning rate and no weight decay; at the head's 1e-4 it could not move."""
        enc, head, alpha = [], [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if n == 'alpha':
                (alpha if self.uses_bi() else head).append(p)
            else:
                (enc if n.startswith('encoder.') else head).append(p)
        groups = []
        if enc:
            groups.append({'params': enc, 'lr': lr_enc})
        if head:
            groups.append({'params': head, 'lr': lr_head})
        if alpha:
            groups.append({'params': alpha, 'lr': lr_alpha or lr_head, 'weight_decay': 0.0})
        return groups


def build(model_spec, arch, prior='shared', head_layers=2):
    enc = load_encoder(model_spec['path'], model_spec.get('revision'))
    return S1Model(enc, arch, pooling=model_spec.get('pooling', 'mean'), prior=prior, head_layers=head_layers)


def read_safetensors(path):
    """Read a safetensors file from bytes (the mmap path segfaults on some Windows builds)."""
    from safetensors.torch import load
    with open(path, 'rb') as f:
        return load(f.read())


def build_laya(laya_dir):
    """Laya checkpoint -> arch-C S1Model. The ModernBERT encoder is built on the meta device (no random
    initialisation of 395M parameters) and the checkpoint tensors are assigned into it."""
    from transformers import AutoConfig, AutoModel
    ecfg = AutoConfig.from_pretrained(str(Path(laya_dir) / 'encoder'))
    if hasattr(ecfg, 'reference_compile'):
        ecfg.reference_compile = False                    # skip torch.compile for portability
    with torch.device('meta'):
        enc = AutoModel.from_config(ecfg, attn_implementation='sdpa')
        model = S1Model(enc, 'C', pooling='mean')
    w = {k: v.float() for k, v in read_safetensors(Path(laya_dir) / 'model.safetensors').items()}
    sd = {k: v for k, v in w.items() if k.split('.')[0] in ('encoder', 'head', 'type_emb', 'scorer', 'act_head')}
    sd = {('cross.' + k if not k.startswith('encoder.') else k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    real_missing = [m for m in missing if m not in ('alpha', 'temperature')]
    if real_missing or unexpected:
        raise ValueError(f'Laya checkpoint mismatch: missing={real_missing[:5]} unexpected={unexpected[:5]}')
    model.alpha = nn.Parameter(torch.ones(3) * 20.0)
    model.temperature = torch.ones(3)
    # Non-persistent buffers (RoPE inverse frequencies) are not in the checkpoint. Re-create each module
    # that still holds one from its own config, so transformers computes them exactly as it normally would.
    for name, mod in list(model.encoder.named_modules()):
        if any(b.is_meta for b in mod.buffers(recurse=False)) and hasattr(mod, 'config'):
            parent = model.encoder.get_submodule(name.rsplit('.', 1)[0]) if '.' in name else model.encoder
            setattr(parent, name.rsplit('.', 1)[-1], type(mod)(config=mod.config))
    leftover = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if leftover:
        raise ValueError(f'tensors still on meta after loading Laya: {leftover[:5]}')
    return model, json.loads((Path(laya_dir) / 'rl_agent_config.json').read_text())
