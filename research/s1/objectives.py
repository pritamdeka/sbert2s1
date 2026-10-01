"""Training objectives for the RLCD decomposition.

  ce            soft cross-entropy (the log score as a loss)
  proper        -(log + w_sph*spherical) + w_rps*RPS[ordinal], differentiable (Laya's reward as a loss)
  rlcd_pg       Laya's estimator: REINFORCE over G Gaussian-noised logit samples, group-normalised
                advantage, reward = proper score; plus w_ce * ce (Laya uses w_ce = 1). Kept unchanged for
                reproducibility. NOTE: the same-sample mean baseline scales the expected gradient by (G-1)/G
                and the batch-wide std normalisation rescales it again, so this is NOT an unbiased
                estimator of the smoothed-score gradient.
  rlcd_pg_loo   REINFORCE with a leave-one-out baseline and no std normalisation. Unbiased for the
                gradient of the noise-smoothed score, on the same scale as rlcd_reparam (matched control).
  rlcd_reparam  the same noise-smoothed proper score, differentiated pathwise (reparameterisation)
                instead of by REINFORCE; plus w_ce * ce

proper_reward reproduces laya.common.proper_reward (Apache-2.0, Convai Innovations).
"""
import torch
import torch.nn.functional as F

from .common import QTYPES

OBJECTIVES = ('ce', 'proper', 'rlcd_pg', 'rlcd_pg_loo', 'rlcd_reparam')


def proper_reward(q, target, qtype, mask, w_sph=0.75, w_rps=1.0, log_floor=-9.21):
    """Composite proper score (higher is better); the log floor removes strict propriety below exp(log_floor).
    q: [..., N, K]; target: [N, K]; mask: [N, K]."""
    mask = mask.to(q.dtype)
    q = q * mask
    logq = torch.log(q.clamp_min(1e-12)).clamp_min(log_floor)
    r = (target * logq).sum(-1)
    r = r + w_sph * (target * q).sum(-1) / q.norm(dim=-1).clamp_min(1e-9)
    is_score = (qtype == QTYPES['score']).to(q.dtype)
    if is_score.any():
        k = mask.sum(-1).clamp(min=2)
        rps = (((torch.cumsum(q, -1) - torch.cumsum(target, -1)) ** 2) * mask).sum(-1) / (k - 1)
        r = r - w_rps * rps * is_score
    return r


def soft_ce(logits, target, mask):
    return -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()


def _noise(shape, mask, sigma, device):
    eps = torch.randn(shape, device=device) * sigma * mask
    k = mask.sum(-1, keepdim=True).clamp(min=1)
    return (eps - eps.sum(-1, keepdim=True) / k) * mask          # zero-mean over active options


def loss(name, logits, target, mask, qtype, sigma=0.25, G=4, w_ce=1.0, w_sph=0.75, w_rps=1.0, w_score=1.0):
    """Returns (loss, stats). logits: float32 [N, K] with padded slots already -1e4.
    w_score multiplies the noise-smoothed score term (default 1.0 = historical behaviour)."""
    maskf = mask.float()
    ce = soft_ce(logits, target, mask)
    stats = {'ce': float(ce.detach())}
    if name == 'ce':
        return ce, stats
    if name == 'proper':
        r = proper_reward(torch.softmax(logits, -1), target, qtype, mask, w_sph, w_rps)
        stats['reward'] = float(r.mean().detach())
        return -r.mean(), stats
    eps = _noise((G,) + tuple(logits.shape), maskf, sigma, logits.device)
    if name in ('rlcd_pg', 'rlcd_pg_loo'):
        z = logits.detach().unsqueeze(0) + eps
        with torch.no_grad():
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            r = proper_reward(q, target.unsqueeze(0), qtype, mask, w_sph, w_rps)
            if name == 'rlcd_pg_loo':
                if G < 2:
                    raise ValueError('rlcd_pg_loo needs G >= 2 (leave-one-out baseline)')
                adv = r - (r.sum(0, keepdim=True) - r) / (G - 1)     # baseline independent of sample g
            else:                                                     # historical released recipe
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)
        logp = -(((z - logits.unsqueeze(0)) ** 2) * maskf).sum(-1) / (2 * sigma ** 2)
        l_rl = -(adv * logp).mean()
        stats['reward'] = float(r.mean())
    elif name == 'rlcd_reparam':
        z = logits.unsqueeze(0) + eps
        q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
        r = proper_reward(q, target.unsqueeze(0), qtype, mask, w_sph, w_rps)
        l_rl = -r.mean()
        stats['reward'] = float(r.mean().detach())
    else:
        raise ValueError(f'unknown objective {name!r}; choose from {OBJECTIVES}')
    return w_score * l_rl + w_ce * ce, stats


def grad_variance_probe(name, logits, target, mask, qtype, sigma, G=4, draws=8, **kw):
    """Variance of the loss gradient w.r.t. the logits across independent noise draws.
    Cheap (no backprop through the encoder); used for H3c (PG vs reparameterised estimator)."""
    if name not in ('rlcd_pg', 'rlcd_pg_loo', 'rlcd_reparam'):
        return None
    base = logits.detach()
    grads = []
    for _ in range(draws):
        z = base.clone().requires_grad_(True)
        l, _ = loss(name, z, target, mask, qtype, sigma=sigma, G=G, w_ce=0.0, **kw)
        g, = torch.autograd.grad(l, z)
        grads.append(g * mask)
    g = torch.stack(grads)
    return float(g.var(0, unbiased=True).sum(-1).mean())


def sigma_at(progress, start=0.4, end=0.1):
    return start + (end - start) * min(1.0, max(0.0, progress))
