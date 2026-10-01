"""Decision and calibration metrics (numpy only; no sklearn/scipy needed).

Input for one (task, question type) group: probs [N, K] (rows sum to 1 over the K active
options), target [N, K] (one-hot or soft gold), qtype. Hard label = argmax(target).
"""
import numpy as np


def _rank(x):
    order = np.argsort(x, kind='mergesort')
    r = np.empty(len(x))
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return r


def auroc(y, s):
    y = np.asarray(y, bool)
    npos, nneg = y.sum(), (~y).sum()
    if npos == 0 or nneg == 0:
        return float('nan')
    r = _rank(np.asarray(s, float))
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def auprc(y, s):
    y = np.asarray(y, bool)
    if y.sum() == 0:
        return float('nan')
    o = np.argsort(-np.asarray(s, float), kind='mergesort')
    tp = np.cumsum(y[o])
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y[o]).sum() / y.sum())


def spearman(a, b):
    a, b = _rank(np.asarray(a, float)), _rank(np.asarray(b, float))
    if a.std() == 0 or b.std() == 0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def ece(conf, correct, bins=15, mass=False):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    if len(conf) == 0:
        return float('nan')
    if mass:
        edges = np.quantile(conf, np.linspace(0, 1, bins + 1))
        edges[0], edges[-1] = -1e-9, 1 + 1e-9
    else:
        edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        sel = ((conf >= lo) if i == 0 else (conf > lo)) & (conf <= hi)
        if sel.any():
            e += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(e)


def classwise_ece(probs, labels, bins=15):
    K = probs.shape[1]
    return float(np.mean([ece(probs[:, k], labels == k, bins) for k in range(K)]))


def aurc(conf, correct):
    """Area under the risk-coverage curve (lower is better) and its excess over the oracle."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    n = len(conf)
    if n == 0:
        return float('nan'), float('nan')
    o = np.argsort(-conf, kind='mergesort')
    risk = np.cumsum(1 - correct[o]) / np.arange(1, n + 1)
    a = float(risk.mean())
    err = 1 - correct.mean()
    oracle = float(np.mean([max(0.0, (k - (n - n * err)) / k) for k in range(1, n + 1)])) if err > 0 else 0.0
    return a, a - oracle


def coverage_at_precision(conf, correct, target=0.95):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    o = np.argsort(-conf, kind='mergesort')
    prec = np.cumsum(correct[o]) / np.arange(1, len(o) + 1)
    ok = np.where(prec >= target)[0]
    return float((ok.max() + 1) / len(o)) if len(ok) else 0.0


def qwk(y, p, K):
    y, p = np.asarray(y, int), np.asarray(p, int)
    O = np.zeros((K, K))
    for a, b in zip(y, p):
        O[a, b] += 1
    W = np.array([[(i - j) ** 2 / (K - 1) ** 2 for j in range(K)] for i in range(K)])
    E = np.outer(O.sum(1), O.sum(0)) / max(1, O.sum())
    den = (W * E).sum()
    return float(1 - (W * O).sum() / den) if den > 0 else float('nan')


def macro_f1(y, p, K):
    f = []
    for k in range(K):
        tp = np.sum((p == k) & (y == k))
        fp = np.sum((p == k) & (y != k))
        fn = np.sum((p != k) & (y == k))
        if tp + fp + fn == 0:
            continue
        f.append(2 * tp / max(1, 2 * tp + fp + fn))
    return float(np.mean(f)) if f else float('nan')


def group_metrics(probs, target, qtype):
    """All metrics for one homogeneous group (same task, question type and option count)."""
    probs = np.clip(np.asarray(probs, float), 1e-12, 1.0)
    probs = probs / probs.sum(1, keepdims=True)
    target = np.asarray(target, float)
    N, K = probs.shape
    y = target.argmax(1)
    pred = probs.argmax(1)
    conf = probs.max(1)
    correct = (pred == y).astype(float)
    onehot = np.eye(K)[y]
    m = dict(n=int(N), K=int(K),
             acc=float(correct.mean()),
             acc_norm=float((correct.mean() - 1 / K) / (1 - 1 / K)),
             macro_f1=macro_f1(y, pred, K),
             nll=float(-np.log(probs[np.arange(N), y]).mean()),
             brier=float(((probs - onehot) ** 2).sum(1).mean()),
             ece=ece(conf, correct), ece_mass=ece(conf, correct, mass=True),
             cw_ece=classwise_ece(probs, y),
             soft_ce=float(-(target * np.log(probs)).sum(1).mean()),
             cov_at_95=coverage_at_precision(conf, correct, 0.95),
             mean_conf=float(conf.mean()))
    m['aurc'], m['eaurc'] = aurc(conf, correct)
    if qtype == 2 or K == 2:
        m['auroc'] = auroc(y == 1, probs[:, 1])
        m['auprc'] = auprc(y == 1, probs[:, 1])
    if qtype == 1:
        levels = np.arange(K)
        exp = (probs * levels).sum(1)
        m.update(mae=float(np.abs(exp - y).mean()),
                 within1=float((np.abs(pred - y) <= 1).mean()),
                 qwk=qwk(y, pred, K),
                 spearman=spearman(exp, y),
                 rps=float((((np.cumsum(probs, 1) - np.cumsum(onehot, 1)) ** 2).sum(1) / (K - 1)).mean()))
    return m


def softmax_np(z, T=1.0):
    z = np.asarray(z, float) / T
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def js_divergence(p, q):
    p, q = np.clip(p, 1e-12, 1), np.clip(q, 1e-12, 1)
    m = (p + q) / 2
    return float(0.5 * (p * np.log(p / m)).sum(-1).mean() + 0.5 * (q * np.log(q / m)).sum(-1).mean())
