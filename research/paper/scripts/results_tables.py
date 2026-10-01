"""Result tables for the paper, computed only from metrics_long.csv (+ train logs for gradient variance).

Aggregation: per run, average the chance-normalised accuracy over the question groups of each task,
then over the tasks of a track; report mean +- sd over seeds.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TRACKS = ['seen', 'heldout', 'clinical', 'knowledge']
# Options that differ in content (entities, categories, answer texts, rubric wording) vs. label words only.
CONTENT = {('medline_s1', 'design'), ('medline_s1', 'disease_area'), ('mtsamples', '*'), ('medqa', '*'),
           ('medmcqa', '*'), ('mmlu_med', '*'), ('druglib', '*'), ('biosses', '*')}
LABEL_ONLY_TASKS = {'pubmedqa', 'scifact', 'healthver', 'pubhealth', 'ddi', 'hoc', 'ade', 'mednli', 'mimic_deid'}
ENC = [('pubmedbert', 'PubMedBERT', 'MLM'), ('spubmedbert', 'S-PubMedBERT-MS-MARCO', 'contr.'),
       ('medcpt_q', 'MedCPT-Query', 'contr.'), ('mpnet', 'MPNet', 'MLM'), ('allmpnet', 'all-MPNet-base-v2', 'contr.'),
       ('modernbert', 'ModernBERT-base', 'MLM'), ('mbembed', 'modernbert-embed-base', 'contr.'),
       ('gtemb', 'gte-modernbert-base', 'contr.'), ('bcmb', 'BioClinical-ModernBERT', 'MLM'),
       ('bcmbembed', 'BioClinical-MB-embeddings', 'contr.'), ('bge', 'BGE-base-v1.5 (unpaired)', 'contr.')]
PAIR_BREAKS = {'medcpt_q', 'allmpnet', 'gtemb', 'bcmbembed'}      # \midrule after these encoders


def load():
    df = pd.read_csv(HERE / 'metrics_long.csv')
    df['qid'] = df['group'].astype(str).str.split('|').str[0]
    return df


def per_run_track(df, tag='final', metric='raw_acc_norm', variant='std'):
    d = df[(df.tag == tag) & (df.variant == variant)]
    t = d.groupby(['run', 'family', 'encoder', 'arch', 'objective', 'w_ce', 'prior', 'frac', 'seed', 'track', 'task'],
                  dropna=False)[metric].mean().reset_index()
    return t.groupby(['run', 'family', 'encoder', 'arch', 'objective', 'w_ce', 'prior', 'frac', 'seed', 'track'],
                     dropna=False)[metric].mean().reset_index()


def cell(values, pct=True, sd=True, nd=1):
    v = np.asarray([x for x in values if x == x], float)
    if len(v) == 0:
        return '--'
    m = v.mean() * (100 if pct else 1)
    if sd and len(v) > 1:
        return f'{m:.{nd}f}\\tiny{{$\\pm${v.std(ddof=1) * (100 if pct else 1):.{nd}f}}}'
    return f'{m:.{nd}f}'


def select(t, **kw):
    for k, v in kw.items():
        t = t[t[k] == v] if v is not None else t[t[k].isna()]
    return t


def main_table(df):
    """One row per encoder with C and PFR side by side (the full per-architecture grid is in the
    appendix via main_table_full); a second block for other variants and the LLMs."""
    acc = per_run_track(df)
    ects = per_run_track(df, metric='ts_ece')
    runs = acc[['run', 'family', 'encoder', 'arch', 'objective', 'w_ce', 'prior', 'frac']].drop_duplicates()

    def val(sel, track, table=acc, col='raw_acc_norm'):
        t = table.merge(sel[['run']].drop_duplicates(), on='run')
        return cell(t[t.track == track][col])

    def pick(family, enc=None, arch=None, frac=None):
        s = runs[runs.family == family]
        if enc is not None:
            s = s[s.encoder == enc]
        if arch is not None:
            s = s[s.arch == arch]
        if frac is not None:
            s = s[s.frac == frac]
        return s
    rows = []
    for enc, name, kind in ENC:
        C, P = pick('main', enc, 'C'), pick('main', enc, 'PFR')
        c10 = pick('lc', enc, 'C', 0.1)
        cells = [val(C, 'seen'), val(P, 'seen'), val(c10, 'seen'), val(C, 'heldout'), val(P, 'heldout'),
                 val(C, 'clinical'), val(C, 'seen', ects, 'ts_ece')]
        rows.append(f'{name} & {kind} & ' + ' & '.join(cells) + r' \\')
        if enc in PAIR_BREAKS:
            rows.append(r'\midrule')
    rows.append(r'\midrule')
    other = [('PubMedBERT, B', pick('bi', 'pubmedbert')), ('S-PubMedBERT, B', pick('bi', 'spubmedbert')),
             ('S-PubMedBERT, C trained with CE', runs[(runs.family == 'rq3c') & (runs.objective == 'ce')]),
             ('S-PubMedBERT, \\pfr{} frozen prior', pick('frozen')), ('S-PubMedBERT, \\pfr{} +clinical train', pick('clinical')),
             ('Laya-large (421M), C fine-tuned', pick('laya_ft')),
             ('Qwen3.8-27B, zero-shot', pick('llm', 'qwen38_27b')), ('Gemma-4-31B, zero-shot', pick('llm', 'gemma4_31b'))]
    for name, sel in other:
        cells = [val(sel, 'seen'), '', '--', val(sel, 'heldout'), '', val(sel, 'clinical'), val(sel, 'seen', ects, 'ts_ece')]
        rows.append(rf'\multicolumn{{2}}{{@{{}}l}}{{{name}}} & ' + ' & '.join(cells) + r' \\')
        if name.startswith('Laya'):
            rows.append(r'\midrule')
    body = '\n'.join(rows)
    return rf"""\begin{{table*}}[t]
\centering\small
\setlength{{\tabcolsep}}{{4.5pt}}
\begin{{tabular}}{{@{{}}llccccccc@{{}}}}
\toprule
 & & \multicolumn{{3}}{{c}}{{Seen tasks, acc$_{{\text{{cn}}}}$}} & \multicolumn{{2}}{{c}}{{Held-out, acc$_{{\text{{cn}}}}$}} & Clinical & ECE$_{{\text{{TS}}}}$ \\
\cmidrule(lr){{3-5}}\cmidrule(lr){{6-7}}
Encoder & Type & C & \pfr{{}} & C, 10\% & C & \pfr{{}} & C & C \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\caption{{Main results ($\times100$; mean$\pm$sd over 3 seeds; 10\% also uses 3 seeds). Encoders are grouped by
matched pair (MLM parent, then contrastive children). acc$_{{\text{{cn}}}}$: chance-normalised accuracy,
averaged over question groups within a task and then over tasks. Seen: the 8 grounded training tasks.
Held-out: PUBHEALTH, BIOSSES, MTSamples. Clinical: MedNLI, MIMIC-III trial questions and MIMIC-IV name
gate, zero-shot except ``+clinical train''. ECE$_{{\text{{TS}}}}$: top-label ECE after temperature scaling
on seen tasks. All rows are trained with the released RLCD (PG) recipe except ``C trained with CE''. In the lower block the C columns hold the row's own model. Knowledge-track results,
raw ECE and all architectures per encoder are in Appendix~\ref{{app:full}}.}}
\label{{tab:main}}
\end{{table*}}
"""


def main_table_full(df):
    """Appendix: every encoder x architecture, with raw and scaled ECE and the knowledge track."""
    acc = per_run_track(df)
    ece = per_run_track(df, metric='raw_ece')
    ects = per_run_track(df, metric='ts_ece')
    runs = acc[['run', 'family', 'encoder', 'arch']].drop_duplicates()
    rows = []
    for enc, name, kind in ENC:
        for arch, fam in (('C', 'main'), ('PFR', 'main'), ('B', 'bi')):
            sel = runs[(runs.family == fam) & (runs.encoder == enc) & (runs.arch == arch)]
            if sel.empty:
                continue
            a = acc.merge(sel[['run']], on='run')
            e = ece.merge(sel[['run']], on='run')
            et = ects.merge(sel[['run']], on='run')
            cells = [cell(a[a.track == 'seen'].raw_acc_norm), cell(e[e.track == 'seen'].raw_ece),
                     cell(et[et.track == 'seen'].ts_ece)] + [cell(a[a.track == t].raw_acc_norm) for t in ('heldout', 'clinical', 'knowledge')]
            rows.append(f'{name if arch == "C" else ""} & {arch} & ' + ' & '.join(cells) + r' \\')
        if enc in PAIR_BREAKS:
            rows.append(r'\midrule')
    body = '\n'.join(rows)
    return rf"""\begin{{table*}}[t]
\centering\small
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{@{{}}llcccccc@{{}}}}
\toprule
Encoder & Arch & Seen acc$_{{\text{{cn}}}}$ & ECE & ECE$_{{\text{{TS}}}}$ & Held-out & Clinical & Knowledge \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\caption{{All encoders and architectures ($\times100$; mean$\pm$sd over 3 seeds).}}
\label{{tab:full}}
\end{{table*}}
"""


def zeroshot_table(df):
    d = df[df.variant == 'std'].copy()
    d['content'] = [((t, q) in CONTENT or (t, '*') in CONTENT) for t, q in zip(d.task, d.qid)]
    d = d[d.content | d.task.isin(LABEL_ONLY_TASKS)]

    def score(sel, tag):
        x = d[(d.run.isin(sel)) & (d.tag == tag)]
        out = []
        for flag in (True, False):
            y = x[x.content == flag].groupby(['run', 'task']).raw_acc_norm.mean().groupby('run').mean()
            out.append(cell(y.values))
        return out
    runs = d[['run', 'family', 'encoder', 'arch']].drop_duplicates()
    rows = []
    for enc, name, kind in ENC:
        z = runs[(runs.family == 'zeroshot') & (runs.encoder == enc)].run
        c0 = runs[(runs.family == 'main') & (runs.encoder == enc) & (runs.arch == 'C')].run
        rows.append(f'{name} & ' + ' & '.join(score(z, 'final') + score(c0, 'init')) + ' \\\\')
    lz = runs[(runs.family == 'zeroshot') & (runs.encoder == 'laya')].run
    rows.append('\\midrule')
    rows.append('Laya-large (Laya\'s own head) & ' + ' & '.join(score(lz, 'final')) + ' & -- & -- \\\\')
    body = '\n'.join(rows)
    return rf"""\begin{{table}}[t]
\centering\small
\setlength{{\tabcolsep}}{{3.5pt}}
\begin{{tabular}}{{@{{}}lcccc@{{}}}}
\toprule
 & \multicolumn{{2}}{{c}}{{Z = \pfr{{}} at step 0}} & \multicolumn{{2}}{{c}}{{C at step 0}} \\
\cmidrule(lr){{2-3}}\cmidrule(lr){{4-5}}
Encoder & content & label & content & label \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\caption{{Untrained (step-0) chance-normalised accuracy ($\times100$, after temperature scaling on the
calibration split) on questions whose options differ in \emph{{content}} (MEDLINE design and disease
area, MTSamples, exam answers, Druglib and BIOSSES rubrics) or only by a \emph{{label}} word
(\noul{{}}, PubMedQA, verification, DDI, MedNLI). C at step 0: 3 seeds.}}
\label{{tab:zeroshot}}
\end{{table}}
"""


def grad_var(runs_dir, run):
    f = Path(runs_dir) / run / 'train_log.jsonl'
    if not f.exists():
        return np.nan
    v = [json.loads(l).get('grad_var') for l in f.read_text().splitlines()]
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else np.nan


def rlcd_table(df, runs_dir):
    acc = per_run_track(df)
    ece = per_run_track(df, metric='raw_ece')
    ects = per_run_track(df, metric='ts_ece')
    nll = per_run_track(df, metric='ts_nll')
    d = df[(df.tag == 'final') & (df.variant == 'std') & (df.task == 'druglib')]
    mae = d.groupby('run').raw_mae.mean()
    runs = acc[['run', 'family', 'encoder', 'arch', 'objective', 'w_ce']].drop_duplicates()
    specs = [('CE', (runs.family == 'rq3') & (runs.objective == 'ce')),
             ('Proper', (runs.family == 'rq3') & (runs.objective == 'proper')),
             ('PG (released RLCD)', (runs.family == 'main') & (runs.encoder == 'spubmedbert') & (runs.arch == 'PFR')),
             ('PG w/o CE', (runs.family == 'rq3') & (runs.objective == 'rlcd_pg')),
             ('Reparam', (runs.family == 'rq3') & (runs.objective == 'rlcd_reparam'))]
    rows = []
    for name, mask in specs:
        sel = set(runs[mask].run)
        a = acc[acc.run.isin(sel) & (acc.track == 'seen')].raw_acc_norm
        e = ece[ece.run.isin(sel) & (ece.track == 'seen')].raw_ece
        et = ects[ects.run.isin(sel) & (ects.track == 'seen')].ts_ece
        n = nll[nll.run.isin(sel) & (nll.track == 'seen')].ts_nll
        m = mae[mae.index.isin(sel)]
        gv = [grad_var(runs_dir, r) for r in sel] if 'PG' in name or name == 'Reparam' else []
        gv_s = f'{np.nanmean(gv):.2g}' if gv and not np.all(np.isnan(gv)) else 'n/a'
        rows.append(f'{name} & {cell(a)} & {cell(e)} & {cell(et)} & {cell(n, pct=False, nd=3)} & {cell(m, pct=False, nd=3)} & {gv_s} \\\\')
    body = '\n'.join(rows)
    return rf"""\begin{{table}}[t]
\centering\small
\setlength{{\tabcolsep}}{{2.2pt}}
\begin{{tabular}}{{@{{}}lcccccc@{{}}}}
\toprule
Objective & acc$_{{\text{{cn}}}}$ & ECE & ECE$_{{\text{{TS}}}}$ & NLL$_{{\text{{TS}}}}$ & MAE$_{{\score}}$ & $\mathrm{{Var}}[\hat\nabla]$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\caption{{RLCD ablation on the seen tasks (\pfr{{}} on S-PubMedBERT-MS-MARCO, mean$\pm$sd over 3 seeds;
acc and ECE $\times100$). TS: after temperature scaling. MAE$_{{\score}}$: error of the expected level
on Druglib. $\mathrm{{Var}}[\hat\nabla]$: variance of the logit gradient of the noise-smoothed term
across 8 draws at training checkpoints, averaged over logged probes. PG and Reparam use different normalisations and training states; these values do not isolate estimator variance.}}
\label{{tab:rlcd}}
\end{{table}}
"""


def write_all(runs_dir, out_dir):
    df = load()
    (Path(out_dir) / 'main_results.tex').write_text(main_table(df), encoding='utf-8')
    (Path(out_dir) / 'main_full.tex').write_text(main_table_full(df), encoding='utf-8')
    (Path(out_dir) / 'zeroshot.tex').write_text(zeroshot_table(df), encoding='utf-8')
    (Path(out_dir) / 'rlcd.tex').write_text(rlcd_table(df, runs_dir), encoding='utf-8')
    print('wrote main_results.tex, zeroshot.tex, rlcd.tex')


GPU_NAME = os.environ.get('S1_GPU_NAME', 'GPU')

def efficiency_table(df, runs_dir):
    """Retrieval retention (S-PubMedBERT-based models, seed 0) and batch-1 latency on one GPU (name: S1_GPU_NAME)."""
    ret = df[df.task.astype(str).str.startswith('ret_')]
    lat = json.loads((Path(runs_dir) / 'latency' / 'latency_cuda.json').read_text())['results']

    def nd(fam, arch, prior='shared', enc='spubmedbert'):
        x = ret[(ret.family == fam) & (ret.arch == arch) & (ret.encoder == enc) & (ret.prior == prior)]
        return [f"{x[x.task == t].metric_ndcg10.mean() * 100:.1f}" if (x.task == t).any() else '--'
                for t in ('ret_scifact', 'ret_nfcorpus')]

    def ms(key, n='10'):
        v = lat.get(key, {}).get(n)
        return f"{v['p50_ms']:.1f}" if isinstance(v, dict) and 'p50_ms' in v else '--'
    rows = [('Z (no training)', nd('zeroshot', 'Z'), ms('pubmedbert/Z')),
            ('B', nd('bi', 'B'), ms('pubmedbert/B')),
            ('C', nd('main', 'C'), ms('pubmedbert/C')),
            ('C, trained with CE', nd('rq3c', 'C'), ms('pubmedbert/C')),
            (r'\pfr{}, shared prior', nd('main', 'PFR'), ms('pubmedbert/PFR')),
            (r'\pfr{}, frozen prior', nd('frozen', 'PFR', 'frozen'), ms('pubmedbert/PFR')),
            ('Laya-large C (421M)', ['--', '--'], ms('laya/C'))]
    body = '\n'.join(f'{n} & {r[0]} & {r[1]} & {l} ' + r'\\' for n, r, l in rows)
    return rf"""\begin{{table}}[t]
\centering\small
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{@{{}}lccc@{{}}}}
\toprule
 & \multicolumn{{2}}{{c}}{{nDCG@10 ($\times100$)}} & Latency \\
\cmidrule(lr){{2-3}}
Model (S-PubMedBERT) & SciFact & NFCorpus & ms \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\caption{{Retrieval retention of the pooled embedding after conversion (seed 0), and p50 latency of one
request with 10 questions at batch size 1 on one {GPU_NAME} (base-encoder latencies measured on the
PubMedBERT architecture, which S-PubMedBERT shares). Times cover the model forward pass only, excluding tokenisation and host/device transfers.}}
\label{{tab:eff}}
\end{{table}}
"""


def write_eff(runs_dir, out_dir):
    (Path(out_dir) / 'efficiency.tex').write_text(efficiency_table(load(), runs_dir), encoding='utf-8')


ROBUST_ROWS = [('main', 'pubmedbert', 'C', 'PubMedBERT C'), ('main', 'spubmedbert', 'C', 'S-PubMedBERT C'),
               ('main', 'spubmedbert', 'PFR', 'S-PubMedBERT PFR'), ('frozen', 'spubmedbert', 'PFR', 'S-PubMedBERT PFR, frozen'),
               ('bi', 'spubmedbert', 'B', 'S-PubMedBERT B'), ('main', 'allmpnet', 'C', 'all-MPNet C'),
               ('main', 'allmpnet', 'PFR', 'all-MPNet PFR'), ('clinical', 'spubmedbert', 'PFR', 'S-PubMedBERT PFR, +clinical'),
               ('laya_ft', 'laya', 'C', 'Laya-large FT'), ('zeroshot', 'laya', 'C', 'Laya-large zero-shot')]
STRATS = ['long_truncate', 'long_window_mean', 'long_window_conf', 'long_rtd']


def robust_table(df):
    df = df.copy()
    df['qid'] = df['group'].astype(str).str.split('|').str[0]
    perm = df[df.variant == 'perm']
    long = df[(df.tag == 'final') & (df.task == 'mimic_trialq') & df.variant.isin(STRATS)]
    gate = df[(df.tag == 'final') & (df.variant == 'std') & (df.task == 'mimic_deid')]
    rows = []
    for fam, enc, arch, name in ROBUST_ROWS:
        sel = lambda d: d[(d.family == fam) & (d.encoder == enc) & (d.arch == arch)]  # noqa: E731
        p = sel(perm).perm_flip.mean() * 100
        cells = [f'{p:.1f}' if p == p else '--']
        for qid in ('answer', 'answerable'):
            for s in STRATS:
                v = sel(long)[(sel(long).qid == qid) & (sel(long).variant == s)].raw_acc_norm.mean() * 100
                cells.append(f'{v:.1f}' if v == v else '--')
        for split in ('mimic_deid/test', 'mimic_deid/polysemy'):
            v = sel(gate)[sel(gate).split == split].raw_auroc.mean() * 100
            cells.append(f'{v:.1f}' if v == v else '--')
        rows.append(f'{name} & ' + ' & '.join(cells) + ' ' + '\\' * 2)
    body = '\n'.join(rows)
    return rf"""\begin{{table*}}[t]
\centering\small
\setlength{{\tabcolsep}}{{3.5pt}}
\begin{{tabular}}{{@{{}}lc cccc cccc cc@{{}}}}
\toprule
 & Perm. & \multicolumn{{4}}{{c}}{{Trial Qs: answer (acc$_{{\text{{cn}}}}$)}} & \multicolumn{{4}}{{c}}{{Trial Qs: answerable (acc$_{{\text{{cn}}}}$)}} & \multicolumn{{2}}{{c}}{{Name gate AUROC}} \\
\cmidrule(lr){{3-6}}\cmidrule(lr){{7-10}}\cmidrule(lr){{11-12}}
Model & flip\,\% & trunc & win-mean & win-conf & RtD & trunc & win-mean & win-conf & RtD & test & common-word \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\caption{{Robustness ($\times100$; mean over 3 seeds where available). Perm.\ flip: share of answers
that change under a random option permutation (5 permutations), averaged over all evaluation splits; on the seen tasks alone C heads flip 2--3\% and \pfr{{}} below 0.5\% (Table~\ref{{tab:objfull}}). Trial Qs: MIMIC-III
trial-eligibility questions over notes longer than the 512-token context, decided by truncation,
sliding windows (mean of logits or most confident window) or retrieve-then-decide (RtD). Name gate:
MIMIC-IV windows, standard and common-word-name splits. Both LLMs reach AUROC 100.0 on both
name-gate splits.}}
\label{{tab:robust}}
\end{{table*}}
"""


def write_robust(out_dir):
    (Path(out_dir) / 'robustness.tex').write_text(robust_table(load()), encoding='utf-8')


def fit_width(path, width=r'\columnwidth'):
    """Scale a generated tabular to the given width (tables here are a few points too wide)."""
    p = Path(path)
    s = p.read_text(encoding='utf-8')
    if 'resizebox' not in s:
        s = s.replace(r'\begin{tabular}', r'\resizebox{' + width + r'}{!}{\begin{tabular}', 1)
        s = s.replace(r'\end{tabular}', r'\end{tabular}}', 1)
        p.write_text(s, encoding='utf-8')


def write_everything(runs_dir, out_dir):
    write_all(runs_dir, out_dir)
    write_eff(runs_dir, out_dir)
    write_robust(out_dir)
    for name in ('zeroshot', 'rlcd', 'efficiency'):
        fit_width(Path(out_dir) / f'{name}.tex')
    fit_width(Path(out_dir) / 'robustness.tex', r'\textwidth')
    fit_width(Path(out_dir) / 'main_results.tex', r'\textwidth')
    fit_width(Path(out_dir) / 'main_full.tex', r'\textwidth')
