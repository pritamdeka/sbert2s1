"""Tables and figures added in the final revision (head x objective grid, matched gradient probe,
optimisation traces, calibration diagrams, per-task results, complete contrast tables).

    python paper/scripts/revision_tables.py runs
Inputs: metrics_long.csv (collect.py), analysis/contrasts.json (bootstrap.py), runs/*/train_log.jsonl,
runs/probe/gradvar/grad_probe.json and saved public-task predictions.
"""
import gzip
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
V2 = HERE.parents[1]
TAB, FIG = HERE.parent / 'tables', HERE.parent / 'figures'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(V2))
from results_tables import cell, load, per_run_track  # noqa: E402

SEEN = ['pubmedqa', 'scifact', 'healthver', 'ddi', 'hoc', 'ade', 'druglib', 'medline_s1']
HELD = ['pubhealth', 'biosses', 'mtsamples']
TASK_NAMES = {'pubmedqa': 'PubMedQA', 'scifact': 'SciFact', 'healthver': 'HealthVer', 'ddi': 'DDI', 'hoc': 'HoC',
              'ade': 'ADE', 'druglib': 'Druglib', 'medline_s1': 'MEDLINE', 'pubhealth': 'PUBHEALTH',
              'biosses': 'BIOSSES', 'mtsamples': 'MTSamples'}
OBJ = [('CE', 'ce', 1.0), ('Proper', 'proper', 1.0), ('PG (released RLCD)', 'rlcd_pg', 1.0),
       ('PG w/o CE', 'rlcd_pg', 0.0), ('PG-LOO (unbiased)', 'rlcd_pg_loo', 1.0), ('Reparam', 'rlcd_reparam', 1.0)]
SHORT = {'CE': 'CE', 'Proper': 'Proper', 'PG (released RLCD)': 'PG', 'PG w/o CE': 'PG w/o CE',
         'PG-LOO (unbiased)': 'PG-LOO', 'Reparam': 'Reparam'}
plt.rcParams.update({'font.size': 7.5, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42,
                     'font.family': 'serif'})
# Okabe-Ito palette (colour-blind safe) plus distinct markers, so figures also read in greyscale
COL = {'rlcd_pg': '#D55E00', 'pg_mean': '#E69F00', 'rlcd_pg_loo': '#0072B2', 'rlcd_reparam': '#009E73',
       'ce': '#000000', 'proper': '#CC79A7'}
MARK = {'rlcd_pg': 'o', 'pg_mean': 's', 'rlcd_pg_loo': '^', 'rlcd_reparam': 'D', 'ce': 'v', 'proper': 'P'}
PFR = r'\pfr{}'


def head_obj_runs(df):
    """(head, objective name) -> sorted run ids, S-PubMedBERT-MS-MARCO."""
    r = df[['run', 'family', 'encoder', 'arch', 'objective', 'w_ce']].drop_duplicates()
    out = {}
    for name, obj, wce in OBJ:
        if obj == 'rlcd_pg' and wce == 1.0:
            pfr = r[(r.family == 'main') & (r.encoder == 'spubmedbert') & (r.arch == 'PFR')]
            c = r[(r.family == 'main') & (r.encoder == 'spubmedbert') & (r.arch == 'C')]
        else:
            pfr = r[(r.family == 'rq3') & (r.objective == obj) & (r.w_ce == wce)]
            c = r[(r.family == 'rq3c') & (r.objective == obj) & (r.w_ce == wce)]
        out[('PFR', name)] = sorted(pfr.run)
        out[('C', name)] = sorted(c.run)
    return out


def vals(table, runs, track, col):
    t = table[table.run.isin(runs) & (table.track == track)]
    return t[col].values


def grad_norms(runs_dir, run):
    f = Path(runs_dir) / run / 'train_log.jsonl'
    if not f.exists():
        return None
    rows = [json.loads(l) for l in f.read_text().splitlines()]
    return pd.DataFrame([dict(step=r['step'], gn=r['grad_norm']) for r in rows])


def objectives_tables(df, runs_dir):
    acc = per_run_track(df)
    ects = per_run_track(df, metric='ts_ece')
    ece = per_run_track(df, metric='raw_ece')
    nll = per_run_track(df, metric='ts_nll')
    d = df[(df.tag == 'final') & (df.variant == 'std') & (df.task == 'druglib')]
    mae = d.groupby('run').raw_mae.mean()
    perm = df[(df.variant == 'perm') & (df.track == 'seen')].groupby('run').perm_flip.mean()
    cells = head_obj_runs(df)
    main_rows, full_rows = [], []
    for name, obj, wce in OBJ:
        p, c = cells[('PFR', name)], cells[('C', name)]

        def m(runs, tab=acc, track='seen', col='raw_acc_norm'):
            v = vals(tab, runs, track, col)
            return f'{v.mean() * 100:.1f}' if len(v) else '--'
        main_rows.append(f'{name} & {m(p)} & {m(c)} & {m(p, track="heldout")} & {m(c, track="heldout")} & '
                         f'{m(p, ects, col="ts_ece")} & {m(c, ects, col="ts_ece")} \\\\')
        for head, runs in ((PFR, p), ('C', c)):
            lab = name if head == PFR else ''
            if not runs:
                full_rows.append(f'{lab} & {head} & \\multicolumn{{10}}{{c}}{{not run}} \\\\')
                continue
            gn = [g for g in (grad_norms(runs_dir, r) for r in runs) if g is not None]
            gmed = np.median(np.concatenate([g.gn.values for g in gn])) if gn else np.nan
            row = [cell(vals(acc, runs, 'seen', 'raw_acc_norm')), cell(vals(acc, runs, 'heldout', 'raw_acc_norm')),
                   cell(vals(acc, runs, 'clinical', 'raw_acc_norm')), cell(vals(acc, runs, 'knowledge', 'raw_acc_norm')),
                   cell(vals(ece, runs, 'seen', 'raw_ece')), cell(vals(ects, runs, 'seen', 'ts_ece')),
                   cell(vals(nll, runs, 'seen', 'ts_nll'), pct=False, nd=3),
                   cell(mae[mae.index.isin(runs)].values, pct=False, nd=3),
                   cell(perm[perm.index.isin(runs)].values), f'{gmed:.1f}']
            full_rows.append(f'{lab} & {head} & ' + ' & '.join(row) + ' \\\\')
        if name != OBJ[-1][0]:
            full_rows.append('\\midrule')
    main = r'''\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{3pt}
\resizebox{\columnwidth}{!}{\begin{tabular}{@{}lcccccc@{}}
\toprule
 & \multicolumn{2}{c}{Seen acc$_{\text{cn}}$} & \multicolumn{2}{c}{Held-out acc$_{\text{cn}}$} & \multicolumn{2}{c}{ECE$_{\text{TS}}$} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Objective & \pfr{} & C & \pfr{} & C & \pfr{} & C \\
\midrule
''' + '\n'.join(main_rows) + r'''
\bottomrule
\end{tabular}}
\caption{Head $\times$ objective on S-PubMedBERT-MS-MARCO ($\times100$, mean over 3 seeds; sd and further
metrics in Table~\ref{tab:objfull}). PG is the released RLCD recipe, used for all other trained models.
PG-LOO replaces its same-sample mean baseline and reward normalisation with a leave-one-out baseline.
PG w/o CE was run for \pfr{} only.}
\label{tab:objectives}
\end{table}
'''
    full = r'''\begin{table*}[t]
\centering\small
\setlength{\tabcolsep}{3pt}
\resizebox{\textwidth}{!}{\begin{tabular}{@{}llcccccccccc@{}}
\toprule
Objective & Head & Seen & Held-out & Clinical & Knowledge & ECE & ECE$_{\text{TS}}$ & NLL$_{\text{TS}}$ & MAE$_{\score}$ & Perm.\ flip & $\lVert\nabla\rVert$ \\
\midrule
''' + '\n'.join(full_rows) + r'''
\bottomrule
\end{tabular}}
\caption{Complete head $\times$ objective grid on S-PubMedBERT-MS-MARCO (mean$\pm$sd over 3 seeds; accuracy,
ECE and flip rates $\times100$). Columns 3--6: chance-normalised accuracy by track. ECE, NLL: seen tasks, top-label,
before and after temperature scaling. MAE$_{\score}$: error of the expected Druglib level. Perm.\ flip: share of
seen-task answers that change under a random option permutation. $\lVert\nabla\rVert$: median global gradient norm
before clipping; the clipping threshold is 1, so every step of every objective is clipped.}
\label{tab:objfull}
\end{table*}
'''
    (TAB / 'objectives.tex').write_text(main, encoding='utf-8')
    (TAB / 'objectives_full.tex').write_text(full, encoding='utf-8')


EST = [('rlcd_pg', 'released PG'), ('pg_mean', 'mean baseline, unnormalised'), ('rlcd_pg_loo', 'PG-LOO'),
       ('rlcd_reparam', 'pathwise (Reparam)')]


def probe_outputs(runs_dir):
    pj = json.loads((Path(runs_dir) / 'probe' / 'gradvar' / 'grad_probe.json').read_text())
    names = {'main/spubmedbert/PFR/s0': (PFR, 'PG'), 'rq3/pg_loo/s0': (PFR, 'PG-LOO'), 'rq3c/ce/s0': ('C', 'CE')}
    rows = []
    for key, run in pj['runs'].items():
        if 'sigma' not in run:
            continue
        head, trained = names.get(key.replace('runs/', ''), (key, ''))
        for i, (e, lab) in enumerate(EST):
            s = [run['sigma'][str(sg)][e] for sg in (0.4, 0.25, 0.1)]
            rows.append((f'{head}, {trained}' if i == 0 else '') + f' & {lab} & ' +
                        ' & '.join(f'{x["slope"]:.2f}' for x in s) + ' & ' +
                        ' & '.join(f'{x["score_vs_ce"]:.2f}' for x in (s[0], s[2])) + ' & ' +
                        f'{s[1]["cosine"]:.3f} & {s[1]["snr"]:.3g} \\\\')
        rows.append('\\midrule')
    rows = rows[:-1]
    tex = r'''\begin{table*}[t]
\centering\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}llccccccc@{}}
\toprule
 & & \multicolumn{3}{c}{Slope vs.\ $\nabla J_\sigma$} & \multicolumn{2}{c}{Score : CE weight} & Cosine & SNR \\
\cmidrule(lr){3-5}\cmidrule(lr){6-7}
Checkpoint (head, objective) & Estimator & $\sigma{=}0.4$ & $0.25$ & $0.1$ & $\sigma{=}0.4$ & $0.1$ & $\sigma{=}0.25$ & $\sigma{=}0.25$ \\
\midrule
''' + '\n'.join(rows) + r'''
\bottomrule
\end{tabular}
\caption{Matched gradient-estimator probe. For each saved checkpoint, the logits of the same 40 training batches
(32 items each) are fixed, and 32 independent estimates of the smoothed-score gradient ($G{=}4$, no CE term)
are drawn per estimator and noise level. Slope: projection of the mean estimate onto a 2{,}048-sample pathwise
reference $\nabla J_\sigma$ (1 is the correct scale). Score : CE weight: slope $\times\lVert\nabla J_\sigma\rVert/\lVert\nabla\mathrm{CE}\rVert$,
the effective weight of the smoothed-score term relative to the CE term in a training step. SNR: per-item
$\lVert\nabla J_\sigma\rVert^2$ divided by the scale-matched variance of one estimate.}
\label{tab:probe}
\end{table*}
'''
    (TAB / 'probe.tex').write_text(tex, encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.35))
    sig = [0.4, 0.25, 0.1]
    good = [run for run in pj['runs'].values() if 'sigma' in run]
    for e, lab in EST:
        sl = np.mean([[run['sigma'][str(s)][e]['slope'] for s in sig] for run in good], 0)
        sn = np.mean([[run['sigma'][str(s)][e]['snr'] for s in sig] for run in good], 0)
        axes[0].plot(sig, sl, color=COL[e], marker=MARK[e], ms=4, lw=1.2, label=lab)
        axes[1].plot(sig, sn, color=COL[e], marker=MARK[e], ms=4, lw=1.2, label=lab)
    for ax, yl in zip(axes, ('scale relative to true gradient', 'per-item SNR (matched scale)')):
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks(sig)
        ax.set_xticklabels(['0.4', '0.25', '0.1'])
        ax.minorticks_off()
        ax.invert_xaxis()
        ax.set_xlabel(r'noise $\sigma$ (training anneals left to right)')
        ax.set_ylabel(yl)
    axes[0].axhline(1, color='#888888', lw=0.8, ls=':')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), fontsize=6.5, frameon=False)
    fig.tight_layout(pad=0.4, rect=(0, 0.09, 1, 1))
    fig.savefig(FIG / 'probe.pdf')
    plt.close(fig)


def gradnorm_figure(df, runs_dir):
    cells = head_obj_runs(df)
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.35), sharey=True)
    for ax, head in zip(axes, ('PFR', 'C')):
        for name, obj, wce in OBJ:
            if wce == 0.0:
                continue
            gs = [g for g in (grad_norms(runs_dir, r) for r in cells[(head, name)]) if g is not None]
            if not gs:
                continue
            m = pd.concat(gs).groupby('step').gn.median().rolling(5, min_periods=1).median()
            ax.plot(m.index, m.values, color=COL[obj], lw=1.1, label=SHORT[name], marker=MARK[obj],
                    markevery=max(1, len(m) // 6), ms=3)
        ax.set_yscale('log')
        ax.axhline(1, color='#888888', lw=0.8, ls=':')
        ax.set_title('PFR head' if head == 'PFR' else 'C head', fontsize=7.5)
        ax.set_xlabel('training step')
    axes[0].set_ylabel('global gradient norm (pre-clip)')
    seen_labels, handles = {}, []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen_labels:
                seen_labels[l] = h
    fig.legend(list(seen_labels.values()), list(seen_labels), loc='lower center', ncol=len(seen_labels),
               fontsize=6.5, frameon=False)
    fig.tight_layout(pad=0.4, rect=(0, 0.09, 1, 1))
    fig.savefig(FIG / 'gradnorm.pdf')
    plt.close(fig)


def reliability_figure(runs_dir):
    """Top-label reliability on seen tasks (items pooled over tasks and seeds), raw vs temperature-scaled."""
    from s1.temperature_lookup import temperature_for
    runs_dir = Path(runs_dir)
    models = [('S-PubMedBERT C, CE', 'rq3c/ce/s*'), ('S-PubMedBERT C, PG', 'main/spubmedbert/C/s*'),
              ('S-PubMedBERT PFR, CE', 'rq3/ce/s*'), ('Gemma-4-31B, zero-shot', 'llm/gemma4_31b')]
    bins = np.linspace(0, 1, 11)
    fig, axes = plt.subplots(1, 4, figsize=(6.6, 2.3), sharey=True)
    for ax, (name, pat) in zip(axes, models):
        eces = {}
        conf_raw, conf_ts, corr = [], [], []
        for run in sorted(runs_dir.glob(pat)):
            tp = json.loads((run / 'temps.json').read_text()) if (run / 'temps.json').exists() else {}
            for task in SEEN:
                f = run / 'preds' / 'final' / f'{task}.test.jsonl.gz'
                if not f.exists():
                    continue
                with gzip.open(f, 'rt') as fh:
                    for line in fh:
                        r = json.loads(line)
                        z = np.asarray(r['logits'], float)
                        T = temperature_for(tp, r['qtype'], len(z)) if tp else 1.0
                        for TT, store in ((1.0, conf_raw), (T, conf_ts)):
                            p = np.exp((z - z.max()) / TT)
                            store.append(p.max() / p.sum())
                        corr.append(float(z.argmax() == int(np.argmax(r['target']))))
        corr = np.asarray(corr)
        for conf, style, lab in ((np.asarray(conf_raw), dict(color='#D55E00', marker='o', ls='--'), 'raw'),
                                 (np.asarray(conf_ts), dict(color='#0072B2', marker='s', ls='-'), 'scaled')):
            idx = np.clip(np.digitize(conf, bins) - 1, 0, 9)
            keep = [b for b in range(10) if (idx == b).sum() >= 30]
            ece = sum((idx == b).mean() * abs(conf[idx == b].mean() - corr[idx == b].mean())
                      for b in range(10) if (idx == b).any())
            eces[lab] = ece
            ax.plot([conf[idx == b].mean() for b in keep], [corr[idx == b].mean() for b in keep], ms=3, lw=1,
                    label={'raw': 'raw confidence', 'scaled': 'temperature-scaled'}[lab], **style)
        ax.plot([0, 1], [0, 1], color='#888888', lw=0.7, ls=':')
        ax.set_title(f"{name}\nECE {eces['raw'] * 100:.1f} raw, {eces['scaled'] * 100:.1f} scaled", fontsize=6.6)
        ax.set_xlabel('confidence')
        ax.set_xlim(0.2, 1.0)
        ax.set_ylim(0.2, 1.0)
    axes[0].set_ylabel('accuracy')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=6.5, frameon=False)
    fig.tight_layout(pad=0.35, rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / 'reliability.pdf')
    plt.close(fig)


def per_task_table(df):
    d = df[(df.tag == 'final') & (df.variant == 'std') & df.task.isin(SEEN + HELD)]
    t = d.groupby(['run', 'family', 'encoder', 'arch', 'objective', 'task'], dropna=False).raw_acc_norm.mean().reset_index()
    models = [('PubMedBERT C (PG)', dict(family='main', encoder='pubmedbert', arch='C')),
              ('S-PubMedBERT C (PG)', dict(family='main', encoder='spubmedbert', arch='C')),
              ('S-PubMedBERT C (CE)', dict(family='rq3c', objective='ce')),
              ('S-PubMedBERT \\pfr{} (PG)', dict(family='main', encoder='spubmedbert', arch='PFR')),
              ('S-PubMedBERT \\pfr{} (CE)', dict(family='rq3', objective='ce')),
              ('MedCPT-Query \\pfr{} (PG)', dict(family='main', encoder='medcpt_q', arch='PFR')),
              ('all-MPNet \\pfr{} (PG)', dict(family='main', encoder='allmpnet', arch='PFR')),
              ('BioClinical-ModernBERT C (PG)', dict(family='main', encoder='bcmb', arch='C')),
              ('Laya-large C (PG)', dict(family='laya_ft')),
              ('Qwen3.8-27B (zero-shot)', dict(family='llm', encoder='qwen38_27b')),
              ('Gemma-4-31B (zero-shot)', dict(family='llm', encoder='gemma4_31b'))]
    rows = []
    for name, sel in models:
        x = t
        for k, v in sel.items():
            x = x[x[k] == v]
        vv = [x[x.task == task].raw_acc_norm.mean() * 100 for task in SEEN + HELD]
        rows.append(name + ' & ' + ' & '.join('--' if v != v else f'{v:.1f}' for v in vv) + ' \\\\')
        if name.startswith('Laya'):
            rows.append('\\midrule')
    head = ' & '.join(TASK_NAMES[t] for t in SEEN) + ' & ' + ' & '.join(TASK_NAMES[t] for t in HELD)
    tex = r'''\begin{table*}[t]
\centering\small
\setlength{\tabcolsep}{3pt}
\resizebox{\textwidth}{!}{\begin{tabular}{@{}l cccccccc ccc@{}}
\toprule
 & \multicolumn{8}{c}{Seen (grounded training tasks)} & \multicolumn{3}{c}{Held-out} \\
\cmidrule(lr){2-9}\cmidrule(lr){10-12}
Model & ''' + head + r''' \\
\midrule
''' + '\n'.join(rows) + r'''
\bottomrule
\end{tabular}}
\caption{Per-task chance-normalised accuracy ($\times100$; mean over seeds; question groups averaged within a task).
The training objective is given in parentheses.}
\label{tab:pertask}
\end{table*}
'''
    (TAB / 'per_task.tex').write_text(tex, encoding='utf-8')


def contrast_tables():
    data = json.loads((V2 / 'analysis' / 'contrasts.json').read_text())
    res = data['results']
    by = {(r['label'], r['track']): r for r in res}
    labels = list(dict.fromkeys(r['label'] for r in res))

    def fmt(r):
        if r is None:
            return '--', '--'
        p = r['p_holm']
        ps = '$<$.001' if p < 0.001 else ('1' if p >= 1 else f'{p:.3f}'.lstrip('0'))
        m = lambda x: f"{x:+.1f}".replace('-', '$-$')
        return f"{m(r['delta'])} [{m(r['lo'])}, {m(r['hi'])}]", ps

    def rows_for(fams):
        out = []
        for lab in labels:
            s, h = by.get((lab, 'seen')), by.get((lab, 'heldout'))
            r = s or h
            if r['rq'] not in fams:
                continue
            (sd, sp), (hd, hp) = fmt(s), fmt(h)
            tl = lab.replace('%', r'\%').replace(' - ', ' $-$ ')
            out.append(f"{tl} & {r['seeds_a']}/{r['seeds_b']} & {sd} & {sp} & {hd} & {hp} \\\\")
        return '\n'.join(out)

    def table(fams, label, caption):
        return r'''\begin{table*}[t]
\centering\small
\setlength{\tabcolsep}{4pt}
\resizebox{\textwidth}{!}{\begin{tabular}{@{}lccccc@{}}
\toprule
Contrast ($a - b$) & Seeds & Seen $\Delta$ [95\% CI] & $p_{\text{Holm}}$ & Held-out $\Delta$ [95\% CI] & $p_{\text{Holm}}$ \\
\midrule
''' + rows_for(fams) + r'''
\bottomrule
\end{tabular}}
\caption{''' + caption + r'''}
\label{''' + label + r'''}
\end{table*}
'''
    common = (f"Document-clustered paired bootstrap ({data['draws']:,} draws; differences in chance-normalised points). "
              "Intervals are pointwise and condition on the observed tasks and on seed-averaged correctness. "
              "Holm adjustment is applied within each family (RQ1; RQ2; RQ3; baselines) across both tracks. "
              f"The smallest attainable unadjusted $p$ is {2 / (data['draws'] + 1):.4f}.")
    (TAB / 'contrasts_rq1.tex').write_text(table({'RQ1'}, 'tab:contrasts_rq1',
        'Initialisation contrasts (contrastive child minus MLM parent) for every available pair, head and training-data '
        'fraction. ' + common), encoding='utf-8')
    (TAB / 'contrasts.tex').write_text(table({'RQ2', 'RQ3', 'LLM'}, 'tab:contrasts',
        'Conversion (RQ2), objective (RQ3) and baseline contrasts on S-PubMedBERT-MS-MARCO. ' + common), encoding='utf-8')


def main():
    runs_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else V2 / 'runs'
    df = load()
    objectives_tables(df, runs_dir)
    probe_outputs(runs_dir)
    gradnorm_figure(df, runs_dir)
    per_task_table(df)
    contrast_tables()
    if '--no-reliability' not in sys.argv:
        reliability_figure(runs_dir)
    print('wrote tables: objectives, objectives_full, probe, per_task, contrasts, contrasts_rq1; '
          'figures: probe, gradnorm, reliability')


if __name__ == '__main__':
    main()
