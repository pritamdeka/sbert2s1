"""Figures for the paper from metrics_long.csv and saved logits (public tasks only).

    python paper/scripts/figures.py runs
Writes paper/figures/learning_curves.pdf and paper/figures/cascade.pdf.
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
FIG = HERE.parent / 'figures'
SEEN = ['pubmedqa', 'scifact', 'healthver', 'ddi', 'hoc', 'ade', 'druglib', 'medline_s1']
HELD = ['pubhealth', 'biosses', 'mtsamples']
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
BLUE, ORANGE, GREY = '#2a6fdb', '#e8762c', '#888888'


def seen_acc(df):
    d = df[(df.tag == 'final') & (df.variant == 'std') & (df.track == 'seen')]
    return d.groupby(['run', 'family', 'encoder', 'arch', 'frac', 'seed', 'task']).raw_acc_norm.mean() \
            .groupby(['run', 'family', 'encoder', 'arch', 'frac', 'seed']).mean().reset_index()


def learning_curves(df):
    a = seen_acc(df)
    a = a[a.family.isin(['main', 'lc'])]
    pairs = [('pubmedbert', 'spubmedbert', 'PubMedBERT $\\rightarrow$\nS-PubMedBERT'),
             ('pubmedbert', 'medcpt_q', 'PubMedBERT $\\rightarrow$\nMedCPT'),
             ('mpnet', 'allmpnet', 'MPNet $\\rightarrow$\nall-MPNet'),
             ('modernbert', 'mbembed', 'ModernBERT $\\rightarrow$\nNomic-embed'),
             ('modernbert', 'gtemb', 'ModernBERT $\\rightarrow$\nGTE-ModernBERT'),
             ('bcmb', 'bcmbembed', 'BioClinical-MB $\\rightarrow$\nits embedder')]
    fig, axes = plt.subplots(1, 6, figsize=(7.0, 2.15), sharey=True)
    for ax, (mlm, con, title) in zip(axes, pairs):
        for enc, color, lab in ((mlm, GREY, 'MLM'), (con, BLUE, 'contrastive')):
            for arch, ls in (('C', '-'), ('PFR', '--')):
                x = a[(a.encoder == enc) & (a.arch == arch)].groupby('frac').raw_acc_norm.agg(['mean', 'std']).sort_index()
                ax.errorbar(x.index, x['mean'] * 100, yerr=x['std'].fillna(0) * 100, color=color, ls=ls, marker='o', ms=2.5,
                            lw=1, capsize=1.5, label=f'{lab}, {arch}')
        ax.set_xscale('log')
        ax.set_xticks([0.02, 0.1, 1.0])
        ax.set_xticklabels(['2%', '10%', '100%'])
        ax.set_title(title, fontsize=6.5)
        ax.set_xlabel('training data')
    axes[0].set_ylabel('seen acc$_{\\mathrm{cn}}$ ($\\times$100)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=6.5, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(pad=0.3, rect=(0, 0.09, 1, 1))
    fig.savefig(FIG / 'learning_curves.pdf')
    print('wrote learning_curves.pdf')


def load_preds(run_dir, task):
    f = run_dir / 'preds' / 'final' / f'{task}.test.jsonl.gz'
    out = {}
    for line in gzip.open(f, 'rt'):
        r = json.loads(line)
        z, t = np.array(r['logits']), np.array(r['target'])
        out[(r['id'], r['qid'])] = (z, int(t.argmax()), len(t))
    return out


def temps(run_dir):
    return json.loads((run_dir / 'temps.json').read_text())


def T_for(tp, qtype, k):
    from analysis_common import temperature_for
    return temperature_for(tp, qtype, k)



def main():
    root = Path(sys.argv[1])
    FIG.mkdir(exist_ok=True)
    df = pd.read_csv(HERE / 'metrics_long.csv')
    learning_curves(df)
    from cascade_v2 import cascade
    cascade(root)


if __name__ == '__main__':
    main()
