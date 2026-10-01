"""Generate every LaTeX table of the paper from data manifests and run metrics. Nothing is typed by hand.

    python paper/scripts/make_tables.py                       # data table only (manifests)
    python paper/scripts/make_tables.py --runs runs           # + result tables from metrics_final.json files

Clinical counts come from prepared/manifest_clinical.json (counts only).
Missing inputs produce tables whose cells read "--" and a \\tbd marker, never invented numbers.
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
PROJECT = HERE.parent
TABLES = HERE / 'tables'

TASKS = [  # (task, display name, track, family, primitive)
    ('pubmedqa', 'PubMedQA-L', 'G', 'Evidence QA', 'choice'),
    ('pubmedqa_art', 'PubMedQA-A$^\\S$', 'G', 'Evidence QA', 'noul'),
    ('scifact', 'SciFact', 'G', 'Claim verif.', 'choice'),
    ('healthver', 'HealthVer', 'G', 'Claim verif.', 'choice'),
    ('pubhealth', 'PUBHEALTH$^\\dagger$', 'G', 'Claim verif.', 'choice'),
    ('ddi', 'DDI-2013', 'G', 'Relation', 'choice'),
    ('hoc', 'HoC', 'G', 'Doc.\\ class.', 'noul'),
    ('ade', 'ADE', 'G', 'Adv.\\ events', 'noul'),
    ('druglib', 'Druglib', 'G', 'Review', 'score'),
    ('biosses', 'BIOSSES$^\\dagger$', 'G', 'Similarity', 'score'),
    ('mtsamples', 'MTSamples$^\\ddagger$', 'G', 'Routing', 'choice'),
    ('medline_s1', '\\medline{}', 'G', 'Indexing', 'mixed'),
    ('medqa', 'MedQA-USMLE', 'K', 'Exam', 'choice'),
    ('medmcqa', 'MedMCQA', 'K', 'Exam', 'choice'),
    ('mmlu_med', 'MMLU-medical$^\\dagger$', 'K', 'Exam', 'choice'),
    ('typed_decisions', 'typed-decisions$^\\dagger$', 'Gen', 'General', 'mixed'),
    ('mednli', 'MedNLI', 'C', 'NLI', 'choice'),
    ('mimic_trialq', 'MIMIC-III trials$^\\dagger$', 'C', 'Long notes', 'mixed'),
    ('mimic_deid', 'MIMIC-IV names', 'C', 'Privacy', 'noul'),
]


def fmt_int(n):
    return '--' if n is None else f'{n:,}'


TRACK_NAMES = [('G', 'Grounded'), ('K', 'Knowledge'), ('Gen', 'General domain'),
               ('C', 'Clinical (PhysioNet credentialed)')]


def data_table():
    """Single-column table, grouped by track (no track column) so that it fits \columnwidth."""
    man = {}
    for f in ('manifest.json', 'manifest_clinical.json'):
        p = PROJECT / 'prepared' / f
        if p.exists():
            man.update(json.loads(p.read_text()))
    groups = {code: [] for code, _ in TRACK_NAMES}
    tot = {'train': 0, 'test': 0}
    for task, name, track, fam, prim in TASKS:
        def d(split):
            v = man.get(f'{task}/{split}')
            return v['decisions'] if v else None
        tr, te = d('train'), d('test')
        if task == 'mimic_deid' and d('polysemy'):
            te = (te or 0) + d('polysemy')
        for k, v in (('train', tr), ('test', te)):
            tot[k] += v or 0
        missing = task in ('mednli', 'mimic_trialq', 'mimic_deid') and te is None
        te_s = chr(92) + 'tbd{}' if missing else fmt_int(te)
        groups[track].append(f'{name} & {fam} & \\texttt{{{prim}}} & {fmt_int(tr)} & {te_s} \\\\')
    body = []
    for code, title in TRACK_NAMES:
        if groups[code]:
            body.append(f'\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\')
            body.extend(groups[code])
    body = '\n'.join(body)
    return rf"""\begin{{table}}[t]
\centering\footnotesize
\setlength{{\tabcolsep}}{{2.5pt}}
\begin{{tabular}}{{@{{}}lllrr@{{}}}}
\toprule
Task & Family & Type & \#Train & \#Test \\
\midrule
{body}
\midrule
Total & & & {tot['train']:,} & {tot['test']:,} \\
\bottomrule
\end{{tabular}}
\caption{{\bench{{}} and \medline{{}}, in decisions (a state may carry several questions).
$\dagger$~test only (never trained on); $\ddagger$~held-out task family; $\S$~training only.
Historical calibration slices are carved from the training data before training and
are not counted. Clinical training counts apply only to the +clinical variant. Counts describe the original splits; the integrity audit reports exclusions separately.}}
\label{{tab:data}}
\end{{table}}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs', default=None)
    a = ap.parse_args()
    TABLES.mkdir(exist_ok=True)
    (TABLES / 'data_stats.tex').write_text(data_table(), encoding='utf-8')
    print('wrote tables/data_stats.tex')
    if a.runs:
        from results_tables import write_all            # added once full-run metrics exist
        write_all(Path(a.runs), TABLES)


if __name__ == '__main__':
    main()
