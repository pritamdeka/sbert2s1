"""Exact-input exclusion sensitivity using existing checkpoints and temperatures."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from analysis_common import ROOT, SEEN, predictions, macro_accuracy, audit


def main():
    root=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'runs';output=[]
    selectors=[('S-PubMedBERT C','main/spubmedbert/C/s*'),('S-PubMedBERT PFR (PG)','main/spubmedbert/PFR/s*'),
               ('S-PubMedBERT PFR (CE)','rq3/ce/s*'),('S-PubMedBERT PFR (Reparam)','rq3/reparam/s*'),
               ('BioClinical-ModernBERT C','main/bcmb/C/s*'),('Laya-large FT','laya_ft/s*'),('Gemma-4-31B','llm/gemma4_31b')]
    for name,pattern in selectors:
        for run in sorted(root.glob(pattern)):
            if not (run/'metrics_final.json').exists():continue
            for filtered in (False,True):
                rows=[];correct=[]
                for task in SEEN:
                    for r in predictions(str(run),task,filtered).values():
                        rows.append(dict(r,task=task));correct.append(np.argmax(r['logits'])==np.argmax(r['target']))
                output.append(dict(model=name,run=str(run.relative_to(root)),filtered=filtered,n=len(rows),accuracy=100*macro_accuracy(rows,correct)))
    df=pd.DataFrame(output);df.to_csv(ROOT/'analysis/sensitivity.csv',index=False)
    body=[]
    for name,_ in selectors:
        d=df[df.model==name];a=d[~d.filtered].accuracy.mean();b=d[d.filtered].accuracy.mean()
        body.append(f'{name} & {a:.2f} & {b:.2f} & {b-a:+.2f} '+r'\\')
    tex=r'''\begin{table}[t]\centering\small
\resizebox{\columnwidth}{!}{\begin{tabular}{@{}lrrr@{}}\toprule
Model & Historical & Filtered & Difference \\
\midrule
'''+ '\n'.join(body)+r'''
\bottomrule\end{tabular}}
\caption{Seen-task accuracy sensitivity to removal of exact test inputs appearing in public
training/calibration data, and repeated test inputs. Seed means, in chance-normalised points.
Checkpoints and temperatures are unchanged; this does not repair calibration contamination.}
\label{tab:sensitivity}\end{table}
'''
    (ROOT/'paper/tables/sensitivity.tex').write_text(tex)
    print('Wrote sensitivity table',flush=True)


if __name__=='__main__':main()
