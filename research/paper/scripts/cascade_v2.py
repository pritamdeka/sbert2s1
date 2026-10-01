"""Retrospective fixed-budget routing, not a validation-selected threshold policy."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analysis_common import ROOT, SEEN, HELD, predictions, temperature_for, macro_accuracy


def evaluate(run, llm, filtered=False):
    tp=json.loads((run/'temps.json').read_text());rows=[];c1=[];c2=[];conf=[]
    for task in SEEN+HELD:
        a,b=predictions(str(run),task,filtered),predictions(str(llm),task,filtered)
        if set(a)!=set(b):raise ValueError(f'Cascade item mismatch: {task}')
        for key in sorted(a):
            r=a[key];z=np.asarray(r['logits']);K=len(z)
            T=temperature_for(tp,r['qtype'],K)
            p=np.exp((z-z.max())/T);p/=p.sum()
            rows.append(dict(r,task=task));conf.append(p.max())
            y=np.argmax(r['target']);c1.append(z.argmax()==y);c2.append(np.argmax(b[key]['logits'])==y)
    c1,c2=np.asarray(c1),np.asarray(c2);order=np.argsort(conf,kind='stable')
    result=[]
    for budget in range(0,101,5):
        esc=np.zeros(len(rows),bool);esc[order[:round(len(rows)*budget/100)]]=True
        correct=np.where(esc,c2,c1)
        result.append(dict(budget=budget,actual_percent=100*esc.mean(),score=100*macro_accuracy(rows,correct),
                           n=len(rows),encoder=100*macro_accuracy(rows,c1),llm=100*macro_accuracy(rows,c2)))
    return result


def cascade(root):
    root=Path(root);llm=root/'llm/gemma4_31b';all_rows=[]
    for prefix,name in [('main/spubmedbert/C','S-PubMedBERT C'),('rq3c/ce','S-PubMedBERT C (CE)'),('laya_ft','Laya-large FT')]:
        for run in sorted((root/prefix).glob('s*')):
            if not (run/'temps.json').exists():continue
            for filtered in (False,True):
                for r in evaluate(run,llm,filtered):
                    all_rows.append(dict(r,model=name,seed=run.name,filtered=filtered))
    df=pd.DataFrame(all_rows);df.to_csv(ROOT/'analysis/cascade.csv',index=False)
    fig,ax=plt.subplots(figsize=(3.25,2.3))
    for name,color,ls in [('S-PubMedBERT C','#0072B2','--'),('S-PubMedBERT C (CE)','#009E73','-'),('Laya-large FT','#D55E00','-.')]:
        sub=df[(df.model==name)&(~df.filtered)]
        agg=sub.groupby('budget').score.agg(['mean','std'])
        ax.plot(agg.index,agg['mean'],label=name.replace('S-PubMedBERT C','S-PubMedBERT C, PG').replace('PG (CE)','CE'),color=color,ls=ls,lw=1.2)
        ax.fill_between(agg.index,agg['mean']-agg['std'].fillna(0),agg['mean']+agg['std'].fillna(0),color=color,alpha=.15)
    ax.axhline(df[~df.filtered].llm.iloc[0],ls=':',color='#777',label='Gemma alone')
    ax.set_xlabel('Retrospective escalation budget (%)');ax.set_ylabel('Macro chance-normalised accuracy')
    ax.legend(fontsize=6,frameon=False);fig.tight_layout(pad=.4)
    fig.savefig(ROOT/'paper/figures/cascade.pdf');plt.close(fig)
    print(df[df.budget==20].groupby(['model','filtered']).score.agg(['mean','std']).to_string())


if __name__=='__main__':cascade(ROOT/'results/s1_results_public/runs')
