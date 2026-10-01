"""Generate paper numbers and compact contrast table from analysis outputs."""
import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]


def main():
    df=pd.read_csv(ROOT/'analysis/cascade.csv');df=df[~df.filtered]
    vals={}
    for name,suffix in [('S-PubMedBERT C','S'),('S-PubMedBERT C (CE)','SCE'),('Laya-large FT','L')]:
        d=df[(df.model==name)&(df.budget==20)]
        vals['Cascade'+suffix]=d.score.mean();vals['Alone'+suffix]=d.encoder.mean()
    vals['AloneGemma']=df.llm.iloc[0]
    (ROOT/'paper/tables/generated_numbers.tex').write_text(''.join('\\newcommand{\\'+k+'}{'+f'{v:.1f}'+'}\n' for k,v in vals.items()))
    # the contrast tables are written by paper/scripts/revision_tables.py
    print(vals)


if __name__=='__main__':main()
