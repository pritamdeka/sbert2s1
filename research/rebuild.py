"""Reproduce v2 analyses and paper. Use --reuse-analysis to rebuild tables/PDF only."""
import argparse
import os
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reuse-analysis',action='store_true')
    ap.add_argument('--no-pdf',action='store_true');args=ap.parse_args()
    env=dict(os.environ);env['PYTHONPATH']=str(ROOT/'tools/python_packages')+os.pathsep+env.get('PYTHONPATH','')
    env['TECTONIC_CACHE_DIR']=str(ROOT/'tools/texcache')
    def py(script,*rest):subprocess.run([sys.executable,str(ROOT/script),*map(str,rest)],cwd=ROOT,env=env,check=True)
    runs=ROOT/'results/s1_results_public/runs'
    if not args.reuse_analysis:
        py('analysis/audit_data.py')
        py('paper/scripts/collect.py',runs.parent)
        py('paper/scripts/bootstrap.py',runs,'--draws','10000')
        py('paper/scripts/sensitivity.py')
        py('paper/scripts/figures.py',runs)
    py('analysis/summarize.py')
    py('paper/scripts/make_tables.py')
    code="import sys;sys.path.insert(0,'paper/scripts');from results_tables import write_everything;write_everything('results/s1_results_public/runs','paper/tables')"
    subprocess.run([sys.executable,'-c',code],cwd=ROOT,env=env,check=True)
    py('paper/scripts/revision_tables.py',runs)
    py('paper/scripts/fix_minus.py')
    if not args.no_pdf:
        compiler=ROOT/'tools/tectonic.exe'
        for tex in ('main.tex','main_preprint.tex'):
            subprocess.run([str(compiler),'--untrusted','--keep-logs',tex],cwd=ROOT/'paper',env=env,check=True)


if __name__=='__main__':main()
