"""Rebuild the paper's tables and figures from a directory of finished runs.

    python rebuild.py                      # reads runs/ (written by experiments.py), writes paper/tables and paper/figures
    python rebuild.py --runs path/to/runs  # any directory with the same run layout
    python rebuild.py --skip-bootstrap     # reuse analysis/contrasts_cache_*.jsonl instead of resampling

Steps: data audit, metrics collection (paper/scripts/metrics_long.csv), paired bootstrap contrasts,
sensitivity analysis, figures, then every LaTeX table. Runs that are missing are skipped or shown as '--'.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', default=str(ROOT / 'runs'))
    ap.add_argument('--skip-bootstrap', action='store_true')
    ap.add_argument('--draws', type=int, default=10000)
    a = ap.parse_args()
    runs = Path(a.runs).resolve()

    def py(script, *rest):
        subprocess.run([sys.executable, str(ROOT / script), *map(str, rest)], cwd=ROOT, check=True)

    py('analysis/audit_data.py')
    py('paper/scripts/collect.py', runs.parent)
    if not a.skip_bootstrap:
        py('paper/scripts/bootstrap.py', runs, '--draws', a.draws)
    py('paper/scripts/sensitivity.py', runs)
    py('paper/scripts/figures.py', runs)
    py('analysis/summarize.py')
    py('paper/scripts/make_tables.py')
    code = ("import sys; sys.path.insert(0, 'paper/scripts'); from results_tables import write_everything; "
            f"write_everything({str(runs)!r}, 'paper/tables')")
    subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True)
    py('paper/scripts/revision_tables.py', runs)
    py('paper/scripts/fix_minus.py')
    print('tables in paper/tables, figures in paper/figures')


if __name__ == '__main__':
    main()
