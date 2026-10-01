#!/usr/bin/env bash
# Patch 8 helper (revision controls). Run from the campaign root (the s1bio_hpc directory on scratch).
#   bash scripts/patch8.sh check            # pre-flight: hashes, objectives, temperature lookup, inputs (CPU, ~1 min)
#   bash scripts/patch8.sh test             # end-to-end tiny CPU test of the new code (~5 min)
#   bash scripts/patch8.sh enqueue [P]      # enqueue priorities <= P (default 1; 0 = must-have only; 2 = + LC seeds)
#   bash scripts/patch8.sh submit [N] [T]   # N packed GPU workers (default 2) with walltime T (default 12:00:00)
#   bash scripts/patch8.sh status           # queue status + patch-8 run progress
#   bash scripts/patch8.sh collect          # tar the new results (no weights, no clinical per-item files)
set -euo pipefail
cd "${S1_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
export S1_ROOT="$PWD"
mkdir -p logs
cmd="${1:-}"
case "$cmd" in
  check)
    ( source hpc/env.sh >/dev/null; python scripts/check_patch8.py ) ;;
  test)
    ( source hpc/env.sh >/dev/null; python tests/test_patch8.py ) ;;
  enqueue)
    p="${2:-1}"
    ( source hpc/env.sh >/dev/null; python scripts/enqueue_patch8.py --dry-run --max-priority "$p" \
      && python scripts/enqueue_patch8.py --max-priority "$p" ) ;;
  submit)
    n="${2:-2}"; t="${3:-12:00:00}"
    for i in $(seq 1 "$n"); do sbatch --export=ALL --time="$t" --job-name=s1-p8 slurm/worker.slurm; done ;;
  status)
    ( source hpc/env.sh >/dev/null; python -m s1.queue status ) || true
    echo; echo 'patch-8 runs finished (DONE files):'
    for d in runs/rq3/pg_loo/s* runs/rq3c/*/s* runs/probe/gradvar; do
      [ -e "$d/DONE" ] && echo "  done     $d" || { [ -d "$d" ] && echo "  pending  $d"; } || true
    done ;;
  collect)
    shopt -s nullglob
    dirs=(runs/rq3/pg_loo runs/rq3c runs/probe)
    for e in modernbert mbembed gtemb bcmb bcmbembed; do
      for a in C PFR; do dirs+=(runs/lc/$e/$a/f0.02 runs/lc/$e/$a/f0.1/s1 runs/lc/$e/$a/f0.1/s2); done
    done
    keep=(); for d in "${dirs[@]}"; do [ -e "$d" ] && keep+=("$d"); done
    tar czf s1_results_patch8.tgz \
      --exclude='*/model.pt' --exclude='*/model.pt.tmp' --exclude='*/ckpt' \
      --exclude='*/preds/*/mednli*' --exclude='*/preds/*/mimic_*' \
      "${keep[@]}" logs/tasks/*rq3* logs/tasks/*probe* logs/tasks/P2_lc__* 2>/dev/null || true
    ls -lh s1_results_patch8.tgz
    echo 'Clinical per-item prediction files are excluded; clinical aggregate metrics are in metrics_final.json.' ;;
  *)
    sed -n '2,9p' "$0"; exit 2 ;;
esac
