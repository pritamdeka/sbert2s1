#!/usr/bin/env bash
# Stages of the campaign:
#   bash scripts/submit.sh smoke        # 1 GPU, 2 h: smoke grid in queue_smoke/ (C, PFR, Z, Laya; forced TERM->resume)
#   bash scripts/submit.sh calibrate    # 1 GPU, 1.5 h: packing calibration -> PACKING_CALIBRATION.md
#   bash scripts/submit.sh full [N]     # enqueue the v1 grid, then N (default 3) packed 3-day workers
#   bash scripts/submit.sh more [N]     # N more workers on the existing queue (resubmission after walltime)
set -euo pipefail
cd "${S1_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
export S1_ROOT="$PWD"
mkdir -p logs
[ -f configs/models.lock.json ] || { echo 'configs/models.lock.json missing: run scripts/stage_models.sh first' >&2; exit 2; }
[ -f prepared/manifest.json ] || { echo 'prepared/ missing: unzip the data bundle first' >&2; exit 2; }
stage="${1:-}"
case "$stage" in
  smoke)
    export S1_QUEUE="$S1_ROOT/queue_smoke"
    ( source hpc/env.sh >/dev/null; python scripts/enqueue_v1.py --smoke )
    sbatch --export=ALL --time=02:00:00 --job-name=s1-smoke slurm/worker.slurm ;;
  calibrate)
    sbatch --export=ALL --time=01:30:00 --job-name=s1-pack --wrap "source hpc/env.sh; export HF_HUB_OFFLINE=1; python scripts/packing_calibration.py --minutes 8" \
      --partition=k2-gpu-amd --gres=gpu:mi300x:1 --cpus-per-task=12 --mem=240G --output=logs/%x-%j.out ;;
  full)
    ( source hpc/env.sh >/dev/null; python scripts/enqueue_v1.py )
    n="${2:-3}"
    for i in $(seq 1 "$n"); do sbatch --export=ALL slurm/worker.slurm; done ;;
  more)
    n="${2:-3}"
    for i in $(seq 1 "$n"); do sbatch --export=ALL slurm/worker.slurm; done ;;
  *) echo 'usage: bash scripts/submit.sh smoke|calibrate|full [N]|more [N]' >&2; exit 2 ;;
esac
