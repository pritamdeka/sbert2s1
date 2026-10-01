#!/usr/bin/env bash
#SBATCH --job-name=s1-worker
#SBATCH --partition=k2-gpu-amd
#SBATCH --gres=gpu:mi300x:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=240G
#SBATCH --time=3-00:00:00
# Ten minutes' warning: the packer forwards TERM to every child, each checkpoints and is requeued.
#SBATCH --signal=B:TERM@600
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# One generic worker per GPU. It packs as many queued tasks onto this GPU as VRAM and slots allow,
# pulling from the shared queue, so 2-3 of these jobs share the whole campaign.
set -euo pipefail
source hpc/env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Absolute end of this allocation, so the packer never starts a task it cannot finish.
left="$(squeue -h -j "$SLURM_JOB_ID" -o %L)"
secs=$(python - "$left" <<'PY'
import sys
s = sys.argv[1].strip()
d, _, hms = s.rpartition('-')
parts = [int(x) for x in hms.split(':')]
while len(parts) < 3:
    parts.insert(0, 0)
print(int(d or 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2])
PY
)
export S1_JOB_END=$(( $(date +%s) + secs ))
echo "job $SLURM_JOB_ID on $(hostname), ends $(date -d @"$S1_JOB_END")"
python - <<'PY'
import torch
print('torch', torch.__version__, 'HIP', torch.version.hip, 'GPU', torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
PY
python -m s1.queue requeue-stale || true
child=0
forward() { [ "$child" -gt 0 ] && kill -TERM "$child" 2>/dev/null || true; }
trap forward TERM
python -m s1.queue pack --vram-gb "${S1_GPU_VRAM_GB:-176}" --max-slots "${S1_MAX_SLOTS:-6}" --cpus "${SLURM_CPUS_PER_TASK:-12}" &
child=$!
rc=0
wait "$child" || rc=$?
if [ $rc -gt 128 ] && kill -0 "$child" 2>/dev/null; then
  rc=0; wait "$child" || rc=$?
fi
python -m s1.queue status || true
exit $rc
