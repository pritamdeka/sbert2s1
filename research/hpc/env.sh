#!/usr/bin/env bash
# Cluster environment for the s1 campaign (Kelvin2 / NI-HPC: ROCm PyTorch module + a clean venv).
# The runtime needs only torch, transformers, safetensors, huggingface_hub and numpy, which the
# environment in $S1_VENV must provide (see research/requirements.txt). Adapt the module line to your cluster.
set -euo pipefail
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  IFS=: read -r -a s1_path_entries <<< "$PATH"
  s1_clean_path=()
  for s1_entry in "${s1_path_entries[@]}"; do
    [[ "$s1_entry" == "$VIRTUAL_ENV/bin" ]] || s1_clean_path+=("$s1_entry")
  done
  PATH="$(IFS=:; echo "${s1_clean_path[*]}")"
  export PATH
fi
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
hash -r
module purge
module load apps/pytorch_rocm/2.8.0.20250313/bin
export S1_ROOT="${S1_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
export S1_VENV="${S1_VENV:?set S1_VENV to your Python environment}"
source "$S1_VENV/bin/activate"
hash -r
cd "$S1_ROOT"
export PYTHONPATH="$S1_ROOT"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
# Hugging Face cache (the LLM baselines need about 120 GB).
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export S1_QUEUE="${S1_QUEUE:-$S1_ROOT/queue}"
mkdir -p logs/tasks "$HF_HUB_CACHE" cache
export HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.cache/huggingface/token}"
if [[ -r "$HF_TOKEN_FILE" ]]; then
  export HF_TOKEN="$(<"$HF_TOKEN_FILE")"
fi
