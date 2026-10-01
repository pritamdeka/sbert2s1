#!/usr/bin/env bash
# Run on a node with outbound Hugging Face access (login node), once, before any compute job.
# Pins each model's Hub commit in configs/models.lock.json, then downloads it into the shared cache.
# Qwen3.8-27B and Gemma-4-31B may already be cached; staging them only records the pin.
set -euo pipefail
source hpc/env.sh
if [[ $# -eq 0 ]]; then set -- all; fi
python -m s1.models_io --stage "$@"
