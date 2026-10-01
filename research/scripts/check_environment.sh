#!/usr/bin/env bash
# Verify the venv has everything s1bio needs. Installs nothing.
set -euo pipefail
source hpc/env.sh
python - <<'PY'
import importlib, torch, transformers, safetensors, huggingface_hub, numpy
print('torch', torch.__version__, 'HIP', torch.version.hip, 'GPU visible', torch.cuda.is_available())
print('transformers', transformers.__version__, '| safetensors', safetensors.__version__,
      '| huggingface_hub', huggingface_hub.__version__, '| numpy', numpy.__version__)
for m in ('s1.common', 's1.schema', 's1.data', 's1.model', 's1.objectives', 's1.metrics', 's1.calibrate',
          's1.evaluate', 's1.train', 's1.queue', 's1.llm_baseline', 's1.models_io', 's1.latency'):
    importlib.import_module(m)
print('all s1 modules import; nothing was installed or changed')
PY
