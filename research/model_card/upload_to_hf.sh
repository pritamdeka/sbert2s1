#!/usr/bin/env bash
# Upload an exported sbert2s1 model to the Hugging Face Hub (needs `huggingface-cli login`).
#   bash research/model_card/upload_to_hf.sh exports/S1-PubMedBERT pritamdeka/S1-PubMedBERT [public]
# Default is a PRIVATE repo; pass "public" as the 3rd argument only after you have tested it.
set -euo pipefail
src="${1:?export directory}"; repo="${2:?hub repo id, e.g. pritamdeka/S1-PubMedBERT}"; vis="${3:-private}"
here="$(cd "$(dirname "$0")" && pwd)"
for f in s1_config.json model.safetensors encoder/config.json tokenizer/tokenizer_config.json; do
  [ -e "$src/$f" ] || { echo "missing $src/$f (run s1.predict.export first)"; exit 2; }
done
# model card, a single-file copy of the pip package's inference code, and its requirements
cp "$here/README.md" "$here/requirements.txt" "$src/"
cp "$here/../../src/sbert2s1/_inference.py" "$src/sbert2s1.py"
python - "$src" "$repo" "$vis" <<'PY'
import json, sys
from huggingface_hub import HfApi
src, repo, vis = sys.argv[1:4]
cfg = json.load(open(f'{src}/s1_config.json'))
print('exported run: arch', cfg['arch'], '| objective', cfg['objective'], '| seed', cfg['seed'],
      '| train tasks', cfg['train_tasks'])
clinical = {'mednli', 'mimic_deid', 'mimic_trialq'} & set(cfg['train_tasks'])
if clinical:
    sys.exit(f'REFUSING: trained on credentialed data {clinical}; only public-data models may be released')
api = HfApi()
print('logged in as', api.whoami()['name'])
api.create_repo(repo, repo_type='model', private=(vis != 'public'), exist_ok=True)
api.upload_folder(folder_path=src, repo_id=repo, repo_type='model',
                  commit_message='sbert2s1 release: model, inference code and model card')
print(f'uploaded -> https://huggingface.co/{repo}  ({"public" if vis == "public" else "private"})')
PY
