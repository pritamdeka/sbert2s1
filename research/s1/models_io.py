"""Pinned model specs: configs/models.json + configs/models.lock.json -> local snapshot paths.

Models are downloaded from the Hugging Face Hub on first use. Optionally, `python -m s1.models_io --stage slug ...`
records the Hub commit SHA in configs/models.lock.json before downloading, so later (possibly offline) runs
load exactly that commit.
"""
import argparse
import os
from pathlib import Path

from .common import ROOT, atomic_json, read_json

MODELS = Path(os.environ.get('S1_MODELS', ROOT / 'configs' / 'models.json'))
LOCK = MODELS.with_name(MODELS.stem + '.lock.json')


def specs():
    models = read_json(MODELS)
    lock = read_json(LOCK) if LOCK.exists() else {}
    for slug, m in models.items():
        if slug in lock:
            m['revision'] = lock[slug]['revision']
    return models


def resolve(spec):
    """Local directory for a spec: an explicit local 'path', else the pinned HF snapshot."""
    if isinstance(spec, str):
        spec = specs()[spec]
    if spec.get('local_path'):
        return spec['local_path']
    from huggingface_hub import snapshot_download
    offline = os.environ.get('HF_HUB_OFFLINE') == '1'
    return snapshot_download(spec['model_id'], revision=spec.get('revision', 'main'),
                             local_files_only=offline,
                             allow_patterns=spec.get('allow_patterns'),
                             ignore_patterns=['*.gguf', '*.msgpack', '*.h5', '*.ot', 'onnx/*', 'openvino/*'])


def spec_with_path(slug):
    s = dict(specs()[slug])
    base = resolve(s)
    s['path'] = os.path.join(base, s['subfolder']) if s.get('subfolder') else base
    return s


def stage(slugs):
    from huggingface_hub import HfApi
    models = read_json(MODELS)
    lock = read_json(LOCK) if LOCK.exists() else {}
    for slug in slugs:
        if slug not in lock:
            info = HfApi().model_info(models[slug]['model_id'], revision=models[slug].get('revision', 'main'),
                                      token=os.getenv('HF_TOKEN'))
            lock[slug] = {'model_id': models[slug]['model_id'], 'revision': info.sha}
            atomic_json(LOCK, lock)
        spec = dict(models[slug], revision=lock[slug]['revision'])
        print(slug, spec['model_id'], spec['revision'], resolve(spec), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', nargs='+', required=True)
    a = p.parse_args()
    slugs = list(read_json(MODELS)) if a.stage == ['all'] else a.stage
    stage(slugs)


if __name__ == '__main__':
    main()
