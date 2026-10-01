"""Small shared helpers: paths, atomic writes, hashing, JSONL."""
import hashlib
import json
import os
import random
from pathlib import Path

ROOT = Path(os.environ.get('S1_ROOT', Path(__file__).resolve().parents[1]))
QTYPES = {'choice': 0, 'score': 1, 'noul': 2}          # identical to laya.common.QTYPES
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
    tmp.write_text(text, encoding='utf-8')
    os.replace(tmp, path)


def atomic_json(path, obj):
    atomic_write(path, json.dumps(obj, indent=1, sort_keys=True, default=str))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(obj, n=12):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:n]


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
    with open(tmp, 'w', encoding='utf-8', newline='\n') as f:   # same bytes on every OS
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_vram_cap():
    """Optionally cap this process's GPU memory at S1_VRAM_GB gigabytes (unset: no cap). Useful when
    several runs share one GPU, so that one greedy process cannot take the whole device."""
    import torch
    gb = os.environ.get('S1_VRAM_GB')
    if not gb or not torch.cuda.is_available():
        return None
    total = torch.cuda.get_device_properties(0).total_memory
    frac = min(0.98, float(gb) * 2 ** 30 / total)
    torch.cuda.set_per_process_memory_fraction(frac, 0)
    return frac


def temp_bucket(qtype, k):
    """Same buckets as laya.common.temp_bucket, so fitted temperatures export unchanged."""
    size = '2' if k <= 2 else '3-5' if k <= 5 else '6-10' if k <= 10 else '11+'
    return f'{QTYPE_NAMES[int(qtype)]}:{size}'
