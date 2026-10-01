"""Temperature lookup shared by evaluation, prediction and offline analysis (patch 8).

Fixes two problems of the earlier lookups:
  * offline analysis matched buckets by option count only and ignored the question type;
  * a numeric type passed as a NumPy/PyTorch integer was not recognised and silently fell back to T=1.0.
Valid types are the names 'choice'/'score'/'noul' or the integer codes 0/1/2 as any integer scalar
(Python int, NumPy integer, 0-d integer tensor). Anything else raises instead of silently using T=1.0.
Lookup order for a valid type: bucket 'type:size' -> per-type temperature -> 1.0.
"""
import operator

QTYPE_NAMES = {0: 'choice', 1: 'score', 2: 'noul'}


def qtype_name(qtype):
    if isinstance(qtype, str):
        if qtype not in QTYPE_NAMES.values():
            raise ValueError(f'unknown question type {qtype!r}')
        return qtype
    if getattr(qtype, 'ndim', 0) != 0:
        raise TypeError('question type must be a scalar')
    scalar = qtype.item() if hasattr(qtype, 'item') else qtype
    if isinstance(scalar, bool):
        raise TypeError('question type must be an integer code, not a bool')
    code = operator.index(scalar)          # rejects floats such as 1.9
    if code not in QTYPE_NAMES:
        raise ValueError(f'unknown question type code {code}')
    return QTYPE_NAMES[code]


def size_bucket(k):
    k = int(k)
    return '2' if k <= 2 else '3-5' if k <= 5 else '6-10' if k <= 10 else '11+'


def temperature_for(temps, qtype, k):
    name = qtype_name(qtype)
    temps = temps or {}
    bucket = f'{name}:{size_bucket(k)}'
    if bucket in temps.get('bucket', {}):
        return temps['bucket'][bucket]
    return temps.get('type', {}).get(name, 1.0)
