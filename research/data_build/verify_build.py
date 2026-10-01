"""Check a local build of BioDecide and MEDLINE-S1 against the paper's build, record by record.

    python data_build/verify_build.py --prepared prepared

`docs/prepared_manifest.json` holds the SHA-256 of every split file and `docs/biodecide_records.tsv.gz`
one line per record: its id, a hash of the whole record, a hash of its state text, and the source
identifiers the record keeps (for example the PMID). If a split file differs, this script says which
records differ and whether the text, the labels or the split membership changed, which is usually
enough to see what a source dataset changed upstream.
"""
import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def record_hashes(rec):
    full = hashlib.sha256(json.dumps(rec, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()[:12]
    st = rec['state'] if isinstance(rec['state'], str) else json.dumps(rec['state'], ensure_ascii=False, sort_keys=True)
    state = hashlib.sha256(st.encode('utf-8')).hexdigest()[:12]
    return full, state


def file_sha256_lf(path):
    """Hash of the file with \n line endings, so that builds on Windows and Linux compare equal."""
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--prepared', default=str(HERE / 'prepared'))
    ap.add_argument('--records', default=str(HERE / 'docs' / 'biodecide_records.tsv.gz'))
    ap.add_argument('--manifest', default=str(HERE / 'docs' / 'prepared_manifest.json'))
    a = ap.parse_args()
    prepared = Path(a.prepared)
    manifest = json.loads(Path(a.manifest).read_text())

    ref = defaultdict(dict)                          # 'task/split' -> id -> (full, state, source)
    with gzip.open(a.records, 'rt', encoding='utf-8') as f:
        next(f)
        for line in f:
            rid, full, state, source = line.rstrip('\n').split('\t')
            task, split, _ = rid.split(':')
            ref[f'{task}/{split}'][rid] = (full, state, source)

    bad = 0
    for key in sorted(ref):
        path = prepared / f'{key}.jsonl'
        if not path.exists():
            print(f'{key:24s} not built')
            continue
        if key in manifest and file_sha256_lf(path) == manifest[key]['sha256_lf']:
            print(f'{key:24s} identical ({len(ref[key])} records)')
            continue
        bad += 1
        mine = {}
        with open(path, encoding='utf-8') as f:
            for line in f:
                rec = json.loads(line)
                mine[rec['id']] = record_hashes(rec)
        same = text = labels = 0
        for rid, (full, state, _) in ref[key].items():
            if rid not in mine:
                continue
            if mine[rid][0] == full:
                same += 1
            elif mine[rid][1] != state:
                text += 1
            else:
                labels += 1
        missing = len(set(ref[key]) - set(mine))
        extra = len(set(mine) - set(ref[key]))
        print(f'{key:24s} DIFFERS: {same} identical, {text} different text, {labels} same text but different '
              f'questions or labels, {missing} missing, {extra} extra')
    print('all splits match the paper build' if not bad else f'{bad} split(s) differ from the paper build')
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
