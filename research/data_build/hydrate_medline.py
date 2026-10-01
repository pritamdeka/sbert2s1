"""Restore the abstracts of the MEDLINE-S1 release and write the full records used in the paper.

The released files hold PMIDs, questions and gold labels, but not the titles and abstracts. This script
fetches them and rebuilds `<out>/<split>.jsonl` in the exact format of `build_medline.py`, which is what
the training and evaluation code reads (`prepared/medline_s1/`).

    # from PubMed E-utilities (needs internet; an NCBI API key makes it faster)
    python hydrate_medline.py --data data --out prepared/medline_s1 --email you@example.org

    # or from the PubMed baseline files the paper used (listed in manifest.json), if you have them
    python hydrate_medline.py --data data --out prepared/medline_s1 --baseline-dir cache/pubmed

Every restored state is checked against the SHA-256 in the release. With the baseline files all states
match and each split file matches the paper's file hash. E-utilities returns the current version of each
citation, so a small number of titles or abstracts may have been revised since the paper's build; they
are reported, kept with the current text by default (or dropped with --drop-changed).

Requires Python 3.8+ and lxml.
"""
import argparse
import gzip
import hashlib
import io
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

EUTILS = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
SPLITS = ('train', 'dev', 'calib', 'test')


def parse(xml_bytes):
    """Yield (pmid, state) exactly as build_medline.py builds the state: title, a space, the abstract."""
    from lxml import etree
    for _, el in etree.iterparse(io.BytesIO(xml_bytes), tag='PubmedArticle'):
        try:
            mc = el.find('MedlineCitation')
            art = mc.find('Article')
            pmid = int(mc.findtext('PMID'))
            abstract = ' '.join(''.join(a.itertext()).strip() for a in art.findall('Abstract/AbstractText'))
            title = ''.join(art.find('ArticleTitle').itertext()).strip() if art.find('ArticleTitle') is not None else ''
            yield pmid, (title + ' ' + abstract).strip()
        except Exception:
            pass
        finally:
            el.clear()


def from_eutils(pmids, email, api_key, batch=200):
    found = {}
    pause = 0.11 if api_key else 0.34                     # NCBI limits: 10 requests/s with a key, 3 without
    todo = sorted(pmids)
    for i in range(0, len(todo), batch):
        ids = todo[i:i + batch]
        body = {'db': 'pubmed', 'retmode': 'xml', 'id': ','.join(map(str, ids)), 'tool': 'sbert2s1-hydrate'}
        if email:
            body['email'] = email
        if api_key:
            body['api_key'] = api_key
        data = urllib.parse.urlencode(body).encode()
        for attempt in range(5):
            try:
                with urllib.request.urlopen(urllib.request.Request(EUTILS, data=data), timeout=120) as r:
                    xml = r.read()
                break
            except Exception as e:                          # transient 429/5xx: back off and retry
                if attempt == 4:
                    raise SystemExit(f'E-utilities failed for batch starting at {ids[0]}: {e}')
                time.sleep(2 ** attempt)
        for pmid, state in parse(xml):
            if pmid in pmids:
                found[pmid] = state
        print(f'fetched {min(i + batch, len(todo))}/{len(todo)}', flush=True)
        time.sleep(pause)
    return found


def from_baseline(pmids, folder, files):
    found = {}
    for f in files:
        path = Path(folder) / f
        if not path.exists():
            raise SystemExit(f'missing baseline file {path}; download it from https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/')
        for pmid, state in parse(gzip.decompress(path.read_bytes())):
            if pmid in pmids:
                found[pmid] = state
        print(f, 'done', flush=True)
    return found


def load(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            for k in ('questions', 'gold'):            # the release stores these as JSON strings
                if isinstance(r[k], str):
                    r[k] = json.loads(r[k])
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default='data', help='folder with the released <split>.jsonl files')
    ap.add_argument('--manifest', default=None, help='manifest.json of the release (default: next to --data)')
    ap.add_argument('--out', default='prepared/medline_s1')
    ap.add_argument('--baseline-dir', default=None, help='use these PubMed baseline files instead of E-utilities')
    ap.add_argument('--email', default=None, help='contact address sent to NCBI, as their usage policy asks')
    ap.add_argument('--api-key', default=None, help='NCBI API key (optional, raises the rate limit)')
    ap.add_argument('--drop-changed', action='store_true', help='drop records whose text changed since the paper')
    a = ap.parse_args()

    data = Path(a.data)
    manifest = json.loads(Path(a.manifest or data.parent / 'manifest.json').read_text())
    splits = {s: load(data / f'{s}.jsonl') for s in SPLITS if (data / f'{s}.jsonl').exists()}
    pmids = {r['pmid'] for rows in splits.values() for r in rows}
    if a.baseline_dir:
        files = sorted({f for s in splits for f in manifest[f'medline_s1/{s}']['files']})
        text = from_baseline(pmids, a.baseline_dir, files)
    else:
        text = from_eutils(pmids, a.email, a.api_key)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    total = {'match': 0, 'changed': 0, 'missing': 0}
    for split, rows in splits.items():
        n = {'match': 0, 'changed': 0, 'missing': 0}
        path = out / f'{split}.jsonl'
        h = hashlib.sha256()
        with open(path, 'w', encoding='utf-8', newline='\n') as f:
            for r in rows:
                state = text.get(r['pmid'])
                if state is None:
                    n['missing'] += 1
                    continue
                same = hashlib.sha256(state.encode('utf-8')).hexdigest() == r['state_sha256']
                n['match' if same else 'changed'] += 1
                if not same and a.drop_changed:
                    continue
                rec = dict(id=r['id'], task=r['task'], split=r['split'], state=state, questions=r['questions'],
                           gold=r['gold'], meta=dict(pmid=r['pmid'], year=r['year']))
                line = (json.dumps(rec, ensure_ascii=False) + '\n').encode('utf-8')
                f.write(line.decode('utf-8'))
                h.update(line)
        ok = h.hexdigest() == manifest[f'medline_s1/{split}']['sha256_lf']        # hash with \n line endings
        print(f'{split:6s} {len(rows):6d} records: {n["match"]} identical, {n["changed"]} changed, '
              f'{n["missing"]} not found; file {"matches" if ok else "differs from"} the paper build')
        for k in total:
            total[k] += n[k]
    if total['changed'] or total['missing']:
        print(f'{total["changed"]} abstracts were revised and {total["missing"]} citations were not returned since the '
              'paper build. Results on the restored files can differ slightly from the paper; use --baseline-dir '
              'for an exact copy.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
