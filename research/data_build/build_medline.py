"""MEDLINE-S1: multi-question typed decisions over PubMed abstracts, labelled by NLM indexing (no LLM).

    python data_build/build_medline.py --out prepared --train-files 12 --test-files 3

Questions per abstract (only those whose label is defined are asked):
  design         choice   publication types (+ observational-design MeSH), precedence SR/MA > RCT > other trial
                          > case report > observational > narrative review; ambiguous or none -> not asked
  disease_area   choice   the single MeSH C-tree top category among *major-topic* descriptors, vs 3-7 random
                          other categories (random, not model-mined, so no encoder is favoured)
  humans         noul     MeSH check tag Humans
  animals        noul     MeSH check tag Animals
  adverse        noul     any heading with subheading "adverse effects"
  drug_therapy   noul     any heading with subheading "drug therapy"
  age_* / sex_*  noul     check tags (asked for Humans articles only; 2 sampled per abstract)
Only citations whose MeSH indexing is not purely automated (IndexingMethod Manual or Curated) from
2005-2021 are used; PMIDs in any
BioDecide dev/test set (prepared/eval_pmids.json) are excluded. Train and test come from disjoint files.

NLM replaces the baseline every year, so this script builds the paper's set only from the 2026 baseline.
To get the paper's set, download the release from the Hugging Face Hub (pritamdeka/MEDLINE-S1) and run
data_build/hydrate_medline.py.
"""
import argparse
import gzip
import io
import json
import random
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s1 import schema  # noqa: E402
from s1.common import file_sha256, write_jsonl  # noqa: E402

FTP = 'https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/'
MESH = 'https://nlmpubs.nlm.nih.gov/projects/mesh/{0}/meshtrees/mtrees{0}.bin'
SEED = 20260929
YEARS = (2005, 2021)

DESIGNS = [
    ('systematic_review', 'a systematic review or meta-analysis of previous studies', {'Meta-Analysis', 'Systematic Review'}),
    ('randomized_trial', 'a randomized controlled trial', {'Randomized Controlled Trial'}),
    ('other_trial', 'a clinical trial without randomization', {'Clinical Trial', 'Controlled Clinical Trial',
                                                              'Clinical Trial, Phase I', 'Clinical Trial, Phase II',
                                                              'Clinical Trial, Phase III', 'Clinical Trial, Phase IV'}),
    ('case_report', 'a case report of one or a few patients', {'Case Reports'}),
    ('observational', 'an observational study (cohort, case-control or cross-sectional)', {'Observational Study'}),
    ('narrative_review', 'a narrative review article', {'Review'}),
]
OBS_MESH = {'Cohort Studies', 'Case-Control Studies', 'Cross-Sectional Studies', 'Prospective Studies',
            'Retrospective Studies', 'Longitudinal Studies', 'Follow-Up Studies'}
AGE = {'infants': ({'Infant', 'Infant, Newborn'}, 'infants or newborns'),
       'children': ({'Child', 'Child, Preschool'}, 'children'),
       'adolescents': ({'Adolescent'}, 'adolescents'),
       'adults': ({'Adult', 'Young Adult', 'Middle Aged'}, 'adults under 65'),
       'older_adults': ({'Aged', 'Aged, 80 and over'}, 'adults aged 65 or older')}
SEX = {'female': ({'Female'}, 'female participants'), 'male': ({'Male'}, 'male participants')}
EXCLUDED_C = {'C22', 'C23'}          # animal diseases; generic signs & symptoms


def fetch(url, timeout=300):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def baseline_files():
    html = fetch(FTP).decode()
    files = sorted(set(re.findall(r'(pubmed\d+n\d{4}\.xml\.gz)"', html)))
    if not files:
        raise SystemExit('No baseline files found at ' + FTP)
    return files


def mesh_trees():
    for year in (2026, 2025, 2024):
        try:
            text = fetch(MESH.format(year)).decode('utf-8', 'replace')
        except Exception:
            continue
        if ';C01' in text:                  # NLM serves an HTML page, not a 404, for missing files
            break
    else:
        raise SystemExit('Could not download a MeSH trees file')
    name_to_top, top_names = {}, {}
    for line in text.splitlines():
        if ';' not in line:
            continue
        name, tree = line.rsplit(';', 1)
        name, tree = name.strip(), tree.strip()          # the file indents every line with a space
        if not tree.startswith('C'):
            continue
        top = tree.split('.')[0]
        if '.' not in tree:
            top_names[top] = name
        name_to_top.setdefault(name, set()).add(top)
    if len(top_names) < 20:
        raise SystemExit(f'MeSH trees parse failed: only {len(top_names)} C categories')
    return name_to_top, top_names, year


def parse(xml_bytes):
    from lxml import etree
    for _, el in etree.iterparse(io.BytesIO(xml_bytes), tag='PubmedArticle'):
        mc = el.find('MedlineCitation')
        try:
            pmid = int(mc.findtext('PMID'))
            art = mc.find('Article')
            year = art.findtext('Journal/JournalIssue/PubDate/Year') or (art.findtext('Journal/JournalIssue/PubDate/MedlineDate') or '')[:4]
            abstract = ' '.join(''.join(a.itertext()).strip() for a in art.findall('Abstract/AbstractText'))
            rec = dict(pmid=pmid, year=int(year) if year.isdigit() else None,
                       # IndexingMethod is Manual, Curated (automated then human-reviewed) or Automated.
                       automated=mc.get('IndexingMethod') == 'Automated',
                       title=''.join(art.find('ArticleTitle').itertext()).strip() if art.find('ArticleTitle') is not None else '',
                       abstract=abstract,
                       ptypes={p.text for p in art.findall('PublicationTypeList/PublicationType')},
                       mesh=[], quals=set())
            for mh in mc.findall('MeshHeadingList/MeshHeading'):
                d = mh.find('DescriptorName')
                qs = mh.findall('QualifierName')
                major = d.get('MajorTopicYN') == 'Y' or any(q.get('MajorTopicYN') == 'Y' for q in qs)
                rec['mesh'].append((d.text, major))
                rec['quals'].update(q.text for q in qs)
            yield rec
        except Exception:
            pass
        finally:
            el.clear()


def design_of(r):
    names = {m for m, _ in r['mesh']}
    for key, text, pts in DESIGNS:
        if r['ptypes'] & pts or (key == 'observational' and names & OBS_MESH and not r['ptypes'] & {'Review'}):
            return key
    return None


def disease_of(r, name_to_top):
    tops = set()
    for name, major in r['mesh']:
        if major:
            tops |= name_to_top.get(name, set())
    tops -= EXCLUDED_C
    return next(iter(tops)) if len(tops) == 1 else None


def to_record(r, i, split, rng, name_to_top, top_names):
    names = {m for m, _ in r['mesh']}
    qs, gold = {}, {}

    def noul(qid, text, value):
        qs[qid] = {'type': 'noul', 'instructions': text}
        gold[qid] = {'true' if value else 'false': 1.0}
    d = design_of(r)
    if d:
        qs['design'] = {'type': 'choice', 'instructions': 'What type of study or article is this?',
                        'criteria': {k: t for k, t, _ in DESIGNS}}
        gold['design'] = {d: 1.0}
    c = disease_of(r, name_to_top)
    if c:
        pool = sorted(k for k in top_names if k not in EXCLUDED_C and k != c)
        opts = rng.sample(pool, rng.randint(3, 7)) + [c]
        rng.shuffle(opts)
        qs['disease_area'] = {'type': 'choice', 'instructions': 'Which disease area is the main focus of this article?',
                              'criteria': {top_names[k]: '' for k in opts}}
        gold['disease_area'] = {top_names[c]: 1.0}
    humans = 'Humans' in names
    noul('humans', 'Did this research study human beings (patients or participants)?', humans)
    noul('animals', 'Did this research involve animals (animal models or veterinary subjects)?', 'Animals' in names)
    noul('adverse', 'Does the article report adverse effects of a drug, device or procedure?', 'adverse effects' in r['quals'])
    noul('drug_therapy', 'Does the article study drug treatment of a disease?', 'drug therapy' in r['quals'])
    if humans:
        extra = list(AGE.items()) + list(SEX.items())
        for key, (tags, text) in rng.sample(extra, 2):
            noul(f'group_{key}', f'Did the study include {text}?', bool(names & tags))
    state = (r['title'] + ' ' + r['abstract']).strip()
    return schema.validate(dict(id=f'medline_s1:{split}:{i:06d}', task='medline_s1', split=split, state=state,
                                questions=qs, gold={q: {'probabilities': p} for q, p in gold.items()},
                                meta=dict(pmid=r['pmid'], year=r['year'])))


def collect(files, cache, exclude, limit):
    out = []
    for f in files:
        path = cache / f
        if not path.exists():
            path.write_bytes(fetch(FTP + f))
        for r in parse(gzip.decompress(path.read_bytes())):
            if (r['automated'] or not r['abstract'] or len(r['abstract']) < 400 or r['pmid'] in exclude
                    or r['year'] is None or not (YEARS[0] <= r['year'] <= YEARS[1]) or not r['mesh']):
                continue
            out.append(r)
        print(f, 'usable so far', len(out), flush=True)
        if len(out) >= limit:
            break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default=str(Path(__file__).resolve().parents[1] / 'prepared'))
    ap.add_argument('--cache', default=str(Path(__file__).resolve().parents[1] / 'cache' / 'pubmed'))
    ap.add_argument('--train-files', type=int, default=14)
    ap.add_argument('--test-files', type=int, default=3)
    ap.add_argument('--n-train', type=int, default=40000)
    ap.add_argument('--n-dev', type=int, default=2000)
    ap.add_argument('--n-test', type=int, default=5000)
    a = ap.parse_args()
    out, cache = Path(a.out), Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    exclude = set(json.loads((out / 'eval_pmids.json').read_text())) if (out / 'eval_pmids.json').exists() else set()
    name_to_top, top_names, mesh_year = mesh_trees()
    files = baseline_files()
    # Files are in PMID order; the 2005-2021 range sits roughly in the middle 40-85% of the list.
    lo, hi = int(len(files) * 0.40), int(len(files) * 0.85)
    rng = random.Random(SEED)
    chosen = rng.sample(files[lo:hi], a.train_files + a.test_files)
    train_files, test_files = chosen[:a.train_files], chosen[a.train_files:]
    train_raw = collect(train_files, cache, exclude, a.n_train + a.n_dev + 400)
    test_raw = collect(test_files, cache, exclude, a.n_test)
    rng.shuffle(train_raw)
    rng.shuffle(test_raw)
    splits = {'calib': train_raw[:400], 'dev': train_raw[400:400 + a.n_dev],
              'train': train_raw[400 + a.n_dev:400 + a.n_dev + a.n_train], 'test': test_raw[:a.n_test]}
    manifest = json.loads((out / 'manifest.json').read_text()) if (out / 'manifest.json').exists() else {}
    for split, raws in splits.items():
        rows = [to_record(r, i, split, random.Random(SEED + r['pmid']), name_to_top, top_names) for i, r in enumerate(raws)]
        path = out / 'medline_s1' / f'{split}.jsonl'
        write_jsonl(path, rows)
        qcount = Counter(q for r in rows for q in r['gold'])
        manifest[f'medline_s1/{split}'] = dict(states=len(rows), decisions=sum(qcount.values()), sha256=file_sha256(path),
                                               questions=dict(qcount), mesh_trees_year=mesh_year,
                                               files=train_files if split != 'test' else test_files)
        print(split, len(rows), dict(qcount), flush=True)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1, sort_keys=True))


if __name__ == '__main__':
    main()
