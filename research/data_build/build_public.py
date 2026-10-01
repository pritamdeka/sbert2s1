"""Download and convert every public BioDecide task into the typed-decision record format.

Run on a machine with internet (the authors' workstation), then upload `prepared/`:
    python data_build/build_public.py --out prepared
Each task gets prepared/<task>/{train,calib,dev,test}.jsonl (only the splits it has). `calib` is carved
from train before any training and is used only for temperature fitting. Test files are hashed into
prepared/manifest.json. Record format: see s1/schema.py.
"""
import argparse
import io
import json
import random
import re
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s1 import schema  # noqa: E402
from s1.common import file_sha256, write_jsonl  # noqa: E402

SEED = 20260929
CALIB_MAX = 400


def hub(repo, path, rev=None):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, path, repo_type='dataset', revision=rev)


def pq(repo, path, rev=None):
    return pd.read_parquet(hub(repo, path, rev))


CONVERT = 'refs/convert/parquet'


def rec(task, split, i, state, questions, gold, **meta):
    return schema.validate(dict(id=f'{task}:{split}:{i:06d}', task=task, split=split, state=state,
                                questions=questions, gold={q: {'probabilities': p} for q, p in gold.items()},
                                meta=meta))


def one_hot(key):
    return {key: 1.0}


def carve(rows, n, seed, key=None):
    """Split rows into (rest, carved) with n carved, grouped by `key` (so a state never straddles)."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for r in rows:
        groups[key(r) if key else id(r)].append(r)
    ks = list(groups)
    rng.shuffle(ks)
    carved, rest, count = [], [], 0
    for k in ks:
        if count < n:
            carved += groups[k]
            count += len(groups[k])
        else:
            rest += groups[k]
    return rest, carved


def calib_n(n):
    return min(CALIB_MAX, max(10, n // 10))


def renumber(task, split, rows):
    for i, r in enumerate(rows):
        r['id'], r['split'] = f'{task}:{split}:{i:06d}', split
    return rows


# ---- tasks -------------------------------------------------------------------------------

PMQA_CRIT = {'yes': 'the study results support answering yes',
             'no': 'the study results support answering no',
             'maybe': 'the evidence is mixed, inconclusive or conditional'}


def pubmedqa():
    df = pq('qiaojin/PubMedQA', 'pqa_labeled/train-00000-of-00001.parquet')
    rows = []
    for i, r in df.iterrows():
        state = ' '.join(r['context']['contexts'])      # long_answer (the conclusion) is excluded: it leaks
        q = {'type': 'choice', 'instructions': r['question'], 'criteria': PMQA_CRIT}
        rows.append(rec('pubmedqa', 'x', i, state, {'answer': q}, {'answer': one_hot(r['final_decision'])},
                        pmid=int(r['pubid'])))
    rest, test = carve(rows, 500, SEED)
    rest, dev = carve(rest, 50, SEED + 1)
    train, calib = carve(rest, 50, SEED + 2)
    return {'train': train, 'calib': calib, 'dev': dev, 'test': test}


def pubmedqa_art(exclude_pmids, n_train=20000):
    df = pq('qiaojin/PubMedQA', 'pqa_artificial/train-00000-of-00001.parquet')
    df = df[~df.pubid.isin(exclude_pmids)].sample(n=n_train + CALIB_MAX, random_state=SEED)
    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        state = ' '.join(r['context']['contexts'])
        q = {'type': 'noul', 'instructions': r['question']}
        rows.append(rec('pubmedqa_art', 'x', i, state, {'answer': q},
                        {'answer': one_hot('true' if r['final_decision'] == 'yes' else 'false')}, pmid=int(r['pubid'])))
    train, calib = carve(rows, CALIB_MAX, SEED)
    return {'train': train, 'calib': calib}


VERIFY_CRIT = {'supports': 'the text provides evidence that the claim is true',
               'refutes': 'the text provides evidence that the claim is false',
               'not_enough_info': 'the text does not provide enough evidence either way'}


def scifact():
    corpus = pq('allenai/scifact', 'corpus/train/0000.parquet', CONVERT)
    docs = {int(r.doc_id): (r.title + '. ' + ' '.join(r.abstract)) for r in corpus.itertuples()}
    out = {}
    for split, src in (('train', 'train'), ('test', 'validation')):
        df = pq('allenai/scifact', f'claims/{src}/0000.parquet', CONVERT)
        seen, rows = set(), []
        for r in df.itertuples():
            if r.evidence_doc_id:
                doc, lab = int(r.evidence_doc_id), {'SUPPORT': 'supports', 'CONTRADICT': 'refutes'}[r.evidence_label]
            else:
                if not len(r.cited_doc_ids):
                    continue
                doc, lab = int(r.cited_doc_ids[0]), 'not_enough_info'
            if (r.id, doc) in seen or doc not in docs:
                continue
            seen.add((r.id, doc))
            q = {'type': 'choice', 'instructions': f'Does this abstract support or refute the claim: {r.claim}',
                 'criteria': VERIFY_CRIT}
            rows.append(rec('scifact', split, len(rows), docs[doc], {'verdict': q}, {'verdict': one_hot(lab)},
                            pmid=doc, claim_id=int(r.id)))
        out[split] = rows
    out['train'], out['calib'] = carve(out['train'], calib_n(len(out['train'])), SEED, key=lambda r: r['meta']['claim_id'])
    return out


def healthver():
    base = 'https://raw.githubusercontent.com/sarrouti/HealthVer/master/data/healthver_{}.csv'
    lab = {'Supports': 'supports', 'Refutes': 'refutes', 'Neutral': 'not_enough_info'}
    out = {}
    for split in ('train', 'dev', 'test'):
        df = pd.read_csv(base.format(split))
        rows = []
        for r in df.itertuples():
            q = {'type': 'choice', 'instructions': f'Does this evidence support or refute the claim: {r.claim.strip()}',
                 'criteria': VERIFY_CRIT}
            rows.append(rec('healthver', split, len(rows), r.evidence.strip(), {'verdict': q},
                            {'verdict': one_hot(lab[r.label])}, topic=int(r.topic_ip)))
        out[split] = rows
    # HealthVer splits are grouped by topic; keep calib topics disjoint from training topics too.
    out['train'], out['calib'] = carve(out['train'], calib_n(len(out['train'])), SEED, key=lambda r: r['meta']['topic'])
    return out


def pubhealth():
    df = pq('bigbio/pubhealth', 'pubhealth_source/test/0000.parquet', CONVERT)
    names = ['true', 'false', 'unproven', 'mixture']      # bigbio pubhealth.py _CLASSES order (not health_fact's)
    crit = {'false': 'the claim is false', 'mixture': 'the claim is partly true and partly false',
            'true': 'the claim is true', 'unproven': 'there is not enough evidence to decide'}
    rows = []
    for r in df.itertuples():
        if not (0 <= int(r.label) < 4) or not isinstance(r.main_text, str) or not r.main_text.strip():
            continue
        q = {'type': 'choice', 'instructions': f'Based on this article, what is the veracity of the health claim: {r.claim}',
             'criteria': crit}
        # The fact-checkers' `explanation` is excluded: it states the verdict.
        rows.append(rec('pubhealth', 'test', len(rows), r.main_text, {'verdict': q}, {'verdict': one_hot(names[int(r.label)])}))
    return {'test': rows}


DDI_CRIT = {'mechanism': 'a pharmacokinetic interaction (one drug changes the absorption, metabolism or levels of the other)',
            'effect': 'an interaction described by its clinical or pharmacodynamic effect',
            'advice': 'a recommendation or warning about using the two drugs together',
            'int': 'an interaction is stated without further detail',
            'none': 'no interaction between the two marked drugs is described'}
DDI_RE = re.compile(r'INPUT:\s*(.*?)\nWhat is the relationship', re.S)


def ddi():
    out = {}
    seen_train = set()
    for split, repo in (('train', 'clinicalnlplab/DDI2013_train'), ('test', 'clinicalnlplab/DDI2013_dev')):
        df = pq(repo, 'data/' + ('train' if split == 'train' else 'valid') + '-00000-of-00001.parquet')
        rows = []
        for r in df.itertuples():
            m = DDI_RE.search(r.conversations[0]['value'])
            lab = r.conversations[1]['value'].strip()
            if not m or lab not in DDI_CRIT:
                continue
            sent = m.group(1).strip()
            if split == 'train':
                seen_train.add(sent)
            elif sent in seen_train:          # never test on a sentence seen in training
                continue
            q = {'type': 'choice', 'instructions': 'What kind of interaction between @DRUG1$ and @DRUG2$ does the sentence describe?',
                 'criteria': DDI_CRIT}
            rows.append(rec('ddi', split, len(rows), sent, {'relation': q}, {'relation': one_hot(lab)}))
        out[split] = rows
    out['train'], out['calib'] = carve(out['train'], calib_n(len(out['train'])), SEED)
    return out


HOC = ['evading growth suppressors', 'tumor promoting inflammation', 'enabling replicative immortality',
       'cellular energetics', 'resisting cell death', 'activating invasion and metastasis',
       'genomic instability and mutation', None, 'inducing angiogenesis', 'sustaining proliferative signaling',
       'avoiding immune destruction']


def hoc():
    out = {}
    for split, src in (('train', 'train'), ('dev', 'validation'), ('test', 'test')):
        df = pq('qanastek/HoC', f'HoC/{src}/0000.parquet', CONVERT)
        docs = defaultdict(lambda: ([], set()))
        for r in df.itertuples():
            pmid, idx = r.document_id.rsplit('_', 1)
            docs[pmid][0].append((int(idx), r.text))
            docs[pmid][1].update(int(x) for x in r.label if int(x) != 7)
        rows = []
        for pmid, (sents, labels) in docs.items():
            state = ' '.join(t for _, t in sorted(sents))
            qs, gold = {}, {}
            for k, name in enumerate(HOC):
                if name is None:
                    continue
                qid = 'h_' + re.sub(r'\W+', '_', name)
                qs[qid] = {'type': 'noul', 'instructions': f'Does this abstract discuss the cancer hallmark "{name}"?'}
                gold[qid] = one_hot('true' if k in labels else 'false')
            rows.append(rec('hoc', split, len(rows), state, qs, gold, pmid=int(pmid)))
        out[split] = rows
    out['train'], out['calib'] = carve(out['train'], calib_n(len(out['train'])), SEED)
    return out


def ade():
    df = pq('ade-benchmark-corpus/ade_corpus_v2', 'Ade_corpus_v2_classification/train-00000-of-00001.parquet')
    df = df.drop_duplicates('text')
    rows = []
    for r in df.itertuples():
        q = {'type': 'noul', 'instructions': 'Does this sentence describe an adverse effect caused by a drug?'}
        rows.append(rec('ade', 'x', len(rows), r.text, {'adverse_effect': q},
                        {'adverse_effect': one_hot('true' if int(r.label) == 1 else 'false')}))
    rest, test = carve(rows, len(rows) // 10, SEED)
    rest, dev = carve(rest, len(rows) // 10, SEED + 1)
    train, calib = carve(rest, CALIB_MAX, SEED + 2)
    return {'train': train, 'calib': calib, 'dev': dev, 'test': test}


EFFECT = ['Ineffective', 'Marginally Effective', 'Moderately Effective', 'Considerably Effective', 'Highly Effective']
SIDE = ['No Side Effects', 'Mild Side Effects', 'Moderate Side Effects', 'Severe Side Effects', 'Extremely Severe Side Effects']
RATING = ['very poor (1-2 of 10)', 'poor (3-4 of 10)', 'average (5-6 of 10)', 'good (7-8 of 10)', 'excellent (9-10 of 10)']


def druglib():
    url = 'https://archive.ics.uci.edu/static/public/461/drug+review+dataset+druglib+com.zip'
    z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url, timeout=120).read()))

    def member(word):
        return next(n for n in z.namelist() if word in n.lower() and n.endswith('.tsv'))
    out = {}
    for split, word in (('train', 'train'), ('test', 'test')):
        df = pd.read_csv(z.open(member(word)), sep='\t')
        rows = []
        for r in df.itertuples():
            if r.effectiveness not in EFFECT or r.sideEffects not in SIDE:
                continue
            parts = [f'Drug: {r.urlDrugName}.', f'Condition: {r.condition}.' if isinstance(r.condition, str) else '',
                     f'Benefits: {r.benefitsReview}' if isinstance(r.benefitsReview, str) else '',
                     f'Side effects: {r.sideEffectsReview}' if isinstance(r.sideEffectsReview, str) else '',
                     f'Comments: {r.commentsReview}' if isinstance(r.commentsReview, str) else '']
            qs = {'effectiveness': {'type': 'score', 'instructions': 'How effective did the patient find the drug?', 'criteria': EFFECT},
                  'side_effects': {'type': 'score', 'instructions': 'How severe were the side effects the patient reports?', 'criteria': SIDE},
                  'rating': {'type': 'score', 'instructions': 'What overall rating would the patient give the drug?', 'criteria': RATING}}
            gold = {'effectiveness': one_hot(str(EFFECT.index(r.effectiveness))),
                    'side_effects': one_hot(str(SIDE.index(r.sideEffects))),
                    'rating': one_hot(str(min(4, (int(r.rating) - 1) // 2)))}
            rows.append(rec('druglib', split, len(rows), ' '.join(p for p in parts if p), qs, gold))
        out[split] = rows
    out['train'], out['calib'] = carve(out['train'], calib_n(len(out['train'])), SEED)
    return out


def mtsamples():
    df = pd.read_csv(hub('hpe-ai/medical-cases-classification-tutorial', 'medical_cases_test.csv'))
    df = df.dropna(subset=['transcription', 'medical_specialty'])
    specs = sorted(s.strip() for s in df.medical_specialty.unique())
    rows = []
    for r in df.itertuples():
        q = {'type': 'choice', 'instructions': 'Which medical specialty does this transcription belong to?',
             'criteria': {s: '' for s in specs}}
        rows.append(rec('mtsamples', 'test', len(rows), r.transcription, {'specialty': q},
                        {'specialty': one_hot(r.medical_specialty.strip())}))
    return {'test': rows}


SIM = ['completely different topics', 'same topic but different meaning', 'partly equivalent, important details differ',
       'mostly equivalent, minor details differ', 'completely equivalent in meaning']


def biosses():
    rows = []
    for split in ('train', 'validation', 'test'):
        df = pq('bigbio/biosses', f'biosses_bigbio_pairs/{split}/0000.parquet', CONVERT)
        for r in df.itertuples():
            s = float(r.label)
            lo = min(3, int(s))
            frac = s - lo
            gold = {str(lo): 1 - frac, str(lo + 1): frac} if frac > 0 else {str(int(s)): 1.0}
            q = {'type': 'score', 'instructions': f'How similar in meaning is this sentence to: "{r.text_2.strip()}"',
                 'criteria': SIM}
            rows.append(rec('biosses', 'test', len(rows), r.text_1.strip(), {'similarity': q}, {'similarity': gold}))
    return {'test': rows}


def _mcq_record(task, split, i, stem, options, answer_key):
    q = {'type': 'choice', 'instructions': 'Which option correctly answers the question in the text?',
         'criteria': {k: v for k, v in options.items()}}
    return rec(task, split, i, stem, {'answer': q}, {'answer': one_hot(answer_key)})


def medqa():
    out = {}
    for split in ('train', 'test'):
        p = hub('GBaker/MedQA-USMLE-4-options', f'phrases_no_exclude_{split}.jsonl')
        df = pd.read_json(p, lines=True)
        out[split] = [_mcq_record('medqa', split, i, r.question, dict(r.options), r.answer_idx) for i, r in enumerate(df.itertuples())]
    out['train'], out['calib'] = carve(out['train'], CALIB_MAX, SEED)
    return out


def medmcqa(n_train=20000):
    out = {}
    for split, src in (('train', 'train'), ('test', 'validation')):
        df = pq('openlifescienceai/medmcqa', f'data/{src}-00000-of-00001.parquet')
        df = df[df.choice_type == 'single']
        if split == 'train':
            df = df.sample(n=min(len(df), n_train + CALIB_MAX), random_state=SEED)
        rows = []
        for r in df.itertuples():
            opts = {'A': r.opa, 'B': r.opb, 'C': r.opc, 'D': r.opd}
            if any(not isinstance(v, str) or not v.strip() for v in opts.values()):
                continue
            rows.append(_mcq_record('medmcqa', split, len(rows), r.question, opts, 'ABCD'[int(r.cop)]))
        out[split] = rows
    out['train'], out['calib'] = carve(out['train'], CALIB_MAX, SEED)
    return out


MMLU_MED = ['anatomy', 'clinical_knowledge', 'college_biology', 'college_medicine', 'medical_genetics', 'professional_medicine']


def mmlu_med():
    rows = []
    for sub in MMLU_MED:
        df = pq('cais/mmlu', f'{sub}/test-00000-of-00001.parquet')
        for r in df.itertuples():
            opts = dict(zip('ABCD', r.choices))
            rows.append(_mcq_record('mmlu_med', 'test', len(rows), r.question, opts, 'ABCD'[int(r.answer)]))
            rows[-1]['meta']['subject'] = sub
    return {'test': rows}


def typed_decisions():
    df = pq('LocalLLaMA/typed-decisions', 'all/test-00000-of-00001.parquet')
    rows = []
    for r in df.itertuples():
        qs, gold = json.loads(r.questions), json.loads(r.gold)
        g = {}
        for qid, gq in gold.items():
            if 'probabilities' in gq:
                g[qid] = {str(k): float(v) for k, v in gq['probabilities'].items()}
            elif 'noul' in gq:
                g[qid] = {'true': float(gq['noul']), 'false': 1 - float(gq['noul'])}
        qs = {k: {kk: vv for kk, vv in v.items() if kk in ('type', 'instructions', 'criteria', 'labels')} for k, v in qs.items()}
        rows.append(rec('typed_decisions', 'test', len(rows), json.loads(r.state), qs, g, workflow=r.workflow))
    return {'test': rows}


def retrieval(out_dir):
    """BEIR SciFact + NFCorpus for the retention experiment (RQ4)."""
    for name in ('scifact', 'nfcorpus'):
        d = Path(out_dir) / '_retrieval' / name
        d.mkdir(parents=True, exist_ok=True)
        for f in ('corpus.jsonl', 'queries.jsonl', 'qrels/test.tsv'):
            src = hub(f'mteb/{name}', f)
            (d / f.replace('qrels/', 'qrels_')).write_bytes(Path(src).read_bytes())
        print('retrieval', name, 'ok', flush=True)


TASKS = ['pubmedqa', 'scifact', 'healthver', 'pubhealth', 'ddi', 'hoc', 'ade', 'druglib', 'mtsamples', 'biosses',
         'medqa', 'medmcqa', 'mmlu_med', 'typed_decisions', 'pubmedqa_art']


def eval_pmids(built):
    s = set()
    for task, splits in built.items():
        for split, rows in splits.items():
            if split in ('test', 'dev'):
                s.update(r['meta'].get('pmid') for r in rows if r['meta'].get('pmid'))
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default=str(Path(__file__).resolve().parents[1] / 'prepared'))
    ap.add_argument('--tasks', nargs='*', default=TASKS)
    ap.add_argument('--no-retrieval', action='store_true')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mpath, ppath = out / 'manifest.json', out / 'eval_pmids.json'
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    pmids = set(json.loads(ppath.read_text())) if ppath.exists() else set()

    def save(t, splits):
        """Write one task immediately, so a failure later does not lose finished tasks."""
        for split, rows in splits.items():
            rows = renumber(t, split, rows)
            path = out / t / f'{split}.jsonl'
            write_jsonl(path, rows)
            manifest[f'{t}/{split}'] = dict(states=len(rows), decisions=sum(len(r['gold']) for r in rows),
                                            sha256=file_sha256(path),
                                            labels=dict(Counter(max(g['probabilities'], key=g['probabilities'].get)
                                                                for r in rows for g in r['gold'].values()).most_common(12)))
        pmids.update(eval_pmids({t: splits}))
        mpath.write_text(json.dumps(manifest, indent=1, sort_keys=True))
        ppath.write_text(json.dumps(sorted(pmids)))
        print(t, {k: len(v) for k, v in splits.items()}, flush=True)

    train_pmids = set()
    for t in a.tasks:
        if t == 'pubmedqa_art':
            continue
        splits = globals()[t]()
        if t == 'pubmedqa':
            train_pmids = {r['meta']['pmid'] for r in splits['train']}
        save(t, splits)
    if 'pubmedqa_art' in a.tasks:
        # Every eval PMID (all tasks, including ones built in earlier invocations) is excluded.
        save('pubmedqa_art', pubmedqa_art(pmids | train_pmids))
    if not a.no_retrieval:
        retrieval(out)


if __name__ == '__main__':
    main()
