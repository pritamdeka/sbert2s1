# sbert2s1

**Calibrated, single-pass typed decisions from biomedical sentence encoders.**

`sbert2s1` turns a Sentence-Transformers encoder into a *System One* decision model: you give it a
text and a set of typed questions (`choice`, `score` or yes/no `noul`), and one forward pass returns a
temperature-scaled probability for every option. The probabilities are meant to be thresholded, for
example to accept confident answers automatically and escalate the rest to a person or an LLM.

This repository contains

* **`inference/`**: a single file of plain PyTorch (`sbert2s1.py`, about 300 lines) that loads an
  exported model from a local folder or the Hugging Face Hub and answers requests. The same file ships
  with the released model.
* **`research/`**: the code used for the paper. It covers the four conversions (Z, B, C, PFR), the
  training objectives (CE, proper scores, RLCD and an unbiased leave-one-out variant), temperature
  fitting, the BioDecide and MEDLINE-S1 data builders, an experiment runner that works on any single
  GPU or scheduler, and the analysis and table and figure scripts.

The released model is [`pritamdeka/S1-PubMedBERT`](https://huggingface.co/pritamdeka/S1-PubMedBERT).

## Quick start

```bash
git clone https://github.com/pritamdeka/sbert2s1.git && cd sbert2s1
pip install -r inference/requirements.txt       # torch, transformers, safetensors, huggingface_hub
python inference/example.py
```

In your own code, copy `inference/sbert2s1.py` next to your script, or download it from the model
repository, then:

```python
from sbert2s1 import load

model = load("pritamdeka/S1-PubMedBERT")          # Hub id or a local export directory; GPU if available

abstract = ("In this double-blind trial, 4,012 adults with atrial fibrillation were randomised to drug X "
            "or placebo. Drug X reduced stroke by 29% (HR 0.71, 95% CI 0.60-0.84).")

questions = {
    "design": {"type": "choice", "instructions": "What is the study design?",
               "criteria": {"rct": "randomised controlled trial", "cohort": "cohort study",
                            "case_report": "case report", "other": "other design"}},
    "humans": {"type": "noul", "instructions": "Does the study involve human participants?"},
    "evidence": {"type": "score", "instructions": "How strong is the evidence that drug X prevents stroke?",
                 "criteria": ["none", "weak", "moderate", "strong"]},
}

result = model.predict(abstract, questions)
result["answers"]["design"]      # {"type": "choice", "choice": ..., "answer_confidence": ..., "probabilities": {...}}
result["answers"]["humans"]      # {"type": "noul", "noul": P(true), "answer_confidence": ...}
result["answers"]["evidence"]    # {"type": "score", "score": expected level, "probabilities": {...}}
```

* `model.predict_batch([(text, questions), ...], batch_size=32)` answers many requests at once.
* `model.logits([...])` returns the unscaled logits.
* The state can be a string or any JSON-serialisable record.
* The model reads at most 512 tokens (question block first). Longer texts are truncated and the answer
  is marked `"truncated": true`.
* `pytest inference/` runs CPU tests on a tiny random model (no downloads).

### Request schema

| Type | `criteria` | Answer |
|---|---|---|
| `choice` | `{key: description, ...}` (at least 2) | `choice`, `probabilities`, `answer_confidence` |
| `score` | `[level_0, level_1, ...]`, ordered low to high | `score` (expected level), `probabilities` |
| `noul` | optional `{"false": ..., "true": ...}` | `noul` = P(true) |

This is the public Jev/Laya typed-decision schema, so requests written for those systems work unchanged.

## What the paper found

On **BioDecide** (eleven public biomedical tasks plus a credentialed clinical track) and
**MEDLINE-S1** (243k decisions derived from NLM indexing), across eleven base-size encoders:

* Retrieval training gives useful **zero-shot** matching of content-bearing options. After
  fine-tuning, its effect depends on the conversion: across five parent-retriever pairs and three data
  sizes it helps the prior-fused residual head (PFR) in 10 of 15 comparisons, most with little labelled
  data, but helps the cross head (C) in one and hurts it in five.
* The **cross head is the strongest conversion** under every training objective; PFR is almost
  invariant to option order.
* **Cross-entropy plus temperature scaling matches or beats the released RLCD recipe.** RLCD's
  deficit comes from its reward normalisation, which inflates a noisy score-function term
  3.6 to 15 times. An unbiased leave-one-out estimator (`rlcd_pg_loo`) recovers most of the gap.
* Fast encoders and LLMs are complementary: escalating the 20% least confident questions to
  Gemma-4-31B raises macro accuracy (seen and held-out tasks) from 59.5 to 68.4 for
  S-PubMedBERT with a cross head.

## Reproducing the experiments

See [`research/README.md`](research/README.md). In short:

```bash
cd research
pip install -r requirements.txt
python data_build/build_public.py --out prepared     # downloads and converts the public tasks
huggingface-cli download pritamdeka/MEDLINE-S1 --repo-type dataset --local-dir medline_s1_release
python data_build/hydrate_medline.py --data medline_s1_release/data --out prepared/medline_s1   # adds the abstracts
python data_build/verify_build.py                    # compares your build with the paper's, record by record
python tests/test_cpu.py                             # end-to-end CPU test
python experiments.py list                           # every run in the paper
python experiments.py run --group rq3c               # run a group locally (or: commands, for any scheduler)
```

Every encoder experiment fits on one GPU. The aggregate metrics of every run behind the paper are in
`research/results/metrics_long.csv.gz`.

**Credentialed data.** The clinical track uses MedNLI, MIMIC-III and MIMIC-IV derivatives from
PhysioNet. No credentialed data, derived examples or per-item clinical predictions are included in this
repository. `data_build/build_clinical.py` converts your own copies of the PhysioNet releases. Under
the PhysioNet data use agreements, never send these data to hosted models or APIs.

## Licence

* Code: Apache License 2.0 (see `LICENSE` and `NOTICE`).
* Released model weights: CC BY-NC 2.0, inherited from
  [S-PubMedBert-MS-MARCO](https://huggingface.co/pritamdeka/S-PubMedBert-MS-MARCO).
* MEDLINE-S1 questions, labels and splits: CC BY 4.0, on the Hugging Face Hub as
  [pritamdeka/MEDLINE-S1](https://huggingface.co/datasets/pritamdeka/MEDLINE-S1), without the abstract
  text (which `hydrate_medline.py` fetches from PubMed).
* The BioDecide source datasets are not redistributed and keep their own licences. The builders
  download them, and `research/docs/` holds the hashes needed to check a build against the paper's.

The model is not a medical device. It is not validated for clinical decisions.

## Citation

```bibtex
@misc{deka2026sbert2s1,
  title  = {From Retrieval to Typed Decisions: Calibrated System One Models from Biomedical Sentence Encoders},
  author = {Deka, Pritam},
  year   = {2026},
  note   = {Preprint}
}
```
