# sbert2s1

**Calibrated, single-pass typed decisions from biomedical sentence encoders.**

`sbert2s1` turns a Sentence-Transformers encoder into a *System One* decision model: you give it a
text and a set of typed questions (`choice`, `score` or yes/no `noul`), and one forward pass returns a
temperature-scaled probability for every option. The probabilities are meant to be thresholded, for
example to accept confident answers automatically and escalate the rest to a person or an LLM.

This repository contains

* the **pip package** (`pip install sbert2s1`): inference for exported models, with a Python API and
  a command-line tool;
* the **research code** (`research/`) used for the paper: the four conversions (Z, B, C, PFR),
  the training objectives (CE, proper scores, RLCD and an unbiased leave-one-out variant), temperature
  fitting, the BioDecide and MEDLINE-S1 data builders, the SLURM campaign scripts, and the analysis
  and table/figure scripts.

The released model is [`pritamdeka/S1-PubMedBERT`](https://huggingface.co/pritamdeka/S1-PubMedBERT).

## Install

```bash
pip install sbert2s1            # needs PyTorch; for CPU-only machines install torch from the CPU index first
```

## Quick start

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

### Command line

```bash
# requests.jsonl: one {"id": ..., "state": ..., "questions": {...}} per line
sbert2s1 --model pritamdeka/S1-PubMedBERT --input requests.jsonl --output answers.jsonl
```

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
  fine-tuning, its effect depends on the conversion: neutral for the cross head (C), positive in three
  of five pairs for the prior-fused residual head (PFR).
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
python data_build/build_medline.py --out prepared    # MEDLINE-S1 from the NLM baseline files
python tests/test_cpu.py                             # end-to-end CPU test
```

The campaign itself (about 40 GPU-hours on AMD MI300X for the main grid) is a packed SLURM queue (`scripts/submit.sh`,
`slurm/worker.slurm`). The aggregate metrics of every run behind the paper are in
`research/results/metrics_long.csv.gz`.

**Credentialed data.** The clinical track uses MedNLI, MIMIC-III and MIMIC-IV derivatives from
PhysioNet. No credentialed data, derived examples or per-item clinical predictions are included in this
repository. `data_build/build_clinical.py` converts your own copies of the PhysioNet releases. Under
the PhysioNet data use agreements, never send these data to hosted models or APIs.

## Licence

* Code: Apache License 2.0 (see `LICENSE` and `NOTICE`).
* Released model weights: CC BY-NC 2.0, inherited from
  [S-PubMedBert-MS-MARCO](https://huggingface.co/pritamdeka/S-PubMedBert-MS-MARCO).
* Datasets are not redistributed and keep their own licences.

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
