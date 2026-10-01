---
license: cc-by-nc-2.0
language:
- en
base_model: pritamdeka/S-PubMedBert-MS-MARCO
pipeline_tag: text-classification
tags:
- biomedical
- typed-decisions
- system-one
- calibration
- sbert2s1
---

# S1-PubMedBERT: a calibrated single-pass decision model for biomedical text

S1-PubMedBERT answers **typed questions about a text in one forward pass** and returns
temperature-scaled probabilities that can be thresholded directly. It is a *System One* decision model
in the sense of Jev and Laya, built with the **sbert2s1** framework by converting the retrieval encoder
[S-PubMedBert-MS-MARCO](https://huggingface.co/pritamdeka/S-PubMedBert-MS-MARCO) into a cross mask-slot
decision model (the "C" architecture).

Three question types share one request schema (the public Jev/Laya schema):

| Type | You give | You get |
|---|---|---|
| `choice` | a dict of option keys and descriptions | the chosen key, a probability per option, the confidence |
| `score` | an ordered list of rubric levels (low to high) | the expected level, a probability per level |
| `noul` | a statement-style instruction | P(true) |

Several questions about the same text can be asked in one call; outputs cannot violate the schema.

## Usage

```bash
pip install torch transformers safetensors huggingface_hub
```

```python
import importlib.util
from huggingface_hub import hf_hub_download

# the inference code ships with the model (read it first: about 300 lines of plain PyTorch)
path = hf_hub_download("pritamdeka/S1-PubMedBERT", "sbert2s1.py")
spec = importlib.util.spec_from_file_location("sbert2s1", path)
sbert2s1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sbert2s1)

model = sbert2s1.load("pritamdeka/S1-PubMedBERT")        # GPU if available, otherwise CPU

abstract = ("In this double-blind trial, 4,012 adults with atrial fibrillation were randomised to drug X "
            "or placebo. Drug X reduced stroke by 29% (HR 0.71, 95% CI 0.60-0.84); major bleeding was "
            "more frequent with drug X (3.1% vs 2.2%).")

questions = {
    "design": {"type": "choice", "instructions": "What is the study design?",
               "criteria": {"rct": "randomised controlled trial", "cohort": "cohort study",
                            "case_control": "case-control study", "case_report": "case report",
                            "review": "review or meta-analysis", "other": "other design"}},
    "humans": {"type": "noul", "instructions": "Does the study involve human participants?"},
    "adverse": {"type": "noul", "instructions": "Does the abstract report adverse effects of a drug?"},
    "evidence": {"type": "score", "instructions": "How strong is the evidence that drug X prevents stroke?",
                 "criteria": ["none", "weak", "moderate", "strong"]},
}

print(model.predict(abstract, questions))
```

Output format (illustrative placeholders, not real predictions):

```text
{"answers": {
  "design":   {"type": "choice", "choice": "<key>", "answer_confidence": <p_max>,
               "probabilities": {"rct": <p>, "cohort": <p>, ...}},
  "humans":   {"type": "noul", "noul": <P(true)>, "answer_confidence": <max(P(true), P(false))>},
  "adverse":  {"type": "noul", "noul": <P(true)>, "answer_confidence": <...>},
  "evidence": {"type": "score", "score": <expected level 0..3>, "answer_confidence": <p_max>,
               "probabilities": {"0": <p>, "1": <p>, "2": <p>, "3": <p>}}}}
```

**Batches.** `model.predict_batch([(text1, questions1), (text2, questions2), ...], batch_size=32)`.
**Raw logits.** `model.logits([...])` returns the unscaled logits per question.
**Using the confidence.** `answer_confidence` (or `noul`) is the quantity to threshold, e.g. accept
automatically above 0.9 and send the rest to a person or a larger model.

**Inputs.** The state may be a string or a JSON-serialisable record (it is serialised to text).
`noul` questions may define `criteria: {"false": ..., "true": ...}` and optional `labels`.
The model reads at most 512 tokens: the question block (instructions plus options, at most 192 tokens)
and then the text. Longer texts are truncated from the end, and the answer is marked `"truncated": true`;
for long documents, split the text and ask per chunk.

The `laya/` folder contains the same weights in the Laya checkpoint layout for Laya-compatible runtimes.

## Training

* **Initialisation:** `pritamdeka/S-PubMedBert-MS-MARCO` (PubMedBERT further trained for MS MARCO retrieval), 110M parameters, plus a two-layer mask-slot decision head (Laya-compatible).
* **Objective:** cross-entropy against the gold option distribution. In our experiments this matched or beat the released RLCD reinforcement-learning recipe (70.2 vs 67.7 seen-task points, 3 seeds each).
* **Data (public only; no credentialed clinical data):** PubMedQA (labelled and artificial), SciFact, HealthVer, DDI-2013, Hallmarks of Cancer, ADE, Druglib reviews, MedQA, MedMCQA, and MEDLINE-S1 (243k decisions derived from NLM MEDLINE indexing: study design, disease area, human/animal subjects, adverse effects, drug therapy, age/sex groups). Options are shuffled during training (except ordinal rubrics).
* **Recipe:** 8,000 steps, batch 32, AdamW (encoder 2.5e-5, head 1e-4), cosine schedule, bf16, temperature-sampled task mixture.
* **Calibration:** one temperature per (question type, option-count bucket), fitted on calibration slices carved from the training data before training.
* **Selection:** three seeds were trained; this checkpoint is the seed with the best seen-task accuracy (the 3-seed mean is 70.2 ± 0.4, so the number below is slightly optimistic).

## Evaluation

Chance-normalised accuracy `(acc - 1/K)/(1 - 1/K)` (0 = chance, 100 = perfect), averaged over question
groups within a task; ECE after temperature scaling (15 bins). Test splits; x100.

| Task (seen in training) | acc_cn | ECE |
|---|---:|---:|
| PubMedQA | 47.8 | 4.3 |
| SciFact | 61.2 | 8.8 |
| HealthVer | 56.0 | 2.8 |
| DDI-2013 | 92.9 | 4.1 |
| Hallmarks of Cancer (10 noul questions) | 88.8 | 2.4 |
| ADE | 88.6 | 1.3 |
| Druglib (3 score questions) | 41.6 | 4.3 |
| MEDLINE-S1 | 87.1 | 2.4 |
| **Mean of the 8 seen tasks** | **70.5** | **3.8** |

| Not trained on | acc_cn | ECE |
|---|---:|---:|
| PUBHEALTH (claim veracity) | 22.7 | 13.1 |
| MTSamples (specialty routing) | 72.2 | 27.5 |
| BIOSSES (similarity rubric) | -5.0 | 16.2 |
| Exam questions (MedQA / MedMCQA / MMLU-medical) | 14.8 / 13.1 / 14.8 | 1.3-3.0 |
| General-domain typed-decisions | 8.7 | 16.5 |

## Intended use and limitations

* **Intended use:** research and prototyping of fast, calibrated decisions on biomedical literature
  and similar text, close to the training tasks (evidence classification, claim verification, drug
  interactions/adverse events, indexing-style questions), typically as the first stage of a pipeline
  that escalates low-confidence cases.
* **Not a medical device** and not validated for clinical decisions. It was **not** trained on clinical
  notes and is weak on them. Calibration was measured on in-distribution test sets; on new tasks the
  probabilities can be badly miscalibrated (see MTSamples, PUBHEALTH). Re-fit temperatures (or validate
  thresholds) on your own labelled sample before relying on the confidence.
* **Out-of-domain and knowledge questions:** near chance on general-domain decisions and on exam
  questions that require memorised medical knowledge; a 110M encoder does not store that knowledge.
* **Sensitivity to wording:** answers depend on how instructions and options are phrased, and the
  cross head is somewhat sensitive to option order (about 3% of seen-task answers change under option
  permutation).
* **Licence:** CC BY-NC 2.0, inherited from the base model; non-commercial use only. Check the licences
  of the training datasets for your use case (several are themselves non-commercial).

## Citation

```bibtex
@misc{deka2026sbert2s1,
  title  = {From Retrieval to Typed Decisions: Calibrated System One Models from Biomedical Sentence Encoders},
  author = {Deka, Pritam},
  year   = {2026},
  note   = {TODO: add arXiv identifier}
}
@article{deka2022improved,
  title   = {Improved Methods To Aid Unsupervised Evidence-Based Fact Checking For Online Health News},
  author  = {Deka, Pritam and Jurek-Loughrey, Anna and Deepak, P.},
  journal = {Journal of Data Intelligence}, volume = {3}, number = {4}, pages = {474--504}, year = {2022}
}
```
