"""Minimal example: python inference/example.py  (downloads pritamdeka/S1-PubMedBERT on first use)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sbert2s1 import load  # noqa: E402

model = load("pritamdeka/S1-PubMedBERT")          # or a local export directory; GPU if available

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
print(json.dumps(model.predict(abstract, questions), indent=2))
