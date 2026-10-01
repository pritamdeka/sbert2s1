"""sbert2s1: calibrated, single-pass typed decisions from biomedical sentence encoders.

    >>> from sbert2s1 import load
    >>> model = load("pritamdeka/S1-PubMedBERT")
    >>> model.predict(abstract, {"design": {"type": "choice", "instructions": "What is the study design?",
    ...                                     "criteria": {"rct": "randomised controlled trial",
    ...                                                  "cohort": "cohort study"}}})
"""
from ._inference import (NOUL_DEFAULT, QTYPES, Predictor, S1Model, load, option_keys,
                         render_options)

__version__ = "0.1.0"
__all__ = ["load", "Predictor", "S1Model", "QTYPES", "NOUL_DEFAULT", "option_keys", "render_options",
           "__version__"]
