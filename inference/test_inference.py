"""CPU tests for inference/sbert2s1.py on a tiny randomly initialised export (no downloads).

    pip install pytest && pytest inference/
"""
import json
import math

import pytest
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sbert2s1 import S1Model, load, option_keys, render_options  # noqa: E402

WORDS = ("the a of and to in was were is with for patients trial randomised study cohort case report drug "
         "placebo effect reduced risk design what question choice score noul level true false yes no "
         "statement holds does not hold rct other").split()

REQ = {
    "design": {"type": "choice", "instructions": "What is the study design?",
               "criteria": {"rct": "randomised trial", "cohort": "cohort study", "case": "case report"}},
    "quality": {"type": "score", "instructions": "How strong is the evidence?",
                "criteria": ["weak", "moderate", "strong", "very strong"]},
    "effect": {"type": "noul", "instructions": "Was the drug effective?"},
}
STATE = "patients were randomised to drug or placebo and the drug reduced risk"


def _export(tmp_path, arch, frozen=False):
    from safetensors.torch import save_file
    from transformers import BertConfig, BertModel, BertTokenizerFast

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", ":", ",", "."] + WORDS + [str(i) for i in range(10)]
    (tmp_path / "tokenizer").mkdir()
    (tmp_path / "tokenizer" / "vocab.txt").write_text("\n".join(vocab))
    tok = BertTokenizerFast(vocab_file=str(tmp_path / "tokenizer" / "vocab.txt"))
    tok.save_pretrained(tmp_path / "tokenizer")
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=128, max_position_embeddings=512)
    torch.manual_seed(0)
    model = S1Model(BertModel(cfg), arch, frozen_prior=frozen)
    cfg.save_pretrained(tmp_path / "encoder")
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(tmp_path / "model.safetensors"))
    temps = {"type": {"choice": 1.3, "score": 0.9, "noul": 1.1}, "bucket": {"choice:3-5": 1.5}}
    (tmp_path / "s1_config.json").write_text(json.dumps(dict(
        arch=arch, pooling="mean", max_len=128, head_max_len=64, bi_state_len=64, temperatures=temps)))
    return tmp_path


@pytest.mark.parametrize("arch,frozen", [("C", False), ("PFR", False), ("PFR", True), ("B", False)])
def test_predict_shapes_and_probabilities(tmp_path, arch, frozen):
    m = load(_export(tmp_path, arch, frozen), device="cpu")
    out = m.predict(STATE, REQ)["answers"]
    assert set(out) == set(REQ)
    d = out["design"]
    assert d["choice"] in REQ["design"]["criteria"]
    assert math.isclose(sum(d["probabilities"].values()), 1.0, abs_tol=1e-3)
    assert d["answer_confidence"] == max(d["probabilities"].values())
    s = out["quality"]
    assert 0.0 <= s["score"] <= 3.0 and len(s["probabilities"]) == 4
    assert 0.0 <= out["effect"]["noul"] <= 1.0


def test_batch_matches_single(tmp_path):
    m = load(_export(tmp_path, "PFR"), device="cpu")
    reqs = [(STATE, REQ), ("a case report of drug effect", REQ), ({"title": "cohort study"}, REQ)]
    batch = m.predict_batch(reqs, batch_size=4)
    for (st, q), b in zip(reqs, batch):
        single = m.predict(st, q)
        for qid in q:
            for k, v in single["answers"][qid].items():
                if isinstance(v, float):
                    assert math.isclose(v, b["answers"][qid][k], abs_tol=2e-4)


def test_temperature_lookup_is_applied(tmp_path):
    m = load(_export(tmp_path, "C"), device="cpu")
    (z, it, _), = m.logits([(STATE, {"design": REQ["design"]})])[0].values()
    p = torch.softmax(z / 1.5, -1)          # bucket choice:3-5 overrides the type temperature
    got = m.predict(STATE, {"design": REQ["design"]})["answers"]["design"]["probabilities"]
    assert all(math.isclose(float(a), b, abs_tol=1e-4) for a, b in zip(p, got.values()))


def test_truncation_flag(tmp_path):
    m = load(_export(tmp_path, "C"), device="cpu")
    out = m.predict(" ".join(["patients"] * 400), {"effect": REQ["effect"]})["answers"]["effect"]
    assert out.get("truncated") is True


def test_schema_validation(tmp_path):
    m = load(_export(tmp_path, "C"), device="cpu")
    with pytest.raises(ValueError):
        m.predict(STATE, {"bad": {"type": "choice", "instructions": "x", "criteria": {"only": "one"}}})
    with pytest.raises(ValueError):
        m.predict(STATE, {"bad": {"type": "multi", "instructions": "x"}})


def test_rendering():
    assert option_keys(REQ["quality"]) == ["0", "1", "2", "3"]
    assert render_options(REQ["design"])[0] == "rct: randomised trial"
    assert render_options(REQ["effect"])[1].startswith("true: ")
