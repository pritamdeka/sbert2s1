# s1bio on Kelvin2: SBERT → System One decision models

How the paper's campaign was run on the Kelvin2 cluster (NI-HPC). Conventions:
`k2-gpu-amd`, one MI300X per job, the ROCm PyTorch module plus a clean venv (`S1_VENV`), pinned model commits, HF offline in compute jobs, resumable immutable run
directories, and smoke → full gating.

## What is in the bundle

| Path | Content |
|---|---|
| `prepared/` | 15 public BioDecide tasks + MEDLINE-S1 + BEIR SciFact/NFCorpus, already converted (built locally; see `prepared/manifest.json` for counts and SHA-256) |
| `data_build/` | the converters. `build_clinical.py` is the only one you run on Kelvin2 |
| `s1/` | the package: schema, pre-tokenisation, models (Z/B/C/PFR), objectives, trainer, evaluation, LLM baseline, queue/packer |
| `scripts/`, `slurm/` | staging, enqueueing, the packed GPU worker |
| `tests/test_cpu.py` | end-to-end CPU test (passes locally; run it on the login node too) |

## 0. Upload and check (login node)

```bash
cd $SCRATCH
unzip s1bio_hpc.zip && cd s1bio_hpc
export S1_ROOT="$PWD"
bash scripts/check_environment.sh          # imports only; installs nothing
```

## 1. Stage models (login node, needs internet; about 3 GB new, the LLMs are already cached)

```bash
bash scripts/stage_models.sh               # pins commits into configs/models.lock.json
```

## 2. Clinical track (PhysioNet data; you run this, it prints counts only)

Copy the three zips from `databases/` (MedNLI, MIMIC-III trial questions, MIMIC-IV name de-id) to a
private directory on scratch (`chmod 700`), then:

```bash
bash scripts/build_clinical.sh $SCRATCH/physionet_zips
```

Run this **before** step 4. Every run's configuration lists its eval splits, so the clinical tests
must exist when the grid is enqueued. Credentialed data never leaves Kelvin2. The LLM baselines on
these splits run locally on the GPU, and the Jev API is never used on them.

## 3. Smoke (about 1 h on 1 GPU), then read it

```bash
python tests/test_cpu.py                   # optional on the login node (~10 min CPU)
bash scripts/submit.sh smoke
# when done:
S1_QUEUE=$PWD/queue_smoke python -m s1.queue status
```

The smoke queue runs:

- pre-tokenisation for two families;
- PFR on S-PubMedBert with a **forced TERM at step 100**, so it must requeue and resume;
- C and Z;
- Laya zero-shot and Laya fine-tuning, which checks that ModernBERT runs on this ROCm stack;
- Qwen3.8-27B option scoring on PubMedQA.

**Gate:** everything is in `done/`; `runs/smoke/*/metrics_final.json` exist; the PFR log shows
`resumed at step 100`. If only the Laya rows fail, continue: Laya is a baseline, and its failure is
reported as "unsupported on this stack".

## 4. Packing calibration (about 1 h), then the full campaign

```bash
bash scripts/submit.sh calibrate           # writes PACKING_CALIBRATION.md (1..8 concurrent runs)
export S1_MAX_SLOTS=<recommended value from that file>
bash scripts/submit.sh full 3              # enqueue the grid + 3 packed 3-day workers
python -m s1.queue status                  # any time
```

The grid is in `scripts/enqueue_v1.py`. P0 is the core paper (S-PubMedBert pair, all zero-shot rows,
Laya zero-shot, Qwen). P1 covers the other encoders, B, the RLCD decomposition, Laya fine-tuning,
learning curves and Gemma. P2 adds seeds 1–2. P3 adds the clinical variant, frozen prior, extra
learning-curve seeds and latency.

A worker that reaches its walltime checkpoints every child and requeues it; `bash scripts/submit.sh
more 3` continues. If a node dies, `python -m s1.queue requeue-stale` returns its tasks (it checks
`squeue`). Failed tasks keep their log tail: `python -m s1.queue status`.

## 5. Bring results back

Only metrics and prediction logits are needed for the paper. Prediction rows carry neutral IDs, not
text:

```bash
tar czf s1_results.tgz runs/*/*/metrics_*.json runs/*/*/*/metrics_*.json runs/*/*/*/*/metrics_*.json \
    runs/**/temps.json runs/**/state.json runs/**/train_log.jsonl runs/llm runs/latency PACKING_CALIBRATION.md
```

The clinical prediction files contain only IDs and logits. Aggregate tables and CIs are computed
offline from these files.
