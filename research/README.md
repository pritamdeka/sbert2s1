# Research code

This folder holds the code behind the paper *From Retrieval to Typed Decisions: Calibrated System One
Models from Biomedical Sentence Encoders*. It is the code that ran the experiments, unchanged apart from
removing cluster-specific paths. Run everything from this folder; the `s1` package is imported from here
(`PYTHONPATH=.`). It is separate from the pip package and is not installed by it.

| Path | Content |
|---|---|
| `s1/` | schema and pre-tokenisation, the Z / B / C / PFR models, objectives (`ce`, `proper`, `rlcd_pg`, `rlcd_pg_loo`, `rlcd_reparam`), trainer with exact resume, temperature fitting, evaluation (permutation probe, long-state strategies, retrieval retention), LLM option scorer, gradient probe, export, file queue and GPU packer |
| `data_build/` | builders for BioDecide (public tasks), MEDLINE-S1 and the credentialed clinical track |
| `scripts/`, `slurm/`, `hpc/` | model staging, campaign enqueueing (`enqueue_v1.py`, `enqueue_patch8.py`), packed SLURM worker, environment |
| `configs/models.json` | encoders, Hub ids, pooling and query/document prefixes |
| `tests/` | CPU end-to-end tests (tiny random encoders) |
| `analysis/`, `paper/scripts/` | bootstrap contrasts, tables, figures, data audit |
| `results/` | aggregate metrics of every run (`metrics_long.csv.gz`) and the 96 bootstrap contrasts (`contrasts.json`) |
| `model_card/` | the Hugging Face model card and upload script for `pritamdeka/S1-PubMedBERT` |
| `docs/` | how the campaign was run on the Kelvin2 cluster, and the patch-8 controls |

## 1. Environment

```bash
pip install -r requirements.txt          # torch, transformers, safetensors, huggingface_hub, numpy, pandas, matplotlib, lxml
export PYTHONPATH=$PWD
```

## 2. Data

```bash
python data_build/build_public.py --out prepared                     # downloads and converts the public tasks
python data_build/build_medline.py --out prepared --train-files 12 --test-files 3   # NLM baseline files
```

Each task becomes `prepared/<task>/{train,calib,dev,test}.jsonl`. The calibration slices are carved
from training data before training. `docs/prepared_manifest.json` lists the counts and SHA-256 hashes
we obtained, so you can check that your build matches.

**Clinical track (credentialed).** Obtain MedNLI, the MIMIC-III trial-eligibility questions and the
MIMIC-IV de-identification data from PhysioNet under your own data use agreement, then run
`python data_build/build_clinical.py --src <dir with the zips> --out prepared`. The builder prints counts
only and writes owner-only directories. Keep these data on approved infrastructure, and never send them
to a hosted model or API. The LLM baselines in the paper ran locally on the cluster for this reason.

## 3. Tests

```bash
python tests/test_cpu.py         # all four conversions and objectives, calibration, evaluation, resume, queue (minutes, CPU)
python tests/test_patch8.py      # leave-one-out objective, CE/Reparam controls, gradient probe
python tests/test_clinical_fixtures.py   # clinical builder on synthetic zips (no real data)
```

`test_cpu.py` uses a small slice of `prepared/` and the PubMedBERT tokenizer.

## 4. Training and evaluation

Models are staged once (`python -m s1.models_io --stage spubmedbert ...`, needs internet; pins the
Hub commit in `configs/models.lock.json`), and each tokenizer family is pre-tokenised once. A single run
then takes a JSON config whose keys override `DEFAULTS` in `s1/train.py`, which are the paper's
settings: 8,000 steps, batch 32, AdamW, bf16 autocast, clipping at norm 1, and square-root task
sampling. For example, S-PubMedBERT-MS-MARCO with a cross head and cross-entropy:

```bash
python -m s1.models_io --stage spubmedbert
python -m s1.data --family pubmedbert
cat > ce.json <<'JSON'
{"encoder": "spubmedbert", "arch": "C", "objective": "ce", "seed": 0, "save_model": true,
 "train_tasks": ["pubmedqa", "pubmedqa_art", "scifact", "healthver", "ddi", "hoc", "ade", "druglib",
                 "medqa", "medmcqa", "medline_s1"],
 "eval_splits": ["pubmedqa/test", "scifact/test", "pubhealth/test", "biosses/test", "mtsamples/test"]}
JSON
python -m s1.train --run-dir runs/example --config ce.json
```

Training fits the temperatures on the calibration splits, evaluates every listed split, and writes
`metrics_final.json`. The run directory is immutable: a changed config needs a new directory. A run
interrupted by SIGTERM resumes exactly from its last checkpoint.

The full campaign is a file queue served by packed GPU workers:

```bash
bash scripts/submit.sh smoke                 # 1 GPU, about 1 hour
bash scripts/submit.sh full 3                # enqueue the grid + 3 workers
python -m s1.queue status
bash scripts/patch8.sh enqueue 1             # head x objective controls and gradient probe
```

`docs/README_KELVIN.md` and `docs/README_PATCH8.md` describe these stages in detail. The main grid
took 11.9 wall-clock hours on three MI300X GPUs, and the patch-8 controls took another 1.3 hours.

## 5. Tables and figures

`rebuild.py` regenerates the analysis, tables and figures from a directory of run outputs
(`results/s1_results_public/runs`). Those run directories hold per-item predictions and are not
released, because some come from credentialed clinical data. The aggregate metrics they produce are
in `results/metrics_long.csv.gz`, with one row per run, split and question group, and 65 metric
columns. The bootstrap results are in `results/contrasts.json`.

## 6. Exporting a model

```bash
python -c "from s1.predict import export; export('runs/example', 'exports/my-model', laya=True)"
```

This command writes the layout that the pip package loads (`s1_config.json`, `model.safetensors`,
`encoder/` and `tokenizer/`), with fitted temperatures, plus a Laya-compatible copy in `laya/`.
`sbert2s1.load('exports/my-model')` then works offline. `model_card/` holds the published card and
the upload script, which refuses to upload a model whose training tasks include clinical data.
