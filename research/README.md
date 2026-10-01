# Research code

This folder holds the code for the paper *From Retrieval to Typed Decisions: Calibrated System One Models
from Biomedical Sentence Encoders*. It runs on any machine with PyTorch: one GPU (CUDA or ROCm) is enough
for every encoder experiment, and everything can be tested on CPU. Run all commands from this folder.
For inference alone, `../inference/sbert2s1.py` is enough.

| Path | Content |
|---|---|
| `s1/` | Python package: schema and pre-tokenisation, the Z / B / C / PFR models, objectives (`ce`, `proper`, `rlcd_pg`, `rlcd_pg_loo`, `rlcd_reparam`), trainer with exact resume, temperature fitting, evaluation (option-order probe, long-state strategies, retrieval retention), LLM option scorer, gradient probe, latency benchmark, export |
| `experiments.py` | every run in the paper as a list, with a local runner and a command printer for any scheduler |
| `data_build/` | builders for BioDecide (public tasks), MEDLINE-S1 and the credentialed clinical track |
| `configs/models.json` | encoders and LLMs: Hub ids, pooling, query/document prefixes |
| `tests/` | CPU tests on tiny random encoders |
| `analysis/`, `paper/scripts/`, `rebuild.py` | metrics collection, bootstrap contrasts, tables and figures |
| `results/` | aggregate metrics of every run in the paper (`metrics_long.csv.gz`) and the bootstrap contrasts (`contrasts.json`) |
| `model_card/` | the Hugging Face model card and upload script |

## 1. Install

```bash
pip install -r requirements.txt     # torch, transformers, safetensors, huggingface_hub, numpy, pandas, matplotlib, lxml
```

Install the PyTorch build that matches your hardware first (see pytorch.org). Models download from the
Hugging Face Hub on first use.

## 2. Data

```bash
python data_build/build_public.py --out prepared                 # public BioDecide tasks
huggingface-cli download pritamdeka/MEDLINE-S1 --repo-type dataset --local-dir medline_s1_release
python data_build/hydrate_medline.py --data medline_s1_release/data --out prepared/medline_s1
python data_build/verify_build.py                                # check the build against the paper's
```

Each task becomes `prepared/<task>/{train,calib,dev,test}.jsonl`.

**MEDLINE-S1** is released on the Hugging Face Hub with PMIDs, questions and labels but without the
abstracts. `hydrate_medline.py` fetches the titles and abstracts from PubMed E-utilities (pass `--email`,
and `--api-key` if you have an NCBI key), writes the records in the format above, and checks each text
against its hash. A few citations may have been revised since the paper's build; the script reports
them. With the 2026 baseline files listed in the release's `manifest.json`, `--baseline-dir` gives an
exact copy. `build_medline.py` is the script that built the set. NLM replaces the baseline every year,
so running it later gives a new sample of the same design, not the paper's set.

**Checking a build.** `docs/prepared_manifest.json` lists the counts and file hashes of the paper's
build, and `docs/biodecide_records.tsv.gz` one hash per record with its source identifiers.
`verify_build.py` reports, for each split that differs, how many records changed text, changed labels,
or are missing, which usually shows what changed upstream in a source dataset.

**Clinical track (optional, credentialed).** Obtain MedNLI, the MIMIC-III trial-eligibility questions and
the MIMIC-IV de-identification data from PhysioNet under your own data use agreement, then run
`python data_build/build_clinical.py --src <folder with the zips> --out prepared`. The builder prints counts
only. Process these data only where your agreement allows, and never send them to a hosted model or API.
Without them, `experiments.py` simply leaves out the clinical evaluations and the clinical training run.

## 3. Test the installation

```bash
python tests/test_cpu.py                  # conversions, objectives, calibration, evaluation, resume, LLM scorer (CPU, minutes)
python tests/test_clinical_fixtures.py    # clinical builder on synthetic files (no real data)
python experiments.py run --smoke --match main/spubmedbert/C/s0    # one tiny end-to-end run
```

## 4. Run experiments

```bash
python experiments.py list                                 # all runs, by group
python experiments.py run --match main/spubmedbert/C/s0    # one run
python experiments.py run --group rq3 rq3c                 # the head x objective grid
python experiments.py commands --group lc > jobs.txt       # commands only, for your own scheduler
```

Groups: `zeroshot`, `main` (every encoder with C and PFR, three seeds), `bi`, `rq3` and `rq3c` (head x
objective on S-PubMedBERT-MS-MARCO), `frozen`, `clinical`, `laya`, `lc` (learning curves at 2% and 10%),
`llm`, `probe` and `latency`.

`run` executes experiments one after another on the local device, pre-tokenises each tokenizer family
once, and skips runs that have finished. Training checkpoints regularly and resumes after an interruption,
so the same command can simply be repeated. `commands` writes each run's config and prints one shell
command per line (pre-tokenisation first), ready for a job array, SLURM, GNU parallel or a shell loop.

A single run is `python -m s1.train --run-dir <dir> --config <config.json>`. The config overrides
`DEFAULTS` in `s1/train.py`, which are the paper's settings: 8,000 steps, batch 32, AdamW, bf16 autocast,
gradient clipping at norm 1, and square-root task sampling. Training fits the temperatures, evaluates
every listed split and writes `metrics_final.json`. A run directory belongs to one config: change the
config, change the directory.

To share one GPU between several runs, set `S1_VRAM_GB` per process to cap its memory. The LLM
baselines need a GPU with enough memory for a 27-31B model in bf16.

## 5. Tables and figures

```bash
python rebuild.py                 # reads runs/, writes paper/tables and paper/figures
```

The aggregate metrics behind the paper are in `results/metrics_long.csv.gz` (one row per run, split and
question group). The per-item predictions are not released, because some come from credentialed clinical
data.

## 6. Export a model

```bash
python -c "from s1.predict import export; export('runs/rq3c/ce/s0', 'exports/my-model', laya=True)"
```

This writes the layout that `inference/sbert2s1.py` loads (`s1_config.json`, `model.safetensors`, `encoder/`,
`tokenizer/`) with the fitted temperatures, plus a Laya-compatible copy in `laya/`. Only runs with
`save_model` in their config keep weights. `model_card/upload_to_hf.sh` uploads an export and refuses
models trained on clinical data.
