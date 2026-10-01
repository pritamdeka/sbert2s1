# Patch 8: revision controls (RQ2 head x objective, RQ3 matched estimator, release model)

Applies on top of the Kelvin2 campaign at `$SCRATCH/s1bio_hpc` (patches 1-7).
All new runs use the original `prepared/` data, the existing tokenizer cache and the exact base
configuration of the historical grid, so every number slots straight into the existing tables.
No existing run is modified or re-run.

## What the patch changes

| File | Change |
|---|---|
| `s1/objectives.py` | new `rlcd_pg_loo` (leave-one-out baseline, no std normalisation = unbiased, same scale as `rlcd_reparam`); optional `w_score`. `ce`, `proper`, `rlcd_pg`, `rlcd_reparam` are bit-identical to patch 7 |
| `s1/train.py` | passes `w_score` (read with `cfg.get`, default 1.0). `DEFAULTS` untouched, so existing config hashes are unchanged |
| `s1/temperature_lookup.py` (new), `s1/calibrate.py` | one type-aware temperature lookup; NumPy/PyTorch integer codes work, invalid types raise |
| `s1/grad_probe.py` (new) | matched gradient-estimator probe on saved checkpoints (same batches, same scale) |
| `scripts/enqueue_patch8.py` (new) | the patch-8 grid |
| `scripts/check_patch8.py`, `tests/test_patch8.py` (new) | pre-flight checks and tiny end-to-end CPU test |
| `scripts/patch8.sh` (new) | check / test / enqueue / submit / status / collect |

## The grid

| Priority | Run IDs | Runs | Answers |
|---|---|---:|---|
| P0 | `rq3/pg_loo/s{0,1,2}` PFR + unbiased LOO PG (+CE) | 3 | Is RLCD's deficit the estimator, or its bias/normalisation? Compare with `rq3/reparam` and `main/spubmedbert/PFR` |
| P0 | `rq3c/ce/s{0,1,2}` C + CE (`save_model`; s0 also retrieval retention) | 3 | Is C > PFR a head effect or an objective effect? (C+PG = `main/spubmedbert/C`). Also the release candidates |
| P0 | `rq3c/reparam/s{0,1,2}` C + Reparam | 3 | same |
| P0 | `probe/gradvar` (no training) | 1 | Estimator variance at the SAME checkpoints and batches, scale-matched |
| P1 | `rq3c/proper/s*`, `rq3c/pg_loo/s*` | 6 | Completes the 2 head x 5 objective table |
| P2 (optional) | `lc/{modernbert,mbembed,gtemb,bcmb,bcmbembed}/{C,PFR}/f0.1/s{1,2}`, `.../f0.02/s{0,1,2}` | 50 | 3-seed learning curves for the extension pairs |

Estimated cost (from the measured wall times of the earlier campaign, under packing; evaluation on all
19 test splits dominates, so the learning-curve runs are NOT short):

| Enqueued | Task-hours | 2 workers | 3 workers |
|---|---:|---:|---:|
| `enqueue 1` (P0+P1, 15 runs + probe) | ~30 | ~4 h | ~3 h |
| `enqueue 2` (+ 50 learning-curve runs) | ~120 | ~12-15 h | ~8-10 h |

For `enqueue 2` submit with a 24 h walltime (`bash scripts/patch8.sh submit 3 24:00:00`); a worker that
runs out of walltime checkpoints and requeues its tasks, and `submit 1` continues.

## Steps on Kelvin2

```bash
# 0. Upload (from your PC):  scp s1bio_hpc_patch8.zip <user>@login.kelvin.alces.network:$SCRATCH/
ssh <user>@login.kelvin.alces.network
cd $SCRATCH
cp -r s1bio_hpc/s1 s1bio_hpc/s1_backup_before_patch8        # optional safety copy
unzip -o s1bio_hpc_patch8.zip                              # overwrites only the files listed above
cd s1bio_hpc
export S1_ROOT="$PWD"

# 1. Make sure no worker from earlier campaigns is still running
squeue -u $USER

# 2. Pre-flight (login node, CPU): must end with "0 failure(s)"
bash scripts/patch8.sh check
bash scripts/patch8.sh test                                # optional, ~5 min CPU

# 3. Enqueue: P0+P1 (recommended). Use 0 for must-have only, 2 to add the learning-curve seeds.
bash scripts/patch8.sh enqueue 1

# 4. Start GPU workers (2 workers, 12 h walltime; they exit by themselves when the queue drains)
bash scripts/patch8.sh submit 2 12:00:00

# 5. Monitor
bash scripts/patch8.sh status
tail -f logs/s1-p8-*.out
tail -n 30 logs/tasks/P0_rq3c__ce__s0.log

# 6. When everything is done, read the gradient probe directly
cat logs/tasks/P0_probe__gradvar.log

# 7. Bring the results home (no weights; clinical per-item predictions excluded)
bash scripts/patch8.sh collect
# then on your PC:  scp <user>@login.kelvin.alces.network:$SCRATCH/s1bio_hpc/s1_results_patch8.tgz .
```

Optional release export, once the runs are analysed (choose the best `rq3c/ce` seed on the seen tasks; all three were trained on public data only):

```bash
( source hpc/env.sh; python -c "from s1.predict import export; export('runs/rq3c/ce/s0', 'exports/S1-PubMedBERT', laya=True)" )
```

The derived model inherits S-PubMedBERT-MS-MARCO's CC-BY-NC licence.

## If something goes wrong

- A worker hits its walltime: tasks checkpoint and requeue; `bash scripts/patch8.sh submit 1`.
- A node died: `( source hpc/env.sh; python -m s1.queue requeue-stale )`.
- A task failed twice: `python -m s1.queue status` shows its log tail; the task file is in `queue/failed/`.
- `probe/gradvar` reports `missing: model.pt` for `main/spubmedbert/PFR/s0`: that checkpoint was deleted;
  the probe still runs on the two new checkpoints.
- Do NOT upload `sbert2s1_revision_v2/s1bio_hpc` to Kelvin2: its `train.py` adds `w_score` to `DEFAULTS`,
  which changes the config hash of every existing run.
