# Patch 8 results (S-PubMedBERT, 3 seeds per cell, original splits)

All 15 trainings and the gradient probe finished. Historical cells reproduce the paper exactly
(PFR: CE 68.3, PG 65.3, Reparam 68.5; C+PG 67.7). Files: `per_run.csv`, `summary.md`,
`contrasts_patch8.json/.txt` (10,000 document-clustered paired draws; Holm within RQ2 and within RQ3,
across both tracks), probe in `results/s1_results_patch8/runs/probe/gradvar/grad_probe.json`.

## Head x objective (mean ± sd, x100 except NLL/MAE)

| Head | Objective | Seen acc | Held-out | ECE raw | ECE TS | NLL TS | MAE score | Seen perm flip % |
|---|---|---|---|---|---|---|---|---|
| PFR | CE | 68.3 ± 0.4 | 20.8 ± 1.6 | 9.3 | 4.0 | 0.510 | 0.588 | 0.4 |
| PFR | Proper | 68.5 ± 0.2 | 22.6 ± 1.9 | 9.1 | 4.4 | 0.510 | 0.582 | 0.2 |
| PFR | PG (released) | 65.3 ± 1.5 | 23.6 ± 1.5 | 7.0 | 3.8 | 0.548 | 0.638 | 0.4 |
| PFR | PG w/o CE | 65.5 ± 2.0 | 20.3 ± 3.3 | 6.6 | 4.0 | 0.543 | 0.646 | 0.2 |
| PFR | **PG-LOO (new)** | 67.8 ± 0.6 | 23.8 ± 5.7 | 7.5 | 3.9 | 0.518 | 0.605 | 0.4 |
| PFR | Reparam | 68.5 ± 0.5 | 21.9 ± 2.0 | 9.0 | 3.7 | 0.508 | 0.587 | 0.4 |
| C | **CE (new)** | 70.2 ± 0.4 | 31.0 ± 1.7 | 7.4 | 3.8 | 0.491 | 0.639 | 2.9 |
| C | **Proper (new)** | 70.7 ± 0.3 | 28.8 ± 0.4 | 7.6 | 3.9 | 0.493 | 0.627 | 2.6 |
| C | PG (released) | 67.7 ± 0.9 | 32.8 ± 1.5 | 6.5 | 3.9 | 0.527 | 0.746 | 3.2 |
| C | **PG-LOO (new)** | 69.2 ± 0.6 | 30.0 ± 2.1 | 6.6 | 3.7 | 0.509 | 0.665 | 3.3 |
| C | **Reparam (new)** | 69.8 ± 1.6 | 25.8 ± 1.5 | 7.2 | 3.6 | 0.498 | 0.668 | 2.4 |

Perm flip here = mean over seen tasks only (the paper's robustness table aggregates differently).
Retrieval retention (SciFact nDCG@10, seed 0): C+PG 43.3, PFR+PG 35.6, C+CE 24.6.

## RQ2: the head effect holds under every objective

C − PFR, seen: CE +1.9 [0.8, 3.1] (Holm .010), Proper +2.2 (.002), PG +2.4 (.002), LOO +1.5 (.046),
Reparam +1.3 [0.1, 2.4] (.067). Held-out: +10.2, +6.2, +9.2, +6.2 (all Holm < .02), Reparam +3.8 (.082).
=> The earlier objection that C > PFR was confounded with the objective is resolved: C is the stronger
head under matched objectives. PFR keeps its specific benefits: ~10x lower option-order sensitivity,
lower score MAE, the MedCPT rescue.

## RQ3: the released recipe's deficit comes mainly from its normalisation, not from using REINFORCE

* Unbiased LOO PG recovers most of the gap: LOO − PG = +2.45 (PFR, Holm .005) and +1.53 (C, Holm .005).
* LOO vs pathwise Reparam: −0.77 (PFR, Holm .25) and −0.58 (C, Holm .65), not significant after Holm.
  LOO vs CE: −0.50 (PFR, n.s.), −0.97 (C, Holm .026).
* Proper − CE: +0.2 (PFR) and +0.5 (C), both n.s. -> the spherical/RPS terms add nothing measurable.
* Held-out ordering is noisier (3 tasks) and does not follow the seen-task ranking; e.g. C+Reparam is
  7.0 below C+PG on held-out (Holm .005). Do not generalise the RQ3 ranking to held-out tasks.

Matched gradient probe (same checkpoints, same 40 batches; consistent across all 3 checkpoints):

| Estimator | Slope vs true smoothed gradient (sigma 0.4 / 0.25 / 0.1) | Score:CE weight | Per-item SNR at matched scale |
|---|---|---|---|
| released PG | 3.6 / 5.7-6.1 / 14.3-15.1 | 4.7-5.0 -> 18.9-20.0 | ~0.85 |
| same-sample mean, unnormalised | 0.75 (= (G-1)/G) | 1.0 | ~0.81 |
| LOO PG (new) | 1.00 | 1.33 | ~0.79 |
| pathwise Reparam | 1.00 | 1.32 | 85 -> 1,300 |

Training logs: the global gradient norm exceeds the clip threshold (1.0) at EVERY step for EVERY objective
(medians: CE 6-7, Reparam/LOO 14-18, released PG 50-88, rising to 80-129 in the second half). Clipping
therefore fixes the step length; objectives differ only in direction. The released normalisation inflates
the high-variance score term to 5-20x the CE term as sigma anneals, so its updates are dominated by noise.
The pathwise estimator is 100-1,500x less noisy than any REINFORCE variant at matched scale, yet LOO trains
almost as well -> estimator variance is a secondary factor; the scale/normalisation is the primary one.

## Practical consequences

* Best S-PubMedBERT recipe: C + Proper (70.7) or C + CE (70.2), +2.5-3.0 over the historical C + PG (67.7).
* Release model: only C+CE checkpoints were saved; best seed on seen tasks is `rq3c/ce/s2` (70.5).
* All encoders in the main table (and Laya-large FT) were trained with the released PG recipe, so their
  absolute numbers are likely ~2-3 points below a CE recipe. RQ1 comparisons stay valid (matched recipe),
  but say so explicitly.
* Holm families above are patch-8 only; merge with the paper's RQ2/RQ3 families when regenerating Table 6.
