# Enqueue 2 results: three-seed learning curves for the extension pairs

Runs: `lc/{modernbert,mbembed,gtemb,bcmb,bcmbembed}/{C,PFR}/f0.1/s{1,2}` and `.../f0.02/s{0,1,2}`.

- **Completion:** all 50 runs finished, each with exit code 0 and `metrics_final.json` present.
- **Configs:** each config differs from the existing seed-0 10% run of its cell only in `seed`, or only in
  `frac` for the 2% runs. The runs are therefore directly comparable with the earlier cells.
- **Compute:** 4.2 wall-clock hours on four MI300X workers, 72 task-hours under packing.
- **Other folders:** the `probe`, `rq3` and `rq3c` folders in this results directory are byte-identical
  copies of patch 8.
- **Where they now sit:** the runs were copied into `results/s1_results_public/runs/lc/`, so
  `metrics_long.csv` now covers 254 runs.
- **Bootstrap:** RQ1 contrasts were re-run with 10,000 document-clustered paired draws. The RQ1 Holm
  family grew from 48 to 60 tests, so the smallest Holm p is now .012. The RQ2, RQ3 and baseline
  families are unchanged.

## Seen-task accuracy (chance-normalised, ×100, mean over 3 seeds)

| Pair (child − parent) | C 2% | C 10% | C full | PFR 2% | PFR 10% | PFR full |
|---|---:|---:|---:|---:|---:|---:|
| S-PubMedBERT − PubMedBERT | −0.2 | **+1.9** | +0.4 | −0.2 | −0.2 | **−1.6** |
| all-MPNet − MPNet | −0.8 | −1.0 | +0.0 | **+2.6** | **+7.7** | **+4.5** |
| Nomic-embed − ModernBERT | −1.0 | **−2.5** | −1.0 | **+9.2** | **+5.3** | **+2.7** |
| GTE-ModernBERT − ModernBERT | +1.5 | **−3.1** | **−2.5** | **+7.4** | **+4.7** | **+2.9** |
| BioClinical-MB-emb − BioClinical-MB | **−8.5** | **−3.7** | −0.2 | **+4.0** | +1.2 | +0.8 |

Bold means Holm-significant (p ≤ .014). The 2% cells for the last three pairs and the 10% cells as
three-seed results are new; previously those 10% cells had a single seed.

On held-out tasks, PFR gains for Nomic and GTE reach +15.6 and +15.5 at 2%, and +16.3 and +13.5 at 10%
(all Holm .012).

## What changes in the paper

- **The RQ1 conclusion becomes sharper:**
  - **PFR:** across 15 comparisons (5 pairs × 3 data sizes), retrieval training significantly helps
    PFR in 10 and hurts it in 1 (S-PubMedBERT, full data). The advantage grows as labelled data shrink.
  - **C:** retrieval training helps in 1 comparison (S-PubMedBERT, 10%) and hurts in 5. The
    ModernBERT-family retrievers are 2.5–3.7 points behind their parents at 10%, and BioClinical-MB is
    8.5 points behind at 2%.
- **Corrections to single-seed numbers in the previous draft:**
  - C 10% GTE moved from −1.7 (n.s.) to −3.1 (significant).
  - C 10% Nomic moved from −2.0 (n.s.) to −2.5 (significant).
  - Held-out C 10% Nomic moved from +12.5 to +6.6.
  - Held-out PFR 10% BioClinical moved from +3.3 to −3.1 (n.s. both times).
- **Edits made:**
  - Abstract (199 words), Introduction (Findings) and §6.1 RQ1.
  - Figure 1 (now three seeds everywhere) and its caption.
  - Table 2 (10% column now mean ± sd) and Table 8 (all RQ1 contrasts are 3/3 seeds).
  - Setup (learning curves), Discussion (practical advice) and Limitations (the one-seed caveat is
    removed).
  - Appendix E (RQ1 family has 60 tests) and Appendix K (compute).
- **Unchanged:** RQ2, RQ3, the LLM and routing results, the probe, and every full-data number.
