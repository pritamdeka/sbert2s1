# Public-data integrity audit

Counts are records/states, not individual questions. Historical results are not retrained.

| Task | Split | Records | Exact inputs in train | Same text in train | Excluded from test sensitivity |
|---|---|---:|---:|---:|---:|
| ade | calib | 400 | 0 | 0 | 0 |
| ade | test | 2089 | 0 | 0 | 0 |
| biosses | test | 100 | 0 | 0 | 0 |
| ddi | calib | 400 | 9 | 9 | 0 |
| ddi | test | 7039 | 0 | 0 | 15 |
| druglib | calib | 310 | 7 | 7 | 0 |
| druglib | test | 1036 | 13 | 13 | 19 |
| healthver | calib | 517 | 0 | 55 | 0 |
| healthver | test | 1823 | 0 | 1722 | 129 |
| hoc | calib | 129 | 0 | 0 | 0 |
| hoc | test | 371 | 0 | 0 | 0 |
| medline_s1 | calib | 400 | 0 | 0 | 0 |
| medline_s1 | test | 5000 | 0 | 0 | 0 |
| medmcqa | calib | 400 | 0 | 0 | 0 |
| medmcqa | test | 2816 | 0 | 0 | 0 |
| medqa | calib | 400 | 0 | 0 | 0 |
| medqa | test | 1273 | 0 | 0 | 0 |
| mmlu_med | test | 1089 | 0 | 0 | 78 |
| mtsamples | test | 370 | 0 | 0 | 20 |
| pubhealth | test | 1231 | 0 | 0 | 0 |
| pubmedqa | calib | 50 | 0 | 0 | 0 |
| pubmedqa | test | 500 | 0 | 0 | 0 |
| pubmedqa_art | calib | 400 | 0 | 0 | 0 |
| scifact | calib | 88 | 0 | 57 | 0 |
| scifact | test | 321 | 2 | 190 | 2 |
| typed_decisions | test | 400 | 0 | 0 | 0 |

Same-text overlap with different questions is reported separately from exact input duplication.
Sensitivity excludes exact inputs in train/calibration and repeated test inputs. Existing temperatures remain unchanged.
prepared_clean is for NEW training and calibration; applying it cannot retroactively repair historical checkpoints.
Document identities normalise whitespace; this is not a semantic near-duplicate or patient-level audit.
