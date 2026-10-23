# PRDA: person-referenced radar activity adaptation

Saved predictions and derived results for **Adapt to the Person, Not the Place: Person-Referenced Deployment Adaptation of Radar-Based Activity Recognition for Older Adults**.

The study evaluates resident-referenced adaptation at two care organisations in the public [Radar signatures of human activities dataset](https://doi.org/10.5525/gla.researchdata.848). The development site has 298 recordings from 20 people; the confirmation site has 289 recordings from 20 people. Results use five training seeds: 7, 42, 123, 3407 and 2026.

## Contents

- `data/development/` and `data/confirmation/`: per-recording prediction arrays (`PREDICTIONS.npz`), full derived results (`RESULT.json`) and evaluation summaries (`REPORT.json`).
- `data/table_and_prose_values.json`: numeric source values used in the manuscript tables and prose.
- `data/recomputed_metrics.csv`: full-enrolment and TTA metrics recalculated from the saved predictions.
- `data/verify_predictions.py`: independent NumPy verification of those metrics.
- `data/metadata.json`: class order, seeds, sites and source-dataset DOI.

## Verify the saved predictions

With Python 3 and NumPy installed, run from this repository's root:

```sh
python data/verify_predictions.py
```

The script recomputes macro-F1, recording accuracy, person-mean slowest-quartile accuracy, class recalls, per-person accuracies and the walking/picking-up-to-drinking error. It checks seed means and sample standard deviations against the saved results and regenerates `recomputed_metrics.csv` and `verification_report.json`. The version 1.0.0 package verifies 220 model/method/seed/site prediction runs, with maximum absolute difference zero.

The partition and short-enrolment results are retained in `RESULT.json`. The verification script does not reconstruct their partition-specific held-out indices. Model training, fresh radar preprocessing and full evaluation are separate from this saved-prediction verification script.

## Prediction arrays

Each NPZ has `labels` and `persons` in recording order. Main-model arrays have keys `main__<backbone>__<method>__<seed>` and five class scores per recording; take `argmax(axis=1)` to obtain predictions. TTA arrays use `tta__<rule>__<grouping>__<seed>` and store predicted class indices. Class order is walking, sitting down, standing up, picking up and drinking.

For the main configurations, `none` is the source model, `site64` re-estimates site statistics, `naive` is person normalisation and `final` is PRDA with person-referenced kinematic descriptors. Additional arrays retain component controls and partition experiments.

## Data access and versioning

Original radar recordings remain with the [University of Glasgow dataset](https://doi.org/10.5525/gla.researchdata.848) under its own access and licence terms. They and model-weight binaries are not redistributed here. Participant identifiers in the saved results are the dataset's pseudonymous collection-local identifiers.

The `v1.0.0` release fixes the submission version. `data/SHA256SUMS.txt` lists SHA-256 hashes of the derived files. This is a research-data/software release; no journal publication DOI is claimed.
