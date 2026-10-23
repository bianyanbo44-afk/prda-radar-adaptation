# Derived radar activity recognition data

This anonymous review copy contains development-site and confirmation-site saved predictions, derived results, numeric source values and a verification script. Original radar recordings are available from https://doi.org/10.5525/gla.researchdata.848 and are not redistributed here. Seeds, sites and class order are recorded in metadata.json.

Run `python verify_predictions.py` with Python 3 and NumPy to verify 220 full-enrolment and TTA prediction runs and regenerate recomputed_metrics.csv. The script checks per-recording and per-person metrics, class recalls, seed means and sample standard deviations. Partition-specific short-enrolment results are supplied in RESULT.json but their held-out indexing is not reconstructed by this script.
