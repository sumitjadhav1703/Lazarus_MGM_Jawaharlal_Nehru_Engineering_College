# Chance analysis (saved-result, descriptive)

Descriptive re-analysis of results that were already saved from the examined cohorts. It does not refit any model, read raw ECG, change a frozen policy or open a new cohort.

- `dev_results.json`, `dev_extra.json`, `dev_summary.json`: development-fold analysis (observed detections versus the same alert policy applied to risk scores shifted to random times; per-patient concentration; fold AUROC).
- `heldout_results.json`: the same descriptive comparison for the frozen fusion model on the primary and LTAF evaluations (200 random-timing draws), plus pooled and patient-macro AUROC.
- `brier_results.json`: Brier score against the constant-prevalence Brier score, and AUROC, per development fold.
- `common.py`, `dev_analysis.py`, `heldout_analysis.py`, `brier.py`, `postprocess.py`: the scripts that produced those JSON files.

The saved inputs the scripts read (per-row prediction streams, next-cycle receipts, model binaries) are not shipped in this repository, so the scripts cannot be re-run from a clean clone; the JSON files are the shipped output. Paths in the scripts are relative to this directory. Figures are descriptive and carry no claim of clinical or predictive usefulness.
