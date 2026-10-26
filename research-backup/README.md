# CardioTwin research continuation archive

This branch preserves the development source separately from the competition entry. The repository's default `main` branch, root README, dashboard and submitted deliverables are unchanged.

## What is preserved here

- `workspace/`: the current Python package, scripts, tests, configurations, dashboard source, dependency lock, development plan and project instructions.
- `source-manifest.json`: SHA256 and byte length for all153 source/documentation files in that snapshot.
- `archive_tools.py`: standard-library multipart archive verification and safe restoration, with five focused tests in `test_archive_tools.py`.

The development workspace README is an older, hash-bound research document. Use the repository root README for the competition entry and published results.

## Get the source again

```bash
git clone --single-branch --branch archive/research-2026-10-26 https://github.com/sumitjadhav1703/Lazarus_MGM_Jawaharlal_Nehru_Engineering_College.git CardioTwin-research
cd CardioTwin-research/research-backup/workspace
uv sync --frozen --extra dev --extra research
uv run cardiotwin --help
```

Use Python3.12–3.14 as declared by the project. Optional neural work also needs the existing `neural` extra. Dataset paths and old absolute host paths may need local configuration after restoration; never rewrite original provenance receipts to make their hashes appear current.

## Large private research archive

**Status: upload and full remote verification are pending. Do not delete the original project yet.**

The separate private Kaggle archive is being prepared at `sumit1703/cardiotwin-research-archive-20261026`. It preserves large datasets, feature caches, trained model artifacts, prediction streams, source snapshots, historical failures, access ledgers and video source/export files. The existing training-input dataset remains unchanged. The archive's exact version, manifest digest and verified part inventory will be added here after recovery verification.

Credentials, browser authentication, virtual environments, installed dependencies, build caches and the200MB random transfer fixture are excluded. Source/data licenses and attribution remain with the archive. Some omitted video-cache symlinks need recreation or regeneration; their original targets are recorded in the private selection summary.

## Research boundary

The completed24-C job passed generated-data engineering checks. ECGFounder forecasting accuracy is still unmeasured; no real outcome fit or model promotion follows from that job. A new measurement route is under preparation, with all original scientific gates and consumed-cohort boundaries retained. Restoring files does not authorize an expired experiment, repeat a failed job, reset a budget or make an examined cohort untouched.

Only load verified own pickle/joblib artifacts in the pinned environment. The code's MIT license does not replace third-party dataset/model terms; retain `workspace/DATA_LICENSE.md` and the original source licenses.
