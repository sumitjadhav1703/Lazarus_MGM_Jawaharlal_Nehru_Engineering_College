# CardioTwin research continuation archive

This branch preserves the development source separately from the competition entry. Archive changes are confined to `research-backup/`; the application, dashboard and submitted deliverables are untouched. The separately requested, 12-line forecasting summary was published to the main README at `662474066901aa932d51ea422638595d0d1f8578`.

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

**Status: private version 1 is uploaded; all 27 remote files were downloaded and their SHA256/lengths verified.** [Pinned archive reference](archive-reference.json). A complete materialized-tree restore has not been tested; retain the original until you have tested restoration.

The separate [private Kaggle archive, version 1](https://www.kaggle.com/datasets/sumit1703/cardiotwin-research-archive-20261026/versions/1) preserves 11,836 files: large datasets, feature caches, trained model artifacts, prediction streams, source snapshots, historical failures, access ledgers and video source/export files. Compressed size: 18,976,407,397 bytes. Original unique file bytes: 26,734,577,785. The existing development training-input dataset remains unchanged. This all-cohort preservation dataset must never be attached as a forecasting-study input.

Download the pinned files with your Kaggle account. Keep the authentic manifest SHA256 from `archive-reference.json`; follow the downloaded `RESTORE.md`, verify each `.bin` part against the reference, rename it using `remote-part-mapping.json`, then run this branch's `archive_tools.py verify --archive-dir downloads --restore /absolute/empty/restore-root`. Reserve at least 48 GB for parts plus the restored hardlinked tree. A whole-dataset ZIP needs additional space.

[Source continuation saved on 27 October](continuation-2026-10-27/README.md) includes the exact historical and later source-only preparations with tests, hashes and scientific limitations. It is not a released experiment or forecasting result.

Credentials, browser authentication, virtual environments, installed dependencies, build caches and the200MB random transfer fixture are excluded. Source/data licenses and attribution remain with the archive. Some omitted video-cache symlinks need recreation or regeneration; their original targets are recorded in the private selection summary.

## Research boundary

The completed24-C job passed generated-data engineering checks. ECGFounder forecasting accuracy is still unmeasured; no real outcome fit or model promotion follows from that job. A new measurement route is under preparation, with all original scientific gates and consumed-cohort boundaries retained. Restoring files does not authorize an expired experiment, repeat a failed job, reset a budget or make an examined cohort untouched.

Only load verified own pickle/joblib artifacts in the pinned environment. The code's MIT license does not replace third-party dataset/model terms; retain `workspace/DATA_LICENSE.md` and the original source licenses.
