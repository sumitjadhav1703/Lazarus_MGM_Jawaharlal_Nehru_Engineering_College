# Independent implementation review

Reviewed 6 October 2026 against `docs/development-plan.txt` and the implementation plan. Scope: every production Python module, tests, React source, configuration, Docker/Compose/CI and research/delivery documentation. This is a source and focused behavioral review; the root agent owns full-suite, clean-install and browser verification. No production files were changed by this reviewer.

## Findings requiring repair

1. **P1 — Excluded warnings can become true positives.** `src/cardiotwin/evaluation.py:102-126` matches all issued warnings to events before checking warning-time labels. Only unmatched warnings are checked for excluded/unknown labels. A matched warning from washout or insufficient labelled history therefore inflates event sensitivity and precision. Confirmed with actual label construction: other 0–900, AF 900–1000, other 1000–1900, AF 1900–2000, other 2000–3000; high risk at 1260/1290/1320 issues a warning at 1320. `label_at(1320)` returns `(None, 'washout')`, yet evaluation of the eligible onset at 1900 reports one matched event, sensitivity 1, precision 1 and zero unassessable warnings. Filter to warning-time label 0/1 before event matching, retain all excluded warnings as unassessable burden, and preserve unmatched eligible events as misses. Check abstention opportunities against the same eligible timing contract. Add the recurrence regression; regenerate validation selection and synthetic test artifacts after this evaluation change.

2. **P2 — Restart identity omits the ordered feature schema.** `src/cardiotwin/service.py:45-55` fingerprints model binary/config/version but not `feature_schema.json`. Reversing the six ECG feature names changes model input semantics while the same persisted database is accepted. Confirmed using a copied real synthetic-trained artifact: initialize a database, reverse schema features, initialize again; no conflict occurs. Include the ordered schema in the restart fingerprint and leave a small changed-schema rejection check.

3. **P2 — A failed training rerun leaves an old frozen release loadable.** `src/cardiotwin/train.py:164-169` writes a failed training report without invalidating or separating an existing model package in the output directory. A previous `manifest.json` still says `frozen: true`, and `model.joblib`, config and split files remain usable even though the newest report says the constraint was not met. Guard an existing release directory or publish successful runs through a separate staged directory; ensure a failed run cannot be mistaken for its prior successful output.

4. **P2 — The direct forecast endpoint omits required data quality.** `src/cardiotwin/api.py:74-76` returns only the forecast object, which has timestamp, target interval, availability reason and model version but no quality payload. The source specification explicitly requires data quality on every forecast response. Add the corresponding quality snapshot to the forecast contract and check available/unavailable endpoint responses. The broader state endpoint already includes quality.

5. **P3 — Downloader documentation claims a file it does not fetch.** `DATA_LICENSE.md:12` says the downloader saves `README.md`, while `src/cardiotwin/data.py` fetches metadata, record list, checksums and license, then selected WFDB files. Correct the claim or deliberately fetch the README. No need to add a download solely to preserve this incidental sentence.

## Requirements assessment

| Area | Assessment |
| --- | --- |
| Dataset/metadata | Conservative `(N` unknown semantics, absent annotation rejection, source/date audit, patient pseudonyms, duplicate/overlap exclusions and signal-qualified feasibility gates are present. Official source preprocessing and Japanese-cohort limits are disclosed. |
| Labels/features | Ten-minute history, inclusive two-to-ten-minute horizon, cadence, episode/washout rules and boundary tests are implemented. Runtime signal features do not access future annotations; prior-only quality-accepted baseline updates and explicit missingness are present. |
| Grouped development | Patient partitions, training-only transforms/weights, grouped modest hyperparameter selection, separate natural-prevalence calibration, all required feature ablations and validation-only model/policy selection are implemented. No final-test signals are consumed for fitting decisions. |
| Alert/evaluation | Shared persistence/hysteresis/cooldown and full chronological replay exist. Patient bootstrap, calibration, PR-AUC/Brier, latency, warning burden and rejection/miss reports exist. Canonical labelled-eligible denominators include quality-rejected time; total monitored coverage and unknown-outcome warning counts are separate. Finding 1 must be repaired before trusting policy selection. |
| State/API/UI | Atomic SQLite replay snapshots and warnings, bounded buffers, deterministic explicit ticks, review actions, safe unavailable scores, historical/synthetic labels and model evidence are present. Findings 2 and 4 leave contract/restart gaps. |
| Delivery | Locked dependencies, Docker/Compose/CI, download guidance, separate code/data attribution, submission checklist, genuine twenty-minute recording script and generated architecture/presentation material are supplied. Clean installation, final README and browser checks were ongoing in the parent task when reviewed. |

## Evidence limits and remaining external gates

The synthetic cohort is a manufactured software fixture: its precursor is intentionally designed, so its small held-out success cannot establish physiological signal, clinical usefulness or Indian generalization. Selecting logistic ECG over XGBoost is permissible under the measured model ladder; clinical fusion/personalization benefit remains unproven. The real-data audit does not pass feasibility, and no real clinical model-training result is claimed. A genuine WFDB signal download/integration check validates ingestion only.

The source already uses zero-phase filtering; local prefix invariance does not prove raw-device causality. The current runtime does not detect existing AF and uses signal-quality checks without annotations; excluded-rhythm warning outcomes must remain visible in offline evaluation. Operating targets, episode/washout choices and minimum subject counts are provisional engineering assumptions. Clinical review, adequate real event subjects, untouched real evaluation, approved Indian clinical data, organizer registration, real team identity/public repository and an actual recorded video remain distinct gates.

## Focused verification performed by reviewer

- Executed a recurrence/washout fixture using `label_at`, `eligible_onsets` and `evaluate_stream`; reproduced finding 1 exactly.
- Initialized `TwinService` with a copied trained artifact, reordered its feature schema, and reopened the same database; reproduced finding 2.
- Inspected the existing meaningful tests and trained/evaluated synthetic package. Full-suite execution was intentionally left to the parent task to avoid duplicating its expensive data fixture runs.

This report records the reviewed snapshot. Findings should be marked resolved only after the repair and its runnable regression pass; parent integration evidence belongs in the final release report.


## Implementer resolutions — 7 October 2026

| Finding | Repair and runnable evidence |
| --- | --- |
| P1 excluded warning match | Filter unavailable warning labels before matching; preserve unassessable burden/reasons and missed eligible onsets. Actual washout recurrence regression passes. Rebuilt synthetic model/policy comparisons and test evaluation. |
| P2 model/schema restart identity | Fingerprint model, schema and manifest; verify binary SHA256 and embedded ordered schema/config at load. Reversed sidecar schema is rejected; final replay/review survives unchanged restart. |
| P2 stale training release | Reject a nonempty output before processing; prior artifact bytes remain unchanged. Regression passes. |
| P2 direct forecast quality | Each forecast now includes corresponding quality and prior baseline, including unavailable responses. API check and available final browser/API verification pass. |
| P3 downloader documentation | Corrected DATA_LICENSE.md to list the files actually downloaded. |

Additional integration repair: training now checks the complete audited dataset fingerprint before fitting; changed-source regression passes. Final specification check added patient-grouped XGBoost early stopping, with training-only fold preprocessing, archived fold tree counts and a fixed final tree count. This change and all final source changes pass 38 tests, Ruff, production build and clean installed-package checks. The root agent performed these repairs/verification; this does not represent a second independent review. See [final verification](../reports/verification.md).
