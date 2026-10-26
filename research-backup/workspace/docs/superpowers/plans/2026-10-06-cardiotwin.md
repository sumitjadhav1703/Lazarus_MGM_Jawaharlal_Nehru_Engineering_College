# CardioTwin Implementation Plan

> For agentic workers: use subagent-driven development with parallel ownership and independent final review. The user explicitly requested planning followed by execution; no repeated authorization handoff is needed.

**Goal:** Implement the AF forecasting research prototype specified in CardioTwin_Development_Plan.docx.
**Architecture:** WFDB loaders and offline annotations feed an audited cohort and patient-grouped learning pipeline. An annotation-free causal feature extractor feeds persistent SQLite patient twins, a frozen calibrated predictor and a warning policy; React displays historical replay and separately labelled evaluation.
**Tech stack:** Python 3.12+, NumPy/SciPy/pandas/wfdb, scikit-learn/XGBoost, FastAPI/SQLite, React/Vite.
**Spec:** ../../development-plan.txt (verbatim ordered extraction of the source DOCX).

## Global constraints
- Ten-minute history; new AF onset in the inclusive 2–10 minute future interval; initial 30-second cadence. Values remain provisional pending audit.
- Current AF, onset inside the first two minutes, uncertain labels, missing history and truncated follow-up are excluded/separated.
- AF_Type, future procedures, undated free text, unverified echo timing, identifiers and filenames never become predictive inputs. `(N` is other/unlabelled, not proven sinus rhythm.
- Subjects and duplicate/overlapping records cannot cross partitions. Preprocessing and sampling fit training patients only; calibration and policy use development patients.
- Runtime never reads annotations. Baselines use earlier reliable data only; weights remain frozen.
- Preserve natural validation/test monitoring and prevalence; bootstrap patients, never overlapping windows.
- Unavailable forecasts are never reassurance; model explanations are not causal; no treatment recommendations or India generalization claims.
- Large data and generated local patient state stay outside Git. Synthetic checks do not become clinical performance results.

## Review focus
Future-sample changes; ambiguous rhythm labels; metadata availability; patient/record separation; restart and quality rejection. Exercise each in runnable tests.

## Tasks and dependencies

1. [x] **Contracts and setup (root)** — pyproject.toml, configs/default.json, config.py, docs/interfaces.md. Freeze shared inputs, versioned forecast/quality/policy config, conservative metadata contract and file ownership. Check install and config validation.
2. [x] **Data, targets and audit (data agent; depends 1)** — data.py, labels.py, audit.py, tests/test_data.py. SHDB-AF acquisition, WFDB chunk reading, metadata normalization and availability audit, conservative rhythm intervals, boundary labels, eligible events/hours, duplicates/overlap, reproducible grouped splits and feasibility gate. Tests cover timing and separation.
3. [x] **Causal features and state (root; depends 1)** — features.py, alerts.py, store.py, service.py, tests/test_runtime.py. Trailing ECG/HRV/quality, prior-only patient baseline, bounded persisted state, persistence/hysteresis/cooldown and replay restart. Prefix and restart checks.
4. [x] **Models and continuous evaluation (model agent; depends 1–3 interfaces)** — modeling.py, evaluation.py, train.py, tests/test_learning.py. Clinical/ECG logistic and XGBoost ablations, grouped tuning/calibration, validation budget selection, frozen artifact/schema/config/splits, final full-record metrics, patient bootstrap, calibration/latency and model card. No invented measured results.
5. [x] **API and dashboard (root + UI agent; depends 3–4 interfaces)** — api.py, dashboard, tests/test_api.py. Patient/state, replay controls, forecasts/warnings/reviews, evaluation and trace/trends. Accessible responsive research dashboard, safe unavailable semantics, bounded paginated history, local deployment.
6. [x] **Delivery (root; depends 2–5)** — README, LICENSE, DATA_LICENSE.md, Dockerfiles/compose, CI, reports, architecture and presentation PDFs, submission checklist and twenty-minute video script. Clean-install reproduction, source attribution, team fields and public links left explicitly pending when unknown.
7. [x] **Verification and independent review (review agents; depends all)** — full tests, actual WFDB integration, synthetic training/replay, frontend build/browser, clean install, requirements matrix and final gap assessment. Fix defects and record actual evidence.

## Shared file ownership
Root owns setup/config/features/alerts/store/service/API/integration/docs, data agent owns data/labels/audit, model agent owns modeling/evaluation/train, UI agent owns dashboard. Agents coordinate interfaces before touching shared files. No Git pushes or registration/submission without real team details and reviewable artifacts.

## Decisions and gates
Engineering episode-duration/washout defaults and operating targets must be labelled provisional. An audit must pass before real-data training, and final-test results must never tune model/policy. Source dataset already uses zero-phase filtering, so prefix-invariant code does not prove raw-device streaming validity. Registration, recording a real 20-minute video and public submission require identity/materials unavailable in the folder; prepare all supporting artifacts and report these dependencies precisely.

## Validation commands
`uv sync --frozen --extra dev`; `uv run pytest`; `uv run cardiotwin demo --output data/demo`; `uv run cardiotwin audit --data-dir data/demo --output reports/demo-audit`; `uv run cardiotwin train --data-dir data/demo --audit reports/demo-audit --output artifacts/final-demo --synthetic --false-warning-budget 1 --minimum-sensitivity 0.5`; `npm ci --prefix dashboard`; `npm run build --prefix dashboard`. Synthetic results must carry their provenance. Real data commands use the audited manifest and explicit operating targets; test data stays untouched until frozen evaluation.


## Final execution status — 7 October 2026
All seven software implementation tasks are complete, including independent review repairs and reproducible synthetic/real-ingestion checks. Checked tasks do not certify a real clinical model: the strict SHDB-AF feasibility gate fails. Real model acceptance, clinician-approved operating targets and Indian validation remain unmet empirical gates. Team identity, public repository, actual video, registration and submission remain pending under the user's development-first instruction. See ../../../reports/verification.md for the coverage and evidence.
