# CardioTwin

CardioTwin is a research prototype that tries to forecast a **new atrial-fibrillation (AF) onset 2-10 minutes ahead** from the preceding 10 minutes of ECG plus paired age and sex, with one decision every 30 seconds. A FastAPI/SQLite service replays each patient's historical ECG as a causal "digital twin" and a React dashboard shows the ECG, trends, risk, signal quality, warnings and clinician review. The main contribution is a leak-controlled, strictly scored evaluation pipeline and an honest result: **on the public data we tested, forecasting is weak.**

> **Read this first.**
> - **Research prototype.** Not a medical device. Not clinically ready. No treatment advice.
> - **No Indian-population evidence.** All data are public retrospective recordings (IRIDIA-AF, LTAF). Nothing here has been validated in an Indian cohort.
> - **Weak results, stated plainly.** The frozen model catches 6 of 241 development events, 1 of 43 on the primary held-out cohort and 2 of 29 on the external cohort (2-7% sensitivity against our own 50% target). It is not clearly better than random warning timing, and we did not show a benefit from adding clinical inputs. The acceptance gates failed (`deployment_ready` is false); the default server refuses this model, and it can only be replayed in an explicit, bannered research mode.
> - **The default dashboard demo uses manufactured data and a model trained on that synthetic data.** It is a software demonstration, not patient evidence. The real frozen model can be replayed on a real record only in opt-in research mode, which proves it runs in the twin, not that it forecasts well ([details](#replay-the-real-frozen-model-research-mode)).

## Submission checklist

The 12 items in the submission form are mapped below. Source, reports, team information and PDFs are supplied. The Unlisted YouTube demo is supplied below, with English captions and seven chapters. Anonymous repository, PDF and video access was checked on 19 Oct 2026. This checklist does not establish registration or final Unstop submission success.

| Form item | Where | Status |
| --- | --- | --- |
| Team details | [Team Details](#team-details) | Supplied |
| College/Incubator information | [College/Incubator Information](#collegeincubator-information) | Supplied |
| Project Title | [Project Title](#project-title) | Done |
| Problem Statement | [Problem Statement](#problem-statement) | Done |
| Healthcare Use Case | [Healthcare Use Case and Impact](#healthcare-use-case-and-impact) | Done |
| Technical Stack | [Technical Stack](#technical-stack) | Done |
| AI/ML Model or Framework Details | [AI/ML Model or Framework Details](#aiml-model-or-framework-details) | Done |
| 15-20 minute Demo Video (unlisted YouTube link) | [Demo Video](#demo-video); script in [docs/video-script.md](docs/video-script.md) | Supplied: [Watch the Unlisted CardioTwin demo](https://youtu.be/4CJFCPdKEto); 20:00 local export |
| Open-source License Details | [Open-source License Details](#open-source-license-details) ([LICENSE](LICENSE), [DATA_LICENSE.md](DATA_LICENSE.md)) | Done |
| Architecture Diagram (PDF/PPT) | [docs/delivery/architecture.pdf](docs/delivery/architecture.pdf); also [Architecture Diagram](#architecture-diagram) | Supplied |
| Presentation (PDF/PPT) | [docs/delivery/presentation.pdf](docs/delivery/presentation.pdf) | Supplied with team and college |
| All files and links are publicly accessible without additional permissions | Repository and PDF links supplied; [video status](#demo-video) | Verified 19 Oct 2026: public repository and PDFs; Unlisted video plays without signing in |

Public GitHub repository (separate form field): [Lazarus_MGM_Jawaharlal_Nehru_Engineering_College](https://github.com/sumitjadhav1703/Lazarus_MGM_Jawaharlal_Nehru_Engineering_College), verified public 19 Oct 2026. Its name follows the team/college convention. Verify anonymous access to the final repository, PDFs and video before submitting (see [submission checklist](docs/submission.md)).

## Reviewer guide

1. Read the result and research boundaries above, then the [architecture PDF](docs/delivery/architecture.pdf).
2. Follow [Quick start](#quick-start) for the synthetic software demonstration: start replay, inspect ECG and risk, pause/step, inspect the unreliable segment, review a warning and open evaluation. Synthetic warnings show the workflow, not forecasting evidence.
3. Compare the saved real-data [results](#results), [model card](reports/ml-investigation/frozen-challenge-fusion/model_card.md) and [investigation report](reports/ml-investigation/report.md). The real frozen finalist failed its acceptance gates; optional research replay requires the separate model and data.
4. Use the [submission checklist](docs/submission.md) for the 12 form items and remaining upload/submission actions.

This repository supplies application and research source, locked dependencies, tests, curated evaluation reports, model identity/configuration and delivery PDFs. Raw signals, frozen model binaries, local SQLite state and the complete private research receipt collection are omitted. The synthetic demo is reproducible from source; the shipped reports document the real-data result but do not provide a runnable frozen-model package.

## Contents

[Project Title](#project-title) | [Problem Statement](#problem-statement) | [Healthcare Use Case and Impact](#healthcare-use-case-and-impact) | [What it does](#what-it-does) | [Architecture Diagram](#architecture-diagram) | [Technical Stack](#technical-stack) | [AI/ML Model or Framework Details](#aiml-model-or-framework-details) | [Quick start](#quick-start) | [Results](#results) | [Methodology and leakage controls](#methodology-and-leakage-controls) | [Deep-learning work](#deep-learning-work) | [FAQ for technical reviewers](#faq-for-technical-reviewers) | [Repository map](#repository-map) | [Limitations](#limitations) | [Future validation plan](#future-validation-plan) | [Team and submission details](#team-and-submission-details) | [Open-source License Details](#open-source-license-details) | [Citations and data sources](#citations-and-data-sources) | [Documentation map](docs/index.md)

## Project Title

**CardioTwin: a causal digital-twin replay and strict evaluation harness for forecasting new atrial-fibrillation onset 2-10 minutes ahead (research prototype).**

## Problem Statement

**Question.** Can a wearable-style ECG stream plus basic clinical history warn a clinician minutes before a new AF episode? The target is a new AF onset only, not heart attack, stroke or epilepsy.

### India relevance

National AF data for India are lacking. Community estimates range from about 0.1% to 5% depending on method and age. The sources we rely on, and nothing more:

- A door-to-door survey of 4,281 adults in rural Andhra Pradesh found AF in 0.32% (95% CI 0.17-0.54), 1.5% above age 60 and 3.5% above 75. Earlier Indian studies range from 0.1% to 1.6%. [Saggu et al., Indian Heart J 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9039683).
- Single-lead ECG screening of 354 people aged 50+ in six rural Gujarat villages found 5.1% AF (95% CI 2.7-8.7). This is a small convenience sample. [Soni et al., JMIR Public Health Surveill 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC5083844). (Saggu 2022 describes the same study as 1.6%; we have not reconciled the two.)
- India's ischemic stroke burden in GBD 2019 was 535,700 incident cases (95% UI 453,200-631,800), 271,200 deaths and 5,689,300 DALYs. These are ischemic stroke figures only; we do not claim any share is caused by AF. [Goyal et al., J Clin Med 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12609587/).
- Prevention gap: the Kerala-AF Registry (3,421 AF patients, 53 hospitals, 2016-17) had a mean CHA2DS2-VASc of 2.8, and NOAC use was only about 5-6% in its rate/rhythm cohort of 2,901. [Calvert et al., J Arrhythmia 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11317704/).

These sources motivate the question. They do not show that this tool would help. We have no Indian data and no clinical partner (see [Future validation plan](#future-validation-plan)).

## Healthcare Use Case and Impact

- **Setting.** Ambulatory or telemetry ECG review, where a stream of single- or two-lead ECG is already recorded and a clinician looks back at flagged stretches.
- **User.** A clinician reviewing windows the twin flags, with the signal-quality state and the patient's own running baseline beside the risk.
- **Action.** Review only. The system sends nothing, decides nothing and gives no treatment advice; a warning is a prompt to look at the record, and the clinician can mark it reviewed.
- **Impact, stated honestly.** At 2-7% event sensitivity, and not clearly above random timing, benefit to patients is unproven. We do not claim reduced stroke, fewer admissions or earlier care. The India figures above motivate the question; they do not show this tool would help.
- **What we contribute.** A reusable stream-level evaluation harness: whole recordings replayed in time order, false warnings counted per eligible hour, one-to-one warning/event matching, a random-timing null, and explicit accounting of time that cannot be assessed (current AF, post-AF washout, unknown). It exposed that a low false-warning rate can be an artifact (see [Results](#results)).

## What it does

- **Causal replay twin.** The service replays each patient's historical ECG on explicit 30-second ticks. Each decision uses only the preceding 10 minutes of ECG. A running per-patient baseline is kept as persistent twin state, updated *after* each prediction.
- **Two data streams, one model.** Static stream: paired age and sex from the IRIDIA-AF record (no diagnoses or labs were invented). Dynamic stream: 23 causal ECG features. The frozen finalist is a logistic regression on both (27 inputs, see [AI/ML Model or Framework Details](#aiml-model-or-framework-details)). A separate ECG-only reference is kept for comparison.
- **The per-patient baseline is not a model input.** The frozen finalist's 27 inputs contain no `baseline_*` feature ([feature_schema.json](reports/ml-investigation/frozen-challenge-fusion/feature_schema.json)). The baseline is twin state shown to the clinician and used for quality and context; it does not change the risk score.
- **Alerting.** A pure state machine applies persistence, hysteresis and cooldown, and abstains when signal quality is poor.
- **Doctor-facing dashboard.** ECG trace, trends, baseline, signal quality, risk panel, warning history and a clinician review action. Start/Pause/Step playback. No warning or high-risk period is shown when the model does not produce one. A persistent red banner appears whenever a gate-failed research model is being served.

Screenshots: [overview](dashboard/screenshots/overview-desktop.png), [evaluation](dashboard/screenshots/evaluation-desktop.png). Both show manufactured demo data.

**Fusion: what is real.** The code supports 16 static EHR-style fields, each with an availability flag (`CLINICAL_FIELDS` in [features.py](src/cardiotwin/features.py)): age, sex_male, bmi, hypertension, diabetes, heart_failure, prior_stroke, vascular_disease, prior_af_ablation, prior_afl, pacemaker, beta_blocker, antiarrhythmic, anticoagulation, echo_lad, echo_lvef. The manufactured demo cohort fills all of them (`make_demo` in [data.py](src/cardiotwin/data.py)) to exercise the pipeline. The real IRIDIA-AF cohort supplies only age and sex, so the real frozen model uses only those two. Fusion showed no benefit (ECG-only 7/241 versus fused 6/241). Whether age and sex alone meet the organizers' "static EHR" requirement is a judgement call; we do not claim they do.

## Architecture Diagram

```
ONLINE (runtime, causal only)
  historical ECG --> TwinService (30 s ticks) --> features (running baseline kept as twin state)
        --> Predictor (frozen model) --> alert state machine --> SQLite store
        --> FastAPI --> React dashboard
        (baseline updated after the prediction; not a model input)

OFFLINE (labels and evaluation, never imported by runtime)
  data / labels --> cohort audit --> grouped patient-level training
        --> freeze (manifest + checksums) --> complete chronological replay
        --> event-level scoring (one-to-one matching, false warnings per eligible hour)
```

`alerts.py` and `evaluation.py` are shared by runtime and offline scoring, so the policy that is scored is the policy that runs. PDF version: [architecture.pdf](docs/delivery/architecture.pdf). Slides: [presentation.pdf](docs/delivery/presentation.pdf).

## Technical Stack

Constraints are from [pyproject.toml](pyproject.toml) and [dashboard/package.json](dashboard/package.json); locked versions are from `uv.lock` and `dashboard/package-lock.json`.

| Layer | Component | Pinned range (locked version) |
| --- | --- | --- |
| Language | Python | >=3.12,<3.15 |
| Numerics / data | numpy | >=2.0,<3 (2.5.3) |
| | scipy | >=1.14,<2 (1.18.1) |
| | pandas | >=2.2,<3 (2.3.3) |
| | wfdb (PhysioNet WFDB I/O) | >=4.2,<5 (4.3.1) |
| ML | scikit-learn (logistic regression, calibration) | >=1.8,<2 (1.9.1) |
| | xgboost (gradient-boosting comparison) | >=3.0,<4 (3.4.1) |
| | joblib (model artifacts; trusted local files only) | >=1.4,<2 (1.6.0) |
| | torch (optional `neural` extra: RR-TCN, waveform CNN) | >=2.6,<3 (2.14.1) |
| | h5py (optional `research` extra: IRIDIA-AF HDF5) | >=3.14,<4 (3.16.0) |
| Service | FastAPI | >=0.115,<1 (0.142.2) |
| | uvicorn | >=0.34,<1 (0.54.0) |
| | SQLite (Python standard library) | replay state, warnings, reviews |
| Dashboard | React / react-dom | ^19.2.0 (19.3.0) |
| | Vite | ^7.1.9 (7.3.7) |
| | Node | 24 (Node 22.12+ also works with Vite 7) |
| Quality | pytest, httpx, ruff | >=8,<10 / >=0.28,<1 / >=0.12,<1 |
| Packaging | hatchling, uv; [Dockerfile](Dockerfile) (node:24-alpine build stage, python:3.13-slim runtime) and [compose.yaml](compose.yaml) | |
| Delivery PDFs | ReportLab 4.4.9 via `scripts/build_delivery_pdfs.py` | |
| CI | GitHub Actions ([checks.yml](.github/workflows/checks.yml)) | |

## AI/ML Model or Framework Details

**Frozen challenge finalist: logistic regression.** Version `investigation-logistic_combined-a5fbcb844aba`. Median imputation, standard scaling, logistic regression (C=100, seed 42), then a sigmoid calibrator fitted on separate calibration groups. Full details: [model card](reports/ml-investigation/frozen-challenge-fusion/model_card.md), [config.json](reports/ml-investigation/frozen-challenge-fusion/config.json), [manifest.json](reports/ml-investigation/frozen-challenge-fusion/manifest.json).

- **27 inputs** ([feature_schema.json](reports/ml-investigation/frozen-challenge-fusion/feature_schema.json)): age and sex, their two availability flags, and 23 causal ECG features (heart rate over 60 s / 300 s windows and their trend and delta, RR-interval spread and variability measures such as SDNN, RMSSD, pNN50, SD1/SD2, RR coefficient of variation, IQR, MAD, lag-1 correlation and sample entropy). No baseline feature, no future information.
- **Target:** new AF onset 120-600 s ahead, 600 s of history, a decision every 30 s; minimum episode 30 s, washout 600 s.
- **Calibration:** sigmoid logistic fit on score-scaled output (coefficient 0.3385, positive; intercept -4.7451), 18 calibration groups, natural prevalence.
- **Alert policy** (`config.json`): threshold 0.013468, release threshold 0.010101, persistence 60 s, cooldown 600 s, at most 0.25 false warnings per eligible hour (our research assumption). A state machine applies hysteresis and abstains on poor signal quality.
- **Groups:** 85 fitting, 18 calibration and 18 policy patient groups from the 121 development groups, disjoint from the 31 primary test groups.
- **Status:** `deployment_ready=false`; every acceptance gate failed. Results: [Results](#results).

**Other models tried (all development-only; none became the finalist):**

| Model | One line | Result |
| --- | --- | --- |
| ECG-only logistic reference | Same pipeline without age/sex | 7/241 development, 0/43 primary, 0/29 LTAF |
| Gradient boosting (XGBoost), ectopy-like motifs, per-patient baseline features, calibration variants | Classical comparison | 1-10 of 241 |
| RR-TCN16 | Temporal convolutional network on RR-interval sequences | 4/241 and 7/241 |
| Waveform CNN with causal TCN head | `ResidualChunkEncoder` with GroupNorm on raw two-lead ECG chunks | 4/241, 33.13% warning occupancy |
| Precursor ranking | Linear precursor ranking | 5/241 |
| ECGFounder frozen encoder | Foundation-model features (attempt) | No outcome fit: 0 fits; the last private Kaggle job failed during pip setup |

See [Deep-learning work](#deep-learning-work) for the caveats and the chance comparison.

## Quick start

Requirements: Python 3.12-3.14, Node 24 (Node 22.12+ also works with Vite 7) and [uv](https://docs.astral.sh/uv/). On macOS, XGBoost needs an OpenMP runtime, and `OMP_NUM_THREADS=1` avoids a known Torch/XGBoost crash. Add `--extra neural` to `uv sync` only if you want PyTorch experiments or the torch tests (they are skipped without it).

Everything below uses **manufactured ECG/profile fixtures** and a model trained on them: a software check, not patient evidence. Output paths must be fresh (`audit`, `train`, `evaluate` and experiment outputs must not already exist). The `data/` paths below replace the earlier `reports/ml-v2-*` paths, which cluttered version control. A clean run of these exact `data/` paths from this README has not been repeated; an earlier clean-install run of the same commands used other `data/_verify/*` paths.

```bash
uv sync --frozen --extra dev --extra research
npm ci --prefix dashboard
npm run build --prefix dashboard

uv run cardiotwin demo --output data/demo
uv run cardiotwin audit --data-dir data/demo --output data/ml-audit

# Training and result paths must be fresh (data/ is gitignored)
uv run python scripts/run_ml_experiments.py --data-dir data/demo --audit data/ml-audit \
  --artifacts data/ml-artifacts --output data/ml-experiments/development-ablations.json \
  --synthetic --false-warning-budget 1 --minimum-sensitivity 0.5
# Read recommended_artifact from that JSON; it was patient_class when last run
uv run cardiotwin evaluate --data-dir data/demo --audit data/ml-audit \
  --model data/ml-artifacts/patient_class --output data/ml-evaluation

uv run cardiotwin serve --data-dir data/demo --model data/ml-artifacts/patient_class \
  --metrics data/ml-evaluation/evaluation.json --db data/state.sqlite
```

Open http://127.0.0.1:8000 (API docs at `/docs`). Press Start for accelerated playback (default: 30 historical seconds per wall second), Step for one frame, or add `--playback-seconds 2` for slower playback. Reset clears that patient's replay and review history. Demo patient 000 has an intentional unreliable segment that exercises abstention. The default server refuses a model whose manifest has `deployment_ready=false`. For development, `npm run dev --prefix dashboard` serves http://127.0.0.1:5173 and proxies `/api` to port 8000.

Notes: the synthetic demo `audit` exits 0, while an `audit` of an infeasible real-data cohort exits 2; `train` exits 2 on `constraint_not_met`. Use a single replay worker. The demo API binds to localhost and has no hospital authentication or audit governance: do not expose it for a care workflow.

### Replay the real frozen model (research mode)

The default demo above never serves the real finalist. An explicit opt-in lets the gate-failed model run in the twin:

```bash
uv run cardiotwin serve --research-model --model <frozen model dir> --data-dir <data dir> --db data/research-state.sqlite
```

- Without `--research-model` the server still refuses a `deployment_ready=false` artifact. `--research-model` requires `--model`.
- `/api/health` reports `research_model` and `deployment_ready`.
- The dashboard shows a red banner: "RESEARCH MODEL ... FAILED its acceptance gates (weak forecasting). Not for clinical use. Records from the model's development cohort are in-sample."
- To replay one real record, `scripts/prepare_single_record_replay.py --source DATA_DIR --record RECORD_ID --output DIR [--split-file primary-split.json]` builds a one-record data directory (output must be fresh; the ECG is symlinked). With `--split-file` it refuses records in the frozen test groups, so held-out records are not replayed. Real data are not shipped: download IRIDIA-AF from [Zenodo 8405941](https://zenodo.org/records/8405941) (see the real-data pipeline below). The frozen model binary is also not shipped (see [Limitations](#limitations)); its identity and hashes are in the [model card](reports/ml-investigation/frozen-challenge-fusion/model_card.md).
- **Proof it runs (local run, 17 Oct 2026; not a shipped receipt).** The frozen model `investigation-logistic_combined-a5fbcb844aba` on IRIDIA development record `record_000` produced forecasts at 3,600 / 7,200 / 10,800 s with risks 0.00920 / 0.00787 / 0.00804, no warnings in the first 10,800 s, and about 26-29 ms per step. That record belongs to the model's development cohort, so it is **in-sample**. This shows the model runs in the twin. It says nothing about forecasting quality.

**Verification:**

```bash
OMP_NUM_THREADS=1 uv run pytest
uv run ruff check src tests scripts
npm run build --prefix dashboard
```

In this public repository, on 19 Oct 2026, the full suite gave **842 passed, 83 skipped, 0 failed** without PyTorch (241.04 s). Ruff and the dashboard production build also passed. Synthetic ECGFounder tests now use an explicit historical clock; the production experiment deadline remains unchanged. The artifact-integrity fixture uses a persistence interval longer than its positive run to avoid numerical threshold ties. With the optional `--extra neural` (PyTorch) the same command gave 948 passed, 37 skipped, 0 failed, measured earlier on 17 Oct 2026 before five tests were added; the PyTorch run has not been repeated since. Skipped tests need private research receipts that are not shipped here (each test names the missing file) or PyTorch.

**Real-data research pipeline (optional, large).** Full IRIDIA-AF acquisition needs about 21.49 GB peak. Each command has `--help`; output directories must be fresh and the frozen primary split must not be regenerated in response to results.

```bash
uv sync --frozen --extra dev --extra research      # add --extra neural only for PyTorch work
uv run python scripts/prepare_research_data.py iridia --output data/iridia-af --parallel-ranges
uv run python scripts/prepare_research_data.py ltaf --output data/ltaf --metadata-only
uv run python scripts/run_investigation.py --help   # split, cache, audit, ladder, neural, select, freeze, evaluate
```

## Results

All results are retrospective replays of public recordings. The frozen finalist is a clinical (age/sex) plus causal-ECG logistic regression. Operating point: at most 0.25 assessed false warnings per eligible hour (our research assumption, not an organizer threshold).

| Cohort | Events detected | Sensitivity [95% CI] | Assessed false warnings / eligible h | AUROC (window level) |
| --- | --- | --- | --- | --- |
| Development (121 patient groups, 3 grouped folds) | 6 / 241 | 2.49% [0.44, 4.96] | 0.171 | pooled 0.453; per fold 0.40 / 0.49 / 0.57 |
| Primary held-out (31 patient groups) | 1 / 43 | 2.33% [0.00, 7.89] | 0.085 | 0.611 [0.542, 0.681] |
| LTAF external (84 recording groups) | 2 / 29 | 6.90% [0.00, 16.67] | 0.008 | 0.631 [0.513, 0.761] |

The separately kept ECG-only reference detects 7/241 in development, 0/43 on the primary cohort and 0/29 on LTAF, with no held-out warnings.

**Our own criteria.** The 0.25 false warnings per hour budget and the 50% sensitivity target are research engineering criteria we set. The challenge organizers published no accuracy threshold or rubric. Measured sensitivity (2-7%) is far below 50%, so our acceptance gate fails and the default serving gate stays closed. The default dashboard serves a synthetic-trained demo model. The frozen real model is replayed only through the opt-in [research mode](#replay-the-real-frozen-model-research-mode), in a banner-marked session.

### What we found when we checked against chance

An independent re-analysis of the saved predictions re-applied the same alert policy to risk scores shifted to random times within each recording.

- **Not clearly above chance.** Development: 6 detections observed versus about 4.5 expected at random timing (range 1-7; chance of at least 6 about 0.33). Held-out: 3 detections in 72 events (1 + 2) versus about 0.7 expected, uncorrected p about 0.035. That is three events, the intervals include zero, and it was a post-hoc check, so we do not treat it as evidence of skill.
- **The low LTAF false-warning rate is not skill.** Of 398 LTAF warning starts, 392 fell in time that cannot be assessed as true or false (222 during AF, 77 in post-AF washout, 93 unknown). Placing the same warning time at random would give about 0.10 assessed false warnings per hour, versus our 0.008.
- **Probabilities add nothing over base rate in development.** Pooled Brier 0.008384 versus 0.008358 for a constant-prevalence predictor.
- **Catches are sparse and scattered.** The 6 development detections fall in 5 patients. On the primary cohort the top 3 of 31 patients produced 57.5% of all warnings and 14 had none.
- **No fusion benefit shown.** The ECG-only reference scored 7/241 against 6/241 for the fused model. Clinical inputs did not improve forecasting.
- **A calibration defect on development fold 0.** The calibrators learned negative slopes, which reversed the ranking (AUROC 0.573 raw to 0.427 calibrated). The frozen finalists' slopes are positive, so the held-out results above are not affected. A slope guard was added to `train.py::_calibrate` afterwards, for exploratory work only; it does not change any frozen result.
- **Broad search found little.** Gradient boosting, a per-patient baseline, ectopy-like motifs, neural models (below) and a precursor ranking (5/241) all stayed in the 1-10 of 241 range. Only one RR-TCN variant beat random timing, in one fold, and it is not the frozen finalist.

**Context from the literature.** The closest published design (30-minute window ending 5 minutes before onset, patient-level split, same IRIDIA-AF source) reports AUROC 0.657-0.680 in a [CinC 2026 abstract](https://cinc.org/2026/Program/accepted/215.html), without stream-level false-alarm rates. Higher published figures use selected windows or other designs and are not comparable to ours. We found no source showing stream-level minutes-ahead AF-onset warning at about 0.25 false warnings per hour: not shown to be achievable.

**Measured replay.** On real development prefixes, a replay receipt ([frozen-challenge-replay.json](reports/ml-investigation/frozen-challenge-replay.json)) records 72 steps with median 25 ms and p95 27 ms per step, and shows that changing only future ECG or annotations leaves earlier outputs identical. This is a causality check, not a held-out or user-interface validation.

Full detail: [investigation report](reports/ml-investigation/report.md), [frozen model card](reports/ml-investigation/frozen-challenge-fusion/model_card.md), [challenge requirements audit](reports/ml-investigation/CHALLENGE_REQUIREMENTS.md).

## Methodology and leakage controls

- **Causal inputs only.** Runtime features use the available trailing ECG, forward-causal filters and missing-gap masks. No future samples, future annotations, onset IDs or pair IDs. Label code is offline and never imported by runtime code.
- **Splits by patient.** Fit, early-stop, calibration and policy groups are disjoint. All 241 development events stay in continuous evaluation.
- **Freeze before opening held-out data.** Model, calibration and alert policy were frozen with checksums before the primary and LTAF cohorts were opened. Those cohorts are now examined and must not be reused for tuning.
- **Complete-stream scoring.** Whole recordings are replayed in time order, including quality-rejected stretches and time with no label. Warnings match events one-to-one using the inclusive 2-10 minute interval. Missed events, false warnings, unknown-outcome warnings, warning occupancy and abstention are reported, not only sensitivity. False warnings are counted per eligible hour.
- **Replay receipt.** A test perturbs only future ECG/annotations and checks that earlier outputs are identical.
- **Pre-declared gates.** A failed attempt is preserved, not edited or re-run under the same identity. Gates are not loosened after seeing results.
- **Scope of claims.** Development numbers come from choosing among many candidates and are exploratory. Synthetic data are software checks only.

## Deep-learning work

All deep-learning arms are development-only and none became the frozen finalist.

| Arm | Description | Development result |
| --- | --- | --- |
| RR-TCN16 | Temporal convolutional network on RR-interval sequences | Two controls: 4/241 and 7/241. Their sensitivity-difference interval includes zero, so 7/241 is not an established superior model. Only arm clearly above random timing (7/241 vs 1.5 expected, uncorrected p 0.016), fold 0 only, 1 of 8 arms. |
| Waveform CNN with causal TCN temporal head | `ResidualChunkEncoder` (two ECG leads plus a validity mask per 500-sample chunk, residual blocks, GroupNorm) feeding the existing causal temporal head | 4/241, but warning occupancy 33.13% of monitored time (226 assessed false starts), AUROC about 0.458. No improvement or promotion followed. |
| ECGFounder frozen encoder ([arXiv 2410.04133](https://arxiv.org/abs/2410.04133)) | Attempted frozen foundation-model features for a heads-on-top study | **No outcome results.** Actual outcome fits: 0. The last private Kaggle T4 job failed during pip dependency setup before any data were read. Forecasting is unmeasured and the full run was never eligible. |

Related: precursor ranking caught 5/241 versus 6/241 for the pointwise control (advancement rejected). No reviewed model approaches the 50% research sensitivity goal. Code: `neural.py`, `neural_architectures.py` (optional PyTorch), `ecgfounder*.py`.

## FAQ for technical reviewers

**Why 2-10 minutes?** It is our design choice, set in [config.json](reports/ml-investigation/frozen-challenge-fusion/config.json) (`gap_seconds` 120, `horizon_seconds` 600). A warning counts only if the onset follows it by 2 to 10 minutes inclusive. Neither the organizers nor a clinical study fixed this window, and we do not claim it is the clinically right lead time.

**What is an eligible event?** An AF onset preceded by a labelled non-AF stretch long enough to make a valid forecast: at least 12 minutes (600 s history + 120 s gap) of labelled non-AF history, an episode of at least 30 s, and at least 600 s after any previous AF episode (washout). See `eligible_onsets` in [labels.py](src/cardiotwin/labels.py). All 241 development events that meet this stay in continuous evaluation, including the 8 with no available forecast opportunity.

**Eligible hours versus monitored hours.** False warnings are divided by *eligible* hours: labelled time in which a warning could be judged true or false. Warning occupancy and time coverage use *all monitored* hours. Development: 5,243.98 monitored and 3,734.85 eligible hours; primary: 1,446.50 and 1,211.70; LTAF: 1,960.60 and 483.33 ([report.md](reports/ml-investigation/report.md)).

**Why is the LTAF false-warning rate so low?** Not skill. 392 of 398 warning starts fell in time that cannot be assessed (222 during AF, 77 in washout, 93 unknown), so they are not in the denominator or the false count. See [Results](#results).

**Are the splits independent?** Training, early-stopping, calibration and policy roles use disjoint patient-level groups, and the primary cohort is disjoint from development. LTAF groups are recording IDs: they do not prove unique people, and independence from IRIDIA-AF is unverified.

**What was the fold-0 calibration problem?** On development fold 0 the calibrators learned negative slopes and reversed the ranking (AUROC 0.573 raw to 0.427 calibrated; [classical-calibration-diagnostic.json](reports/ml-investigation/classical-calibration-diagnostic.json)). The frozen finalists' slopes are positive, so frozen results are unaffected. A guard that rejects a non-positive slope was added to `train.py::_calibrate`; it is exploratory and changes no frozen result.

**Why not compare with AUROC 0.85-0.99 in the literature?** Those figures use selected windows, recording-level labels or separate-patient controls, and report no stream-level false warnings per hour. Our metric is complete-stream, per eligible hour. The closest design on the same database reports AUROC 0.657-0.680 ([CinC 2026 abstract](https://cinc.org/2026/Program/accepted/215.html)).

**Is it a "twin" if the baseline is not a model input?** The twin is the persistent, causally updated per-patient state (baseline, quality, warning and review history) that is replayed and shown to the clinician. It is not used by the frozen finalist's score. We describe it as a replay twin, not a personalised predictor.

**Does the dashboard risk equal the evaluated risk?** Unverified. The service and the offline evaluation use the same extractor and predictor classes, but a parity test has not been run.

## Repository map

| Path | Responsibility |
| --- | --- |
| `src/cardiotwin/data.py`, `labels.py`, `audit.py` | WFDB/metadata loading, future labels (offline), cohort feasibility gate |
| `features.py`, `alerts.py` | Causal ECG features and prior baseline; shared quality/warning policy |
| `modeling.py`, `train.py`, `evaluation.py` | Grouped learning, calibration, freeze, continuous evaluation |
| `store.py`, `service.py`, `api.py`, `cli.py` | SQLite replay state, warning/review history, FastAPI, command line |
| `investigation.py`, `precursor.py`, `morphology.py` | IRIDIA-AF investigation pipeline and closed precursor/morphology attempts |
| `neural.py`, `neural_architectures.py`, `ecgfounder*.py` | RR-TCN, waveform CNN, ECGFounder attempt |
| `dashboard/` | React ECG, trends, risk, review timeline, evaluation view |
| `scripts/` | Data preparation, investigation, experiments, delivery PDF builder |
| `configs/`, `tests/` | Versioned contract and checks |
| `docs/` | [Documentation map](docs/index.md), [delivery PDFs](docs/delivery/), [video script](docs/video-script.md), [submission checklist](docs/submission.md) |
| `reports/ml-investigation/` | [Report](reports/ml-investigation/report.md), [requirements audit](reports/ml-investigation/CHALLENGE_REQUIREMENTS.md), [model card](reports/ml-investigation/frozen-challenge-fusion/model_card.md) |

API: `/api/health`, `/api/patients`, `/api/patients/{id}/state`, `/forecast`, `/warnings`, POST `/api/patients/{id}/replay`, POST `/api/patients/{id}/warnings/{warning_id}/review`, `/api/metrics`. Every forecast carries its timestamp, target interval, data quality, prior baseline, model version, availability reason and uncertainty context. SQLite cursor/baseline/alert/warning commits are atomic. Load **trusted local model artifacts only** (joblib is a Python object format).

## Limitations

- Research prototype. No clinical-readiness claim; all readiness gates are false.
- Weak forecasting: 2-7% event sensitivity, not clearly above random timing, no fusion benefit shown.
- Public retrospective data only (IRIDIA-AF for development and primary, LTAF external). **No Indian-population evidence.** LTAF groups are recording IDs; patient independence is unverified. LTAF clinical inputs are missing.
- The primary and LTAF cohorts are examined and cannot be reused for tuning. Development results are exploratory.
- The default dashboard uses manufactured fixtures and a synthetic-trained model. The real frozen model runs only in opt-in research mode; records from its development cohort are in-sample, and no held-out record is replayed in the dashboard. The model binary is not shipped.
- Dashboard risk versus evaluated risk parity is unverified: both use the same extractor and predictor classes, but a parity test has not been written or run.
- Whether age and sex alone satisfy a "static EHR" requirement is a judgement call; the real cohort has no other clinical fields.
- Target is AF onset only. Heart attack, stroke and treatment advice are out of scope.
- No hospital authentication, audit governance or PHI contract. Pseudonymization is not clinical de-identification.
- Literature figures are not comparable to ours (different windows and metrics).

## Future validation plan

Planned, not started, and no clinical partner is secured:

1. An ethically approved Indian clinical partner.
2. A prospective **silent** evaluation (alerts logged but not shown to clinicians) at a pre-registered operating point.
3. Fresh cohorts and pre-declared gates for any new model. The current held-out cohorts are used up for this model family.
4. Local data on Indian patients before any claim about Indian performance.

## Team and submission details

Team and college information are supplied below. Platform-only details and video upload status are tracked in the [submission checklist](docs/submission.md).

### Team Details

| Item | Value |
| --- | --- |
| Team name | Lazarus |
| Team members (1-4 full-time students) | Atharva Kulkarni (Team Leader), Pranav Mirkar, Sumit Jadhav |
| Contact | Team leader contact details belong in the Unstop registration/profile; complete entry is unverified. No phone numbers or emails are published in this repository. |

### College/Incubator Information

| Item | Value |
| --- | --- |
| College / incubator | MGM's Jawaharlal Nehru Engineering College |

The repository is named `Lazarus_MGM_Jawaharlal_Nehru_Engineering_College` using the team/college convention.

### Demo Video

**[Watch the Unlisted CardioTwin demo](https://youtu.be/4CJFCPdKEto).** Uploaded 19 October 2026 with English captions and seven description chapters; anonymous playback was verified. The improved local export is exactly **20:00**, with larger synchronized visuals, a fuller manufactured-data dashboard walkthrough, normalized free local Kokoro male narration with more conversational wording, seven chapters and selectable English captions. A midpoint frame from each of the 37 visual units was inspected, and the complete file decoded without errors. Four diagrams explain the forecast window, runtime pipeline, patient roles and measured results. Full subjective playback remains pending. The original 19:59 draft is retained. The recorded listing/form wording differs on duration, so the submission target is exactly 20:00. Script: [docs/video-script.md](docs/video-script.md). The video is Unlisted: anyone with its link can watch without an access request. [Media details](docs/video/README.md), [chapters](docs/video/chapters.md) and [captions](docs/video/captions.srt) are supplied.

### Presentation

[docs/delivery/presentation.pdf](docs/delivery/presentation.pdf) (12 slides). Architecture: [docs/delivery/architecture.pdf](docs/delivery/architecture.pdf). The presentation includes team, college and member details; the architecture PDF describes the system. The presentation includes a clickable link to the uploaded Unlisted demo on slide 12.

### Repository

[Lazarus_MGM_Jawaharlal_Nehru_Engineering_College](https://github.com/sumitjadhav1703/Lazarus_MGM_Jawaharlal_Nehru_Engineering_College) — public repository verified on 19 Oct 2026. Verify that the final reviewed revision and delivery files are published before submission.

Challenge: CRP Digital Twin Challenge 2026 (Happiest Health) on Unstop. Registration closes 18 Oct 2026 19:00 IST; submission is due 20 Oct 2026 19:00 IST.

## Open-source License Details

Code: [MIT](LICENSE). Third-party data keep their own licenses: see [DATA_LICENSE.md](DATA_LICENSE.md). IRIDIA-AF is CC BY 4.0. Raw signals, model binaries and local SQLite state are not shipped. Manufactured fixtures are code-test examples, not de-identified patients.

## Citations and data sources

- **IRIDIA-AF** v1.0.1 (CC BY 4.0): [Zenodo 8405941](https://zenodo.org/records/8405941); Sci Data paper [doi:10.1038/s41597-023-02621-1](https://doi.org/10.1038/s41597-023-02621-1). 167 recordings, 152 patients, 6,690.476 monitoring hours.
- **LTAF** (Long Term AF Database v1.0.0, external cohort): [PhysioNet](https://physionet.org/content/ltafdb/1.0.0/).
- **SHDB-AF** v1.0.1: [PhysioNet](https://physionet.org/content/shdb-af/1.0.1/), an earlier feasibility audit only, not the final cohort.
- **ECGFounder** (attempted, no outcome results): [arXiv 2410.04133](https://arxiv.org/abs/2410.04133).
- India sources: as linked in [Problem and India relevance](#problem-statement).
