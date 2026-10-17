# CardioTwin

CardioTwin is a research prototype that tries to forecast a **new atrial-fibrillation (AF) onset 2-10 minutes ahead** from the preceding 10 minutes of ECG plus paired age and sex, with one decision every 30 seconds. A FastAPI/SQLite service replays each patient's historical ECG as a causal "digital twin" and a React dashboard shows the ECG, trends, risk, signal quality, warnings and clinician review. The main contribution is a leak-controlled, strictly scored evaluation pipeline and an honest result: **on the public data we tested, forecasting is weak.**

> **Read this first.**
> - **Research prototype.** Not a medical device. Not clinically ready. No treatment advice.
> - **No Indian-population evidence.** All data are public retrospective recordings (IRIDIA-AF, LTAF). Nothing here has been validated in an Indian cohort.
> - **Weak results, stated plainly.** The frozen model catches 6 of 241 development events, 1 of 43 on the primary held-out cohort and 2 of 29 on the external cohort (2-7% sensitivity against our own 50% target). It is not clearly better than random warning timing, and we did not show a benefit from adding clinical inputs. The serving gates stay closed.
> - **The dashboard demo uses manufactured data.** It is a software demonstration, not patient evidence.

## Contents

[Problem and India relevance](#problem-and-india-relevance) | [What it does](#what-it-does) | [Architecture](#architecture) | [Quick start](#quick-start) | [Results](#results) | [Methodology and leakage controls](#methodology-and-leakage-controls) | [Deep-learning work](#deep-learning-work) | [Repository map](#repository-map) | [Limitations](#limitations) | [Future validation plan](#future-validation-plan) | [Team and submission details](#team-and-submission-details) | [Licensing](#licensing) | [Citations and data sources](#citations-and-data-sources) | [Documentation map](docs/index.md)

## Problem and India relevance

**Question.** Can a wearable-style ECG stream plus basic clinical history warn a clinician minutes before a new AF episode? The target is a new AF onset only, not heart attack, stroke or epilepsy.

**Why India.** National AF data for India are lacking. Community estimates range from about 0.1% to 5% depending on method and age. The sources we rely on, and nothing more:

- A door-to-door survey of 4,281 adults in rural Andhra Pradesh found AF in 0.32% (95% CI 0.17-0.54), 1.5% above age 60 and 3.5% above 75. Earlier Indian studies range from 0.1% to 1.6%. [Saggu et al., Indian Heart J 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9039683).
- Single-lead ECG screening of 354 people aged 50+ in six rural Gujarat villages found 5.1% AF (95% CI 2.7-8.7). This is a small convenience sample. [Soni et al., JMIR Public Health Surveill 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC5083844). (Saggu 2022 describes the same study as 1.6%; we have not reconciled the two.)
- India's ischemic stroke burden in GBD 2019 was 535,700 incident cases (95% UI 453,200-631,800), 271,200 deaths and 5,689,300 DALYs. These are ischemic stroke figures only; we do not claim any share is caused by AF. [Goyal et al., J Clin Med 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12609587/).
- Prevention gap: the Kerala-AF Registry (3,421 AF patients, 53 hospitals, 2016-17) had a mean CHA2DS2-VASc of 2.8, and NOAC use was only about 5-6% in its rate/rhythm cohort of 2,901. [Calvert et al., J Arrhythmia 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11317704/).

These sources motivate the question. They do not show that this tool would help. We have no Indian data and no clinical partner (see [Future validation plan](#future-validation-plan)).

## What it does

- **Causal replay twin.** The service replays each patient's historical ECG on explicit 30-second ticks. Each decision uses only the preceding 10 minutes and a running per-patient baseline that is updated *after* the prediction.
- **Two data streams, one model.** Static stream: paired age and sex from the IRIDIA-AF record (no diagnoses or labs were invented). Dynamic stream: causal ECG features plus the patient baseline. The frozen finalist is a logistic regression on both. A separate ECG-only reference is kept for comparison.
- **Alerting.** A pure state machine applies persistence, hysteresis and cooldown, and abstains when signal quality is poor.
- **Doctor-facing dashboard.** ECG trace, trends, baseline, signal quality, risk panel, warning history and a clinician review action. Start/Pause/Step playback. No warning or high-risk period is shown when the model does not produce one.

Screenshots: [overview](dashboard/screenshots/overview-desktop.png), [evaluation](dashboard/screenshots/evaluation-desktop.png). Both show manufactured demo data.

## Architecture

```
ONLINE (runtime, causal only)
  historical ECG --> TwinService (30 s ticks) --> features + running baseline
        --> Predictor (frozen model) --> alert state machine --> SQLite store
        --> FastAPI --> React dashboard
        (baseline is updated after the prediction)

OFFLINE (labels and evaluation, never imported by runtime)
  data / labels --> cohort audit --> grouped patient-level training
        --> freeze (manifest + checksums) --> complete chronological replay
        --> event-level scoring (one-to-one matching, false warnings per eligible hour)
```

`alerts.py` and `evaluation.py` are shared by runtime and offline scoring, so the policy that is scored is the policy that runs. See the [architecture PDF](docs/delivery/architecture.pdf) and the [presentation PDF](docs/delivery/presentation.pdf).

## Quick start

Requirements: Python 3.12-3.14, Node 24 (Node 22.12+ also works with Vite 7) and [uv](https://docs.astral.sh/uv/). On macOS, XGBoost needs an OpenMP runtime, and `OMP_NUM_THREADS=1` avoids a known Torch/XGBoost crash. Add `--extra neural` to `uv sync` only if you want PyTorch experiments or the torch tests (they are skipped without it).

Everything below uses **manufactured ECG/profile fixtures** and is a software check, not patient evidence.

```bash
uv sync --frozen --extra dev --extra research
npm ci --prefix dashboard
npm run build --prefix dashboard

uv run cardiotwin demo --output data/demo
uv run cardiotwin audit --data-dir data/demo --output reports/ml-v2-audit

# Training and result paths must be fresh
uv run python scripts/run_ml_experiments.py --data-dir data/demo --audit reports/ml-v2-audit \
  --artifacts artifacts/ml-v2-reviewed --output reports/ml-improvements/development-ablations.json \
  --synthetic --false-warning-budget 1 --minimum-sensitivity 0.5
# Read recommended_artifact from that JSON; it was patient_class when last run
uv run cardiotwin evaluate --data-dir data/demo --audit reports/ml-v2-audit \
  --model artifacts/ml-v2-reviewed/patient_class --output reports/ml-v2-evaluation

uv run cardiotwin serve --data-dir data/demo --model artifacts/ml-v2-reviewed/patient_class \
  --metrics reports/ml-v2-evaluation/evaluation.json --db data/ml-v2-state.sqlite
```

Open http://127.0.0.1:8000 (API docs at `/docs`). Press Start for accelerated playback (default: 30 historical seconds per wall second), Step for one frame, or add `--playback-seconds 2` for slower playback. Reset clears that patient's replay and review history. Demo patient 000 has an intentional unreliable segment that exercises abstention. For development, `npm run dev --prefix dashboard` serves http://127.0.0.1:5173 and proxies `/api` to port 8000.

Notes: the synthetic demo `audit` exits 0, while an `audit` of an infeasible real-data cohort exits 2; `train` exits 2 on `constraint_not_met`. Use a single replay worker. The demo API binds to localhost and has no hospital authentication or audit governance: do not expose it for a care workflow.

**Verification:**

```bash
OMP_NUM_THREADS=1 uv run pytest
uv run ruff check src tests scripts
npm run build --prefix dashboard
```

In this public repository, on 17 Oct 2026, the full suite gave 836 passed, 83 skipped, 0 failed without PyTorch. With the optional `--extra neural` (PyTorch) the same command gave 948 passed, 37 skipped, 0 failed, measured 17 Oct 2026. Skipped tests need private research receipts that are not shipped here (each test names the missing file) or PyTorch.

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

**Our own criteria.** The 0.25 false warnings per hour budget and the 50% sensitivity target are research engineering criteria we set. The challenge organizers published no accuracy threshold or rubric. Measured sensitivity (2-7%) is far below 50%, so our acceptance gate fails and the serving gates stay closed. The frozen model is therefore not served in the dashboard.

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
- Dashboard fixtures are manufactured. The real-data frozen model is not served because its gate failed.
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

To be filled in by the team. Nothing below has been supplied yet.

| Item | Value |
| --- | --- |
| Team name | `<<TEAM NAME>>` |
| College / incubator | `<<COLLEGE NAME>>` |
| Team members (1-4 full-time students) | `<<MEMBER NAMES>>` |
| Public repository | `<<GITHUB REPOSITORY URL>>` |
| Demo video (unlisted YouTube, 20:00) | `<<YOUTUBE UNLISTED URL>>` |
| Contact | `<<TEAM LEADER CONTACT>>` |

Challenge: CRP Digital Twin Challenge 2026 (Happiest Health) on Unstop. Registration closes 18 Oct 2026 19:00 IST; submission is due 20 Oct 2026 19:00 IST.

## Licensing

Code: [MIT](LICENSE). Third-party data keep their own licenses: see [DATA_LICENSE.md](DATA_LICENSE.md). Raw signals, model binaries and local SQLite state are not shipped. Manufactured fixtures are code-test examples, not de-identified patients.

## Citations and data sources

- **IRIDIA-AF** v1.0.1 (CC BY 4.0): [Zenodo 8405941](https://zenodo.org/records/8405941); Sci Data paper [doi:10.1038/s41597-023-02621-1](https://doi.org/10.1038/s41597-023-02621-1). 167 recordings, 152 patients, 6,690.476 monitoring hours.
- **LTAF** (Long Term AF Database v1.0.0, external cohort): [PhysioNet](https://physionet.org/content/ltafdb/1.0.0/).
- **SHDB-AF** v1.0.1: [PhysioNet](https://physionet.org/content/shdb-af/1.0.1/), an earlier feasibility audit only, not the final cohort.
- **ECGFounder** (attempted, no outcome results): [arXiv 2410.04133](https://arxiv.org/abs/2410.04133).
- India sources: as linked in [Problem and India relevance](#problem-and-india-relevance).
