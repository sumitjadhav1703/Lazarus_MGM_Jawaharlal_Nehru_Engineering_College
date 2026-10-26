# CardioTwin

A reproducible AF-onset forecasting research prototype combining permitted clinical history,
causal ECG features and an evolving patient baseline. FastAPI/SQLite maintains each patient twin;
a React dashboard shows historical ECG replay, future risk, quality, warnings, evidence and clinician reviews.

**Implemented software, unproven clinical outcome.** The exact target is a new AF onset in the inclusive
2-10 minutes after prediction, using the previous ten minutes, initially every thirty seconds.
Heart attack, stroke, epilepsy, treatment advice and zero-error guarantees are outside this project.

## Current evidence

Real IRIDIA development now covers 132 recordings and 121 independent groups. The challenge finalist is frozen clinical-plus-ECG logistic regression (paired age/sex plus causal engineered ECG): 6/241 development events detected. The separate ECG reference detects 7/241; CUDA RR16 detects 4/241. These results fail the engineering sensitivity gate and do not establish fusion benefit. Both classical artifacts were frozen before held-out access; all four primary/LTAF evaluations are complete. Fusion detects only 1/43 primary and 2/29 external events; the ECG reference emits no warnings. These are poor forecasting results and the serving gates stay closed. See the [investigation report](reports/ml-investigation/report.md), the [frozen model card](reports/ml-investigation/frozen-challenge-fusion/model_card.md), and [challenge requirements](reports/ml-investigation/CHALLENGE_REQUIREMENTS.md).

## Earlier feasibility and software evidence

- Official SHDB-AF v1.0.1 cohort metadata, 128 headers and 98 rhythm annotations are downloaded and checksum-verified.
- Conservative `(N` = unknown labeling leaves **zero eligible forecasting events** in the current strict audit (four earlier annotation candidates did not survive the complete endpoint rules). The cohort does not pass the current independent-event/grouped-development feasibility gate. Most signals are intentionally not downloaded after this ceiling is established; metadata audit is not a clean-signal feasibility certificate.
- Treating `(N` as adequately labeled non-AF yields **103 annotation candidates across 54 subjects**; this is an explicit alternative assumption, not a validated clinical conclusion. `(N` must never be described as confirmed normal sinus rhythm.
- A complete **manufactured** 24-subject dataset exercises audit, ten grouped/calibrated model variants, frozen policies and continuous test replay. The core ECG logistic model wins the explicit simplicity tie at the synthetic event ceiling; enriched challengers improve synthetic probability fit, while the selected core model's test Brier regressed versus the previous release. Fusion/personalization benefit is not established by these fixtures.
- All synthetic results are **software checks only**. No model is released with a clinical-readiness claim. The SHDB training path stops when its audit gate fails.

See [data research](docs/research.md), [preliminary real cohort audit](reports/shdb-preliminary/audit.json),
[latest synthetic evaluation](reports/ml-v2-evaluation/evaluation.json), [ML research and measured findings](docs/ml-findings.md), and [latest verification](reports/ml-improvements/verification.md).

The [challenge requirements audit](reports/ml-investigation/CHALLENGE_REQUIREMENTS.md) is the current delivery priority. The challenge requires clinical-plus-sensor fusion; the ECG-only scientific reference is retained separately. Team, college and registration details are deferred to the final stage at the user’s request.

## Evidence-driven investigation

The new investigation is **research-only, with frozen primary/external evaluation complete and forecasting inadequate**. Dataset acquisition, software checks
and beat-detector benchmarks do not establish forecasting superiority or clinical readiness. Follow
[the investigation report](reports/ml-investigation/report.md) for actual experiments, failures, costs and unresolved decisions;
[the preserved baseline](reports/ml-investigation/baseline-a/manifest.json) records the original source, configuration and model identities.

The complete checksum-verified IRIDIA-AF v1.0.1 cohort provides 167 recordings from 152 patients and 6,690.476 monitoring hours.
Its annotation-only audit finds 284 eligible events across 143 patients; one cadence-unavailable event remains in the sensitivity
denominator. Signal quality and runtime availability can still cause misses. Reviewed sample indices define continuous
multi-day timelines, with the first 30 seconds of each daily file excluded. Only paired age/sex are used as clinical covariates;
automated vendor RR is not a model input. [Cohort audit](reports/ml-investigation/cohort-audit.json) and
[dataset attribution](DATA_LICENSE.md) document source semantics and integrity.

The frozen primary split reserves 31 groups for testing and 121 for development. Preserve it; do not rerun split generation
in response to model results. Development uses three grouped outer folds with separate fitting, calibration and policy groups.
The objective is sensitivity at ≤0.25 assessed false warnings per eligible hour, with 0.1/1-hour frontiers, unknown/excluded
warning burden, warning occupancy, independent-group Brier and cost reported alongside it. These are research operating points.
LTAF is reserved for external evaluation after freeze; public recording identifiers do not resolve cross-record subject identity.

Base installation does not require neural frameworks. `research` installs HDF5 support; `neural` installs optional PyTorch.
The examples below reproduce acquisition/audit and development stages using fresh output locations; full acquisition needs
about 21.49 GB peak for the IRIDIA archive plus extraction and removes the verified archive afterward.

```bash
uv sync --frozen --extra dev --extra research
# Add --extra neural to the sync command only for RR-TCN/waveform-CNN experiments.
uv run python scripts/prepare_research_data.py iridia --output data/iridia-af --parallel-ranges
uv run python scripts/prepare_research_data.py iridia --output data/iridia-af --continuity-audit
uv run python scripts/prepare_research_data.py iridia --output data/iridia-af --structural-audit \
  --config configs/investigation.json
uv run python scripts/prepare_research_data.py ltaf --output data/ltaf --metadata-only

# Reuse the already frozen primary split; choose fresh outputs for this reproduction.
uv run python scripts/run_investigation.py cache --data data/iridia-af \
  --split reports/ml-investigation/primary-split.json --config configs/investigation.json \
  --partition development --motifs --workers 2 --max-hours 8 --output reports/reproduction-cache
uv run python scripts/run_investigation.py audit --cache reports/reproduction-cache \
  --split reports/ml-investigation/primary-split.json --output reports/reproduction-availability.json
uv run python scripts/run_investigation.py ladder --cache reports/reproduction-cache \
  --split reports/ml-investigation/primary-split.json --trials 6 --max-hours 8 \
  --output reports/reproduction-ladder
# Optional bounded neural screening after installing the neural extra.
uv run python scripts/run_investigation.py neural --data data/iridia-af \
  --cache reports/reproduction-cache --split reports/ml-investigation/primary-split.json \
  --epochs 4 --training-windows 10000 --max-hours 3 --output reports/reproduction-neural
```

`split`, `cache`, `audit`, `ladder`, `neural`, `select`, `freeze` and `evaluate` each expose `--help`. A new independent reproduction may create
its own split before fitting, declaring previously inspected patients development-only; the current split and all results remain
immutable. Output directories must be fresh. Interrupted extraction is incomplete and cannot be used as a finished cache.
Per-command time limits are not a cumulative budget manager: track the approved eight local hours and six GPU hours across stages.
Neural screening defaults are bounded experiments, not the full maximum-30-epoch finalist protocol.

The declared bounded neural screen uses four epochs and at most10,000 fitting windows. An optional `--sequence-cache`
reuses RR sequences only when their source/representation contract matches; it still requires a fresh output directory.
Combine complete classical and neural summaries with `select --summary CLASSICAL --neural NEURAL --split SPLIT --output FRESH_JSON`
before freezing a finalist. Incomplete outer-fold candidates cannot qualify.

`freeze` binds the development-selected reference to its checksum-verified summary. Deployment readiness requires both
the development acceptance gate and the final policy gate; failed gates keep the artifact research-only. If no candidate is
selected, `--research-baseline` explicitly preserves the current recipe. Primary test cache access requires the frozen artifact,
and `evaluate` records held-out access. Do not
use test failures to retune. External signal acquisition (`prepare_research_data.py ltaf --signals`) occurs after freeze.
Authenticated Kaggle quota checks passed on 9 October 2026; use the dated receipts for remaining budget. The portable
`scripts/kaggle_investigation.py` entrypoint requires CUDA and prepared private inputs, and does not upload or launch jobs.
It schedules one CUDA device even when two T4 GPUs are available. Current source emits flushed JSON phase/counter logs
and samples each GPU's utilization and memory every 30 seconds. See [GPU diagnostics](reports/ml-investigation/GPU_DEBUGGING.md).
These changes are absent from the immutable version 1 upload; the completed job therefore lacks them. Returned artifacts confirm five CUDA fits and 5.8923 allocated hours including the smoke test; RR16 completed three folds, waveform only two, and RR32/elastic net were not reached. Utilization percentages were not recorded.
Local neural execution may use available CUDA/MPS or CPU; record the actual device and measured cost.
On macOS, the investigation launcher sets `OMP_NUM_THREADS=1` for the verified Torch/XGBoost OpenMP interoperability issue.

Measurement reproduction is separate from forecasting:

```bash
uv run python scripts/prepare_measurement.py --dataset mitdb --output data/measurement/mitdb
uv run python scripts/prepare_measurement.py --dataset nstdb --output data/measurement/nstdb
uv run python scripts/benchmark_beats.py --data-dir data/measurement/mitdb --dataset mitdb \
  --records $(cat data/measurement/mitdb/RECORDS) --seconds 0 --xqrs --output reports/reproduction-mitdb.json
uv run python scripts/benchmark_beats.py --data-dir data/measurement/nstdb --dataset nstdb \
  --records 118e24 118e18 118e12 118e06 118e00 118e_6 119e24 119e18 119e12 119e06 119e00 119e_6 \
  --seconds 0 --xqrs --output reports/reproduction-nstdb.json
```

The completed grouped comparison retains the six-feature logistic scientific reference and separately declares the genuine
clinical-plus-ECG challenge finalist. Incomplete neural candidates are excluded. Fusion benefit and clinical usefulness are not established.

## Run locally

Requirements: Python 3.12-3.14, Node 24 (Node 22.12+ also supports Vite 7), and uv.
XGBoost on macOS needs an OpenMP runtime if not already available; use the official XGBoost installation instructions.

```bash
uv sync --frozen --extra dev
npm ci --prefix dashboard
npm run build --prefix dashboard

# Manufactured ECG/profile fixtures, explicitly not patient evidence
uv run cardiotwin demo --output data/demo
uv run cardiotwin audit --data-dir data/demo --output reports/ml-v2-audit

# Development ablations select calibration strength on validation; fresh outputs required
uv run python scripts/run_ml_experiments.py --data-dir data/demo --audit reports/ml-v2-audit \
  --artifacts artifacts/ml-v2-reviewed --output reports/ml-improvements/development-ablations.json \
  --synthetic --false-warning-budget 1 --minimum-sensitivity 0.5
# Read recommended_artifact from that report before separate evaluation; currently patient_class
uv run cardiotwin evaluate --data-dir data/demo --audit reports/ml-v2-audit \
  --model artifacts/ml-v2-reviewed/patient_class --output reports/ml-v2-evaluation

uv run cardiotwin serve --data-dir data/demo --model artifacts/ml-v2-reviewed/patient_class \
  --metrics reports/ml-v2-evaluation/evaluation.json --db data/ml-v2-state.sqlite
```

Training requires a fresh empty output directory; the experiment result JSON must also be a fresh path. The current workspace already contains the verified demo;
skip generation/training to serve it, or choose fresh `--artifacts` and result `--output` paths. Read the runner's `recommended_artifact` and use it in evaluate/serve.

Open http://127.0.0.1:8000. The built dashboard is served by FastAPI. API documentation: http://127.0.0.1:8000/docs.
Use Start/Pause for accelerated playback (default: thirty historical seconds per wall second), Step for one frame,
or `--playback-seconds 2` for slower playback. Reset clears that patient's replay and review history.
No warning or high-risk period is substituted into the UI when the model does not generate it.
Synthetic patient 000 contains an intentional unreliable segment to exercise abstention.

Development UI: `npm run dev --prefix dashboard` at http://127.0.0.1:5173; `/api` proxies to port 8000.
For signal-only research replay, omit `--model`; forecasts correctly remain unavailable while baseline/quality still evolve.

## Real SHDB-AF data

```bash
# All cohort headers, annotated rhythms, metadata and license; tiny compared with signals
uv run cardiotwin download --output data/shdb-af --metadata-only
uv run cardiotwin audit --data-dir data/shdb-af --output reports/shdb-preliminary
# Infeasible audit intentionally exits 2 and still writes a complete reasoned report.

# Optional: checksum-verified single signal for ingestion/replay verification
uv run cardiotwin download --output data/shdb-af --records 005
# Full signal download (8.3GB) is supported, but cannot cure uncertain-label limitations.
# uv run cardiotwin download --output data/shdb-af
```

The audit writes cohort.csv/cohort.json, records.json, field_audit.json, audit.json, config.json and splits.json.
It counts missing signals, annotation coverage, fragmentation, duration-qualified transitions, excluded windows,
quality-rejected monitoring, content duplicates and overlapping records. All records belonging to one subject
or duplicate-connected group stay in one partition. Reuse the same audit directory to preserve assigned groups;
conflicting new duplicate links require a new audit/split decision. Never alter a frozen split after examining test results.

Do not label 30 unannotated recordings negative. Exclude `AF_Type`, future diagnosis dates/procedures,
undated free text, and same-day/after-recording echoes absent timing proof. Medication absence is kept missing
where the source does not establish absence. Baseline comorbidity flags are taken as supplied at Holter;
independently dated clinical validation is not available. Source zero-phase filtering/resampling limits streaming claims.

An alternative labeling contract can set `audit.n_rhythm_is_other=true` in a separate versioned config, then
re-audit into a new directory. Document why complete AF/non-AF coverage is justified and acquire enough raw signals
before training. This setting is not permission to call `(N` sinus rhythm or to claim real-data performance.
The genuine record 005 ingestion/quality check is recorded in [SHDB integration](reports/shdb-integration.json).
That v1 window was quality-rejected; the [v2 measurement repair](reports/ml-improvements/shdb-feature-check.json) accepts it under unchanged limits. No model was applied, and the [fresh strict audit](reports/ml-v2-shdb-audit/audit.json) still has zero eligible events.

Clinical review should freeze episode duration, washout, horizon and the numeric false-warning/sensitivity budget.
By default budget and sensitivity are unset; training requires explicit values.

## Model and evaluation protocol

The split unit is a pseudonymous subject. Train-only imputation, explicit missingness indicators,
independent-group/class-cell weights and group-macro tuning build clinical/core-ECG logistic baselines and
clinical/ECG/combined/personal XGBoost comparisons. Seventeen RR dynamics features add enriched logistic, elastic-net fusion/personal and enriched personal XGBoost challengers. Linked patient aliases share one training mass. Held-out calibration patients fit an unweighted sigmoid calibrator at natural prevalence, with archived fit and validation diagnostics.
Grouped early stopping selects XGBoost tree count within training patients. Validation selects the threshold for the configured persistence/hysteresis/cooldown policy against a declared budget. All candidate
weights/policies are frozen before the separate evaluation command opens held-out ECG and labels.

The release directory contains model.joblib, feature_schema.json, config.json, splits.json, manifest.json,
training_report.json, model_card.md and up to ten successfully fitted comparison packages; failed candidates remain recorded. Load **trusted local model artifacts only**;
joblib is a Python object format. Dataset fingerprints bind both training and frozen evaluation to audited inputs. The serialized package
contains its ordered schema and configuration; checksums and sidecar agreement reject mixed releases.
Model versions include a content hash. The causal-rr-v2 extractor rejects legacy packages; re-audit and retrain instead of mixing feature semantics. The served artifact freezes calibration C100; generic C1 is an unvalidated starting hypothesis, rejected for this demo after poorer development calibration.

Evaluation replays complete held-out records chronologically, preserving warmup, quality rejection and cooldown.
Warnings match events one-to-one using the inclusive forecast interval. Unassessable warnings remain explicit.
The false-warning budget uses adequately labeled forecast-eligible hours, including quality-rejected opportunities;
all-recording hours and rates are also reported so unknown labels cannot dilute the operating constraint.
Event sensitivity includes eligible onsets missed during abstention. Output includes lead times, burden,
rejection/exclusion reasons, probability metrics/calibration bins, independent-group bootstrap intervals, per-patient results, prevalence/macro Brier context, raw-versus-calibrated probabilities and separate inference/full-row latency.
Very small or all-correct cohorts can give degenerate bootstrap intervals; these cannot establish zero future error.

Features operate only on available trailing ECG. Robust peak scaling reduces isolated-artifact threshold poisoning; missing-gap masks prevent RR statistics from bridging absent samples. Bounded entropy and baseline zscores remain unavailable when their prerequisites fail. Filters are forward causal; supplied qrs files are unaudited
and not runtime inputs. RR variability is not certified normal-to-normal HRV or a diagnosis. The extrema repetition
fraction is a descriptive saturation proxy, not a calibrated device-rail detector. Quality thresholds and peak
selection remain experimental. Baselines update after prediction with reliable full-history samples; after the
configured cap, running means and variances use a bounded exponential update. Weights never retrain during replay.

## Repository and API

| Path | Responsibility |
| --- | --- |
| `src/cardiotwin/data.py`, `labels.py`, `audit.py` | WFDB/metadata loading, future labels and cohort gate |
| `features.py`, `alerts.py` | Causal ECG/prior baseline and shared quality/warning policy |
| `modeling.py`, `train.py`, `evaluation.py` | Grouped learning, calibration, freeze and continuous evaluation |
| `store.py`, `service.py`, `api.py` | Persistent replay state, warning/review history and local API |
| `dashboard/` | React ECG, trends, future risk, review timeline and evaluation |
| `configs/`, `tests/`, `reports/` | Versioned contract, meaningful checks and actual evidence |
| `docs/delivery/` | Architecture/presentation PDFs |

Endpoints: `/api/health`, `/api/patients`, `/api/patients/{id}/state`, `/forecast`, `/warnings`,
POST `/api/patients/{id}/replay`, POST `/api/patients/{id}/warnings/{warning_id}/review`, and `/api/metrics`.
Every forecast includes timestamp, target interval, data quality, prior baseline, model version, availability reason and uncertainty context;
state includes quality, clinical values, baseline, trace, bounded feature buffer and timeline.
Warning history is paginated. SQLite cursor/baseline/alert/warning commits are atomic, including across processes.
Use one replay worker to avoid accelerated duplicate background ticking. Changed source/model/configuration
requires a fresh database; preserve the same input path spelling on restart, keeping historical state from being silently mixed with a different model.

The demo API binds to localhost. It has no hospital authentication, authorization, audit governance or PHI deployment
contract; do not expose it publicly for a care workflow. Pseudonymization is not sufficient clinical de-identification.

## Verification and deployment

```bash
uv run pytest
uv run ruff check src tests scripts
npm run build --prefix dashboard
# Optional container, after generating artifacts/ml-v2-reviewed/patient_class and data/demo
# docker compose up --build
```

Python and npm lockfiles reproduce dependencies. CI runs tests/lint/build. The container serves the React build
through FastAPI and mounts raw data read-only and persists state in a named volume outside the image; Docker execution depends on a local Docker installation.
Large raw data, model binaries and local SQLite state are ignored by Git. Synthetic source fixtures reproduce
model artifacts; audited manifests and reports are kept as evidence without shipping raw signals.

## Delivery and licensing

[Current architecture PDF](docs/delivery/architecture.pdf), [current presentation PDF](docs/delivery/presentation.pdf),
[twenty-minute recording script](docs/video-script.md), and [submission checklist](docs/submission.md).
These PDFs describe the frozen IRIDIA/LTAF investigation and its weak measured forecasting results. All twelve pages were rendered and visually checked; the earlier PDFs are preserved in `reports/ml-investigation/delivery-pre-final/`.
Team/college details, public repository/video links, actual recording, registration and submission remain pending
until those real details are supplied. RR TCN and waveform CNN are optional investigation challengers; hardware and advanced simulation remain deferred.

Code: [MIT](LICENSE). All third-party data: [separate dataset attribution and reuse terms](DATA_LICENSE.md),
[official resource](https://physionet.org/content/shdb-af/1.0.1/). Synthetic profiles cannot establish Indian
performance; the later validation path requires an ethically approved Indian clinical partner and prospective silent evaluation.
