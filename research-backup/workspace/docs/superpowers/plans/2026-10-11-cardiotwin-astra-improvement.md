# CardioTwin Precursor Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The user requested Astra for technical work; execute natively with GPT-6 Astra. Do not delegate to another model.

**Goal:** Test whether a small within-person ranking objective improves patient-separated continuous AF forecasting, with a matched pointwise control and honest warning-burden accounting.

**Architecture:** Reuse compact development features, the existing nested patient roles, sklearn preprocessing/calibration and the shared evaluator. Add one scoped experiment module and command, plus an opt-in alert-policy version. Only after explicit evidence gates, reuse the current RR and waveform encoders in a matched fusion comparison.

**Tech Stack:** Existing Python 3.12–3.14 environment, NumPy, SciPy, scikit-learn, joblib, pytest/Ruff; existing Torch/HDF5 only for the conditional neural stage. No new dependency.

**Spec:** [Research design](../../../reports/ml-investigation/next-cycle/astra-research-20261011/spec.md). Read [research and source evidence](../../../reports/ml-investigation/next-cycle/astra-research-20261011/research.md) first. Initial plan name retained from 11 October; revision: **12 October 2026 IST**.

**Status:** Plan only. None of the implementation, fitting, signal replay or job submission below has occurred in this research phase. A user review of this concrete plan precedes execution. The workspace is not a Git repository; use source manifests and fresh output directories instead of invented commit/worktree commands.

## Global Constraints

- Same original 121 development people, 132 records and all 241 eligible events.
- Same 600-second history, onset 120–600 seconds ahead, 30-second cadence, minimum AF duration 30 seconds and 600-second washout.
- Same three original outer folds; disjoint fit/stop/calibration/policy/outer groups. No resplitting after seeing outcomes.
- Original primary and LTAF cohorts are already examined. No access in this cycle and no claim of untouched confirmation.
- Runtime inputs are historical signals/features only. Future samples, annotations, onset identity and pair identity are offline supervision/evaluation only.
- Preserve frozen outputs/source snapshots/ledgers, 22 final-document and 21 frozen-evidence bindings; do not edit root README.
- Project dataset/report/artifact storage stays below **25,000,000,000 bytes**, checking both unique-inode allocated bytes and apparent payload sizes.
- Kaggle whole-session ceiling **41,400 seconds**, including bootstrap; reserve **300 seconds** for receipts. No automatic resubmission or cross-session continuation.
- Initial diagnostic/linear outputs reserve at most **250,000,000 bytes**, subject to stricter current headroom. Bound raw measurement outputs separately; no raw duplication or checkpoint download.
- First execution scope is Tasks 1–4. Tasks 5–6 are conditional and must not be automatically queued. Every fit has a predeclaration/source/data/role/pair identity before it begins.
- Provisional clinical research goal remains sensitivity ≥0.50 at assessed FAR ≤0.25/eligible hour; 5% burden, C=0.1, pair timing/support and neural stopping values are **research assumptions**, not clinical standards.

## Review Focus

- A patient alias or a duplicate event must not cross roles or create repeated negative controls; test in Task 1.
- A missing cadence, expiry during a final partial interval, or persistent high risk after expiry must not hide warning burden or rearm a policy; test in Task 2.
- A feature constant within each person, all-missing fit column, or unseen missingness must not create paired signal or fit preprocessing on later roles; test in Task 3.
- A stale search/checkpoint, interrupted subprocess, or cap failure must never be labeled a complete experiment or silently reuse outputs; test in Tasks 2/4/6.
- A changed future signal/annotation or per-window filter restart must not alter an earlier representation; test in Tasks 5/6.

## Dependencies and file map

`Task 1 → Task 3`; `Tasks 1 + 2 + 3 → Task 4`; `Task 4 decision → Task 5`; `Task 5 passed gates → Task 6`. Task 2 can be implemented independently of fitting. A failed Task 1 metadata/pair-support gate stops real fits; Task 4 still produces a negative feasibility report.

| File | Responsibility |
|---|---|
| New `src/cardiotwin/precursor.py` | Fixed roles/pairs, metadata diagnostics, matched linear objective, control diagnostics and experiment reports. Reuse existing helpers; do not refactor the older ladder. |
| New `scripts/run_precursor_cycle.py` | Explicit `audit`, `linear`, `review`, then conditional `measure`, `neural-probe`, `neural` commands; fresh directories and receipt validation. |
| New `scripts/kaggle_precursor_cycle.py` | Bounded private execution wrapper, source/space/time receipts, conditional measured GPU gate. Reuse `kaggle_cycle.run_child`. |
| New `configs/precursor-cycle.json` | Copy of investigation contract plus the finite-policy research fields; defaults remain untouched. |
| Existing `alerts.py`, `config.py`, `evaluation.py`, `investigation.py`, `train.py` | Versioned state machine, interval accounting, feasibility and search provenance at both selectors. |
| Existing `service.py`, `store.py`; tests | Verify/repair only the state/close-event round trip required by the new policy. No UI, database schema or serving promotion. |
| Existing `scripts/kaggle_cycle.py` | Backward-compatible opt-in allocated-size watchdog; no changes to old submitted snapshots. |
| Conditional new `src/cardiotwin/precursor_neural.py`; existing `neural.py` | Matched pair training/provider/checkpoint loop; expose encoder outputs while preserving current forward behavior. |
| New `tests/test_precursor.py`, `tests/test_precursor_cli.py`, `tests/test_kaggle_precursor_cycle.py`, conditional `tests/test_precursor_neural.py` | Scientific contract, fitting isolation, lifecycle and resource checks. Existing relevant tests remain regression checks. |

All source paths are relative to `/Users/sumitjadhav/CardioTwin Development`. Existing references are the current workspace; old jobs keep their immutable submitted versions. Runtime paths come from manifests, never from a primary/held-out directory search.

---

### Task 1: Make the diagnostic cohort, roles and pairs auditable

**Files:** Create `src/cardiotwin/precursor.py`, `scripts/run_precursor_cycle.py`, `tests/test_precursor.py`, `tests/test_precursor_cli.py`. Read existing `investigation.py:550–605`, `neural.py:793–849`, `data.py:407–450,498–541`, `labels.py:49–103` and `gap-study-20261011/exposure-audit.json`. Do not change extraction or labels as a side effect of this task.

**Interfaces:**

```python
@dataclass(frozen=True)
class EventPair:
    group_id: str
    patient_id: str
    record_id: str
    onset: float
    positive_index: int       # global index into the bound CompactRows
    negative_index: int

def make_roles(rows, events, index, split, *, seed=42) -> list[dict]: ...
def build_event_pairs(rows, events, groups, config) -> tuple[list[EventPair], dict]: ...
def audit_metadata(rows, events, index, metadata_root, config) -> dict: ...
def run_diagnostic(cache, split_path, metadata_root, config, output, *, deadline) -> dict: ...
```

The ellipses in these signatures denote interface declarations, not implementation instructions. The algorithm and tests below define their behavior. `make_roles` returns three dictionaries with `fold`, sorted `fit`, `stop`, `calibration`, `policy`, `outer` group lists. Pair diagnostics include denominators, exclusions by reason, unique positive people, recurrence strata and per-person pair mass. No learned model is produced.

- [ ] Write the boundary/pair tests before the implementation. Use ordinary metadata mappings, not signal fixtures:

```python
def test_pair_choice_is_fixed_and_role_bounded():
    cfg = load_config("configs/investigation.json")
    def row(t, label, group="g"):
        return dict(patient_id="p", record_id="r", group_id=group,
                    timestamp=t, accepted=True, label=label)
    rows = [row(900, 0), row(3210, 1), row(3240, 1), row(3270, 1)]
    events = [dict(patient_id="p", record_id="r", group_id="g", onset=3600)]
    pairs, report = build_event_pairs(rows, events, {"g"}, cfg)
    assert [(p.positive_index, p.negative_index) for p in pairs] == [(2, 0)]
    assert report["paired_events"] == 1
    assert build_event_pairs(rows, events, {"another"}, cfg)[0] == []
```

Add parameterized cases for exact −600/−120 boundaries, closest-time ties, unknown/AF/washout/rejected rows, negative reuse, two events claiming one positive, identity aliases, zero pairs and no cadence opportunity. A duplicate `(record_id,onset)` fails instead of silently weighting it twice. Assert that changing features cannot change pair membership and changing a later role cannot change fit pairs.

- [ ] Run the new tests and confirm failures are missing behavior, then implement sorted indexed selection. Positives are nearest −360 s; negatives nearest −2700 s within −3600 to −1800 s; ties prefer earlier time. Require unique event association, same person/record and nonoverlapping `[t−600,t)` inputs. Assign negative rows at most once in sorted event order. Do not sample additional pairs to meet a count.

```python
positive = min(positive_candidates, key=lambda i: (abs(rows[i]["timestamp"] - (onset-360)),
                                                   rows[i]["timestamp"]))
negative = min(negative_candidates, key=lambda i: (abs(rows[i]["timestamp"] - (onset-2700)),
                                                   rows[i]["timestamp"]))
```

- [ ] Reproduce the exact existing group split: sorted development groups, event-presence strata, `StratifiedGroupKFold(3,shuffle=True,random_state=42)`, `_partitions(outer_train,42+fold)`, then `_partitions(fit,142+fold)` with its second partition as stop and the remaining fitting groups merged. Do not invoke the old window sampler. Verify all role memberships against the gap-study role evidence and original outer receipts. Reject missing/extra allowed groups, aliases in different roles, changed seed, changed extraction config, or any primary group. Persist `roles.json` before analysis and bind it to the cache index hash.

- [ ] Independently audit annotations without calling `label_at`, `eligible_onsets` or `_iridia_intervals` as the reference. Resolve only the 132 allowed record directories from the bound index. Read annotation CSVs/acquisition JSON metadata, using recorded per-part sample counts/fs for day offsets. Confirm first-30-s acquisition masks and `[start,end)` semantics. If counts needed for a multi-day record are absent from metadata, report `independent_reference_incomplete` and stop before fitting; do not open raw HDF5 to hide that limitation. Test a two-day fixture where an AF interval crosses midnight, and timestamps exactly one sample/one cadence around history, onset, washout and unknown boundaries. Reconcile all 241 events and every label disagreement, including the eight events with no accepted positive.

- [ ] Add saved-score diagnostics using the exact prediction paths/hashes in the waveform and RR review manifests. Stream GZIP rows; do not load pickle or raw ECG. Report calibrated pair concordance, person-balanced score differences, fold/person prevalence/score ranges and active-warning misses. Mark unsaved raw logits unavailable. Report cached feature/quality/baseline strata without selecting features from outer associations. Emit pair support **before fitting**: every fit role needs ≥80% observable events and ≥80% positive people, every stop role ≥2 paired people, calibration/policy both classes. Missing support returns `feasibility_failed`, not a smaller cohort.

- [ ] Test that `audit` cannot call `read_signal`, `h5py.File`, a model `.fit`, or any network client by monkeypatching them to raise. Run:

```bash
.venv/bin/python -m pytest tests/test_precursor.py tests/test_precursor_cli.py tests/test_data.py tests/test_research_data.py -q
.venv/bin/python -m ruff check src/cardiotwin/precursor.py scripts/run_precursor_cycle.py tests/test_precursor.py tests/test_precursor_cli.py
```

**Acceptance:** Complete metadata/role/pair-support report with explicit pass/fail and no unexplained disagreement. Software tests alone do not satisfy the real-data gate. Save a source-manifest checkpoint; never overwrite original caches or reports.

### Task 2: Implement an opt-in policy that cannot hide persistent risk

**Files:** Modify `src/cardiotwin/alerts.py`, `config.py`, `evaluation.py`, `investigation.py::select_policy`, `train.py::_select_policy`; create `configs/precursor-cycle.json`. Extend `tests/test_runtime.py`, `tests/test_evaluation_diagnostics.py`, `tests/test_investigation.py`, `tests/test_learning.py`. Inspect `service.py:192–220,246` and `store.py:41–56`; modify only if new-key/close-event tests expose a problem.

**Interfaces:** Preserve `advance_alert(state,timestamp,risk,available,config) -> (dict, warning_or_none)` and `evaluate_stream` signatures. Add:

```python
def alert_interval_seconds(state, start, end, config) -> dict: ...  # alerts.py
def policy_constraints(report, config, *, budget, minimum_sensitivity=0.0) -> dict: ...  # evaluation.py
def policy_search_identity(rows, events, config) -> str: ...       # investigation.py
```

`alert_interval_seconds` is used only by `finite-validity-v1`; returns `warning_seconds`, `unresolved_elevated_seconds`, `rearm_wait_seconds`, `unavailable_rearm_seconds`, `burden_seconds`, `available_seconds`. `policy_constraints` returns `passed` and a list of named failed criteria. Existing no-version configs follow the unchanged legacy path.

- [ ] Add tests for a sustained-high trajectory, expiry, rearming and a partial final interval. This is the critical invariant:

```python
def test_expiry_does_not_hide_high_risk():
    cfg = load_config("configs/precursor-cycle.json")
    state, starts = {}, []
    for t in range(0, 1230, 30):
        state, warning = advance_alert(state, t, 0.99, True, cfg)
        if warning and warning["end_time"] is None:
            starts.append(warning["timestamp"])
    assert starts == [60]
    assert state["active"] is False
    assert state["awaiting_rearm"] is True
    seconds = alert_interval_seconds(state, 1200, 1215, cfg)
    assert seconds["warning_seconds"] == 0
    assert seconds["unresolved_elevated_seconds"] == 15
    assert seconds["burden_seconds"] == 15
```

Add cases: risk exactly release never rearms; risk below release for 30 s does not satisfy 60 s; unavailable/gap interrupts that timer; restart after expiry has the same result as uninterrupted updates; missing-cadence interval closes at `last_timestamp+30`, not late arrival, and its unobserved remainder cannot dilute eligible-hour FAR; record end 15 s after a decision integrates 15 s. A future annotation change cannot alter any state transition. No-warning output has zero burden but retains all events in sensitivity.

- [ ] Validate the finite config as an all-or-none block: recognized version, duration/rearm finite and positive, duration at least cadence, occupancy limits finite in `[0,1]`. Add exactly the spec values under `alert`; preserve default/investigation config files. Persist state fields `policy_version`, `awaiting_rearm`, `low_since`, `last_available`, `last_risk` in addition to existing fields; do not serialize infinities. Reject a state/config version mismatch. Service fingerprint already includes the config; retain its changed-config refusal.

- [ ] Keep legacy transition code in its existing branch. Implement finite closure priority (missing cadence boundary, unavailable update, release, expiry) before considering a new start. A closure sets `awaiting_rearm`; only 60 continuous available seconds strictly below release clear it. No new start at the same update as a close. Missingness never rearms. Record start is initially armed. On-time forecasts extend support; a gap never holds the prior score beyond one cadence.

```python
support_end = min(end, start + config["target"]["cadence_seconds"])
expiry = state["active_warning"]["timestamp"] + config["alert"]["max_duration_seconds"]
active_end = min(support_end, expiry)  # only when active after the update at start
active_seconds = max(0.0, active_end - start)
# If available risk remains >= release after expiry, remaining supported time
# belongs to unresolved elevated burden, not to zero-burden silence.
```

Implement the remaining branches explicitly: below-release rearm wait, unavailable rearm, and already-awaiting high risk. Return a disjoint duration partition; assert nonnegative durations and burden ≤ monitored duration. For finite evaluation, integrate availability and row-label support over at most one cadence. Remaining interval time is `missing_cadence` with unknown outcome eligibility: retain it in monitored time, exclude it from eligible hours, and report it separately. Explicit rejected rows with known labels still count their supported cadence in eligible hours. Preserve old denominator/interval behavior for legacy replay.

- [ ] Extend finite evaluation totals/per-person rows and offline category decomposition. Add eligible-negative monitored/burden seconds, all-time burden occupancy, eligible-negative burden occupancy, unresolved/rearm/unavailable time, warning duration quantiles/max and starts by assessable/known-excluded/unknown. Use exact closure times for `warning_details`. `burden=active+unresolved` with no double count. Keep original event matching; an already-active warning does not become a new timely start when an onset approaches. Extend bootstrap sums for these fields while retaining legacy output values.

- [ ] Use `policy_constraints` in **both** selectors. Finite `investigation.select_policy` keeps its fixed/quantile grid and includes threshold 1 with bounded probabilities; select max sensitivity, then FAR, total starts, burden, higher threshold. `train._select_policy` retains its existing minimum-sensitivity gate and grid but must apply the same finite burden constraints. A finite model cannot pass the older selector while ignoring occupancy. Legacy selectors keep their old ranking. Before any cached search reuse, compare an identity over complete policy rows/events, release ratio, target, state version/duration/rearm, coverage rule and provenance. Store occupancy/FAR requests with each selected result; these may refilter the same compatible state-machine search, but never substitute a changed stream or state contract. Include every field needed to recompute feasibility in candidates.

- [ ] Test search-reuse rejection after changing one risk, an event, cadence or state version; test FAR-passing/high-burden rejection at both selectors. Test all-risk-flat and no-positive-role failure clearly. Use `Store.save_in` to round-trip the new state and an expiry close event, preserving warning ID/evidence/status. Compare service `advance_alert` results with evaluator state replay, including a process restart. Generic JSON storage already supports extra keys; avoid a migration unless a failing test requires one.

```bash
.venv/bin/python -m pytest tests/test_runtime.py tests/test_evaluation_diagnostics.py tests/test_investigation.py tests/test_learning.py tests/test_api.py -q
.venv/bin/python -m ruff check src/cardiotwin/alerts.py src/cardiotwin/config.py src/cardiotwin/evaluation.py src/cardiotwin/investigation.py src/cardiotwin/train.py src/cardiotwin/service.py src/cardiotwin/store.py
```

**Acceptance:** Synthetic persistent-risk burden remains high after expiry; gaps/missingness cannot rearm or extend support; both selectors enforce constraints; service/store and evaluator agree. Existing frozen files and default behavior remain preserved. No retrospective “corrected” performance claim.

### Task 3: Fit two genuinely matched linear objectives

**Files:** Extend `src/cardiotwin/precursor.py` and `tests/test_precursor.py`. Reuse `modeling.matrix/build_pipeline/raw_score`, `train._calibrate/calibration_diagnostics`, `investigation.select_policy` and `evaluation.evaluate_stream`. Do not alter the historical ladder's sampling, parameters or outputs.

**Interfaces:**

```python
def pair_weights(pairs: list[EventPair]) -> np.ndarray: ...
def fit_linear_pair_arms(rows, pairs, schema, *, seed=42) -> tuple[dict, dict]: ...
def diagnostic_controls(rows, fit_pairs, stop_pairs, schema, fitted_arms, *, seed=42) -> dict: ...
def run_linear_comparison(cache, split_path, audit_path, config, output, *, deadline) -> dict: ...
```

Arm dictionary keys are exactly `pointwise_pair_control` and `within_person_rank`; values are ordinary fitted sklearn `Pipeline`s with `impute`, `scale`, `model` steps, compatible with existing scoring/calibration. Diagnostics carry training rows/people/events/weights, preprocessing hash, matrix rank, coefficient norm and convergence.

- [ ] Test person/event weights, shared preprocessing and static-identity cancellation before real fitting:

```python
def test_pair_weights_equalize_people():
    pairs = [EventPair("a", "a", "r1", 3600, 0, 1),
             EventPair("a", "a", "r1", 7200, 2, 3),
             EventPair("b", "b", "r2", 3600, 4, 5)]
    weights = pair_weights(pairs)
    assert weights.mean() == pytest.approx(1.0)
    assert weights[:2].sum() == pytest.approx(weights[2])
```

Use an identity-only synthetic dataset where each person's two vectors are identical but persons differ. Require a zero difference matrix, zero rank-model coefficients within numerical tolerance and tied pair scores. Add a real transient synthetic feature that distinguishes positives under unseen person offsets. This verifies the mechanism, not AF efficacy. All-missing fit columns and newly missing stop columns must retain the same schema/indicator dimensions.

- [ ] Implement the fixed 23-field schema from `CORE_ECG_FIELDS + ENGINEERED_ECG_FIELDS`, sorted and checked against the bound cache; `matrix` adds its existing 23 missingness indicators. Missing required cache fields fail; do not silently select a smaller schema. Pair weights before mean-one normalization are `1/(number_of_people * pairs_for_person)`. Give each of the two rows half its pair mass; preprocessing on unique rows sums any duplicate-row contribution. Fit median imputation only on fit-pair rows and weighted scaling only there. Stop/calibration/policy rows cannot influence any transform.

```python
imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(x_unique_fit)
scale = StandardScaler().fit(imputer.transform(x_unique_fit), sample_weight=unique_row_weights)
z_pos = scale.transform(imputer.transform(x_pos))
z_neg = scale.transform(imputer.transform(x_neg))
d = z_pos - z_neg
training = {
    "pointwise_pair_control": (np.vstack([z_pos, z_neg]), np.r_[np.ones(len(d)), np.zeros(len(d))]),
    "within_person_rank": (np.vstack([d, -d]), np.r_[np.ones(len(d)), np.zeros(len(d))]),
}
```

Use identical `np.r_[pair_weights, pair_weights]` normalized to mean one in both fits, fixed L2 `C=0.1`, `lbfgs`, `max_iter=3000`, `fit_intercept=False`, `class_weight=None`, seed 42. Preserve the installed sklearn parameter convention used by `build_pipeline`; set its logistic model fields instead of introducing incompatible deprecated options. Deep-copy the fitted preprocessing into each pipeline. Treat any convergence warning or nonfinite coefficient as failed fitting; never silently increase iterations/C or discard a fold.

- [ ] Add the spec's 19 fit-only person-block orientation controls (seeds 101–119), using the same fit preprocessing and C; score only stop pairs. Do not send them to calibration, policy or outer scoring. Report the actual control distribution without a precise p-value or tuning. Build all-negative drift pairs at exact cadence with later timestamps on record-hour boundaries and earlier timestamp 2,340 s earlier, both accepted label 0; at most one per record-hour, no row reuse, with support counts. Score these only within fit/stop diagnostic roles. Preserve recurrence, prior-AF-distance, record-phase, quality and baseline-count strata as offline diagnostics, excluded from model inputs.

- [ ] Test isolation by poisoning each later role's features/labels and verifying unchanged fitted coefficients/preprocessing/pair schedule; calibration changes may change only calibration and downstream policy. Use natural calibration with `_calibrate(..., calibration_c=100.0)` and record existing slope/intercept metrics. Nonpositive slope marks directional transfer failed; preserve all results and never choose a sign using outer data. Score ordinary single-history vectors with `raw_score`, then calibrate. Clip numerical risk to the open interval as in the neural runner so threshold 1 is a true no-warning policy; record this numeric rule.

- [ ] Freeze all nine inner-policy frontier choices (FAR 0.1/0.25/1.0 × occupancy 1%/5%/10%) plus the declared legacy-policy bridge before outer scoring. Primary is FAR 0.25 and occupancy 5%; neither secondary preset may replace it. Save each raw logit/calibrated risk with row identity, accepted/label/reasons and record duration. Include all original outer rows/events and negative-only people, even when an event lacked a fit pair. Keep the unchanged legacy scoring of the **new** models separate from original saved-model results.

```bash
.venv/bin/python -m pytest tests/test_precursor.py tests/test_precursor_cli.py tests/test_learning.py tests/test_investigation.py -q
.venv/bin/python -m ruff check src/cardiotwin/precursor.py scripts/run_precursor_cycle.py tests/test_precursor.py tests/test_precursor_cli.py
```

**Acceptance:** Same paired pool/weights/features/C/preprocessor; only objective differs. Each pipeline scores one window without pair metadata. Every role separation and natural calibration property is tested. No claim of generalization before Task 4's complete chronological comparison.

### Task 4: Run the bounded CPU study and make the decision reproducible

**Files:** Complete `scripts/run_precursor_cycle.py`; create `scripts/kaggle_precursor_cycle.py`, `tests/test_kaggle_precursor_cycle.py`; extend `scripts/kaggle_cycle.py`, `tests/test_kaggle_cycle.py` and `tests/test_precursor_cli.py`. Save prospective outputs under new `reports/ml-investigation/next-cycle/precursor-{diagnostic,linear,review}-20261012/` directories. An existing path is an error, never an implicit resume.

**Interfaces:**

```python
# Existing public functions gain optional keywords only; old callers keep behavior.
def dataset_bytes(roots, *, allocated=False): ...
def run_child(command, log_path, timeout_seconds, storage_roots=(), *, check_allocated=False): ...
# New wrapper/report functions:
def run_guarded(command, output, storage_roots, *, session_started, session_seconds=41400,
                reserve_seconds=300, output_reserve_bytes=250_000_000) -> dict: ...
def review_comparison(run_dir, output, *, seed=42, bootstrap_samples=10000) -> dict: ...
```

`run_guarded` uses the existing owned process-group termination/log-drain behavior. `session_started` is a monotonic time captured at the **first bootstrap instruction** and propagated within that process; it is not reset after installations/source validation. If a separate process cannot share that clock value, pass the remaining allowance calculated by the bootstrap and subtract child startup from it. Do not interpret a different machine's monotonic clock.

- [ ] Write lifecycle tests for a successful three-fold/two-arm receipt, an incomplete fold, stale source/pair hashes, existing output, source-package mismatch, killed child, original stderr preservation and storage cap failures. Test inode deduplication separately from apparent/allocated size:

```python
def test_allocated_measure_is_opt_in(tmp_path):
    item = tmp_path / "tiny.bin"
    item.write_bytes(b"x")
    assert dataset_bytes([tmp_path]) == 1
    assert dataset_bytes([tmp_path], allocated=True) == item.stat().st_blocks * 512
```

Mock file `stat` values where allocated size exceeds apparent size and require the new watchdog to refuse. Test preflight headroom smaller than the output reservation before any child starts. Hard links and nested roots count once; transient file disappearance is handled as the existing helper does. Final completion rechecks both measures. Legacy `run_child` callers keep their prior apparent-size behavior.

- [ ] Implement space/time preflight and atomic receipts. With `check_allocated=True`, check both unique-inode sizes every 30 s and at completion; use the larger measure for the hard 25-GB guard. Keep a 250-MB initial output allowance, refuse if it does not fit, and enforce it for the new output directory as well as the project cap. Record package bytes before uploading and verify downloaded bytes/hashes before counting completion. Source validation for this CPU study hashes compact inputs/metadata and the experiment package, not all raw HDF5 files. Pin the environment versions already installed. Save original exceptions/exit codes before cleanup.

```python
remaining = 41400 - (time.monotonic() - session_started) - 300
if remaining <= 0:
    raise TimeoutError("No fitting allowance remains after bootstrap and receipt reserve")
occupied = max(dataset_bytes(storage_roots), dataset_bytes(storage_roots, allocated=True))
if occupied + output_reserve_bytes > 25_000_000_000:
    raise RuntimeError("Output reservation does not fit the project storage cap")
receipt = run_child(command, output / "child.log", remaining,
                    storage_roots=storage_roots, check_allocated=True)
```

Persist `predeclaration.json`, `source-manifest.json`, `roles.json`, `pairs.jsonl`, `receipt.json`, logs, per-fold models/calibrators/metrics, selected policies/searches and compressed complete streams. Predeclaration lists the two arms, fixed numerical assumptions, diagnostics, all secondary presets, failure criteria and prior development reuse. Every artifact references its source/cache/roles/pairs identity. Logs flush phase/arm/fold/completed/total/elapsed at least every 30 s. CPU execution does not reserve a GPU.

- [ ] Expose these exact local command contracts after implementation. Run the diagnostic first; the `linear` command must independently verify the diagnostic's passed status and hashes before `.fit`:

```bash
.venv/bin/python scripts/run_precursor_cycle.py audit --cache data/iridia-development-cache-recovered --split reports/ml-investigation/primary-split.json --metadata-root data/iridia-af --config configs/precursor-cycle.json --output reports/ml-investigation/next-cycle/precursor-diagnostic-20261012
.venv/bin/python scripts/run_precursor_cycle.py linear --cache data/iridia-development-cache-recovered --split reports/ml-investigation/primary-split.json --audit reports/ml-investigation/next-cycle/precursor-diagnostic-20261012/receipt.json --config configs/precursor-cycle.json --output reports/ml-investigation/next-cycle/precursor-linear-20261012 --max-seconds 41100
.venv/bin/python scripts/run_precursor_cycle.py review --run reports/ml-investigation/next-cycle/precursor-linear-20261012 --output reports/ml-investigation/next-cycle/precursor-review-20261012 --bootstrap-samples 10000 --seed 42
```

`--max-seconds` is a ceiling, not an expected runtime. Under Kaggle the supervisor supplies the smaller remaining allowance. A CPU timing pilot on synthetic/mocked folds checks the command lifecycle; do not estimate real runtime from it. For the real cached experiment use either bounded local CPU or one private Kaggle CPU job, never both as duplicate attempts. Record the chosen location before fitting. Kaggle package metadata must set privacy true and GPU false; existing attached compact inputs are reused. Read-only status/source/log checks precede any recovery decision; ambiguous remote state never triggers another push.

- [ ] Review only complete, source-bound streams. Recompute event matches, all starts/exclusions/unknowns, coverage, FAR, active/unresolved/eligible-negative burden, durations, probability metrics and calibration slopes. Check full row/event membership against the index; all 241 events remain. Compare arms with whole-person paired bootstrap, 10,000 seed-42 draws, and log invalid/degenerate draws. Aggregate independent group IDs, not record aliases. Report per-fold values, each person's contribution, negative-only people and leave-one-person-out sensitivity differences. Do not estimate uncertainty as the standard error of three folds.

```python
promising = (
    three_complete_folds and provenance_ok and coverage_unchanged
    and pooled_sensitivity_delta > 0 and folds_with_positive_delta >= 2
    and all(slope > 0 for slope in candidate_calibration_slopes)
    and all(primary_policy_passes_each_outer_fold)
)
supported = promising and delta_ci_low > 0 and direction_survives_person_and_preset_checks
```

Here `direction_survives_person_and_preset_checks` means no single-person deletion reverses the pooled sensitivity difference, and the difference remains positive at each of the predeclared 1%/5%/10% occupancy settings at FAR 0.25. These are conservative advancement rules, not multiplicity-adjusted clinical evidence. Report confidence intervals for rates/burden and the matching event denominators. A no-warning policy is valid, complete and usually unsuccessful. Failure preserves the result; it does not authorize threshold/C/interval tuning or a new seed.

- [ ] Run the scoped suite, then the full software suite once because shared policy behavior changed. Record failures honestly; fix implementation failures without reusing outer results to redesign science:

```bash
.venv/bin/python -m pytest tests/test_precursor.py tests/test_precursor_cli.py tests/test_kaggle_precursor_cycle.py tests/test_kaggle_cycle.py -q
.venv/bin/python -m ruff check src/cardiotwin/precursor.py scripts/run_precursor_cycle.py scripts/kaggle_precursor_cycle.py scripts/kaggle_cycle.py tests/test_precursor.py tests/test_precursor_cli.py tests/test_kaggle_precursor_cycle.py tests/test_kaggle_cycle.py
.venv/bin/python -m pytest -q
```

**Acceptance:** Three paired folds, complete chronological accounting, preservation hashes and a written `advance`, `reject`, `uncertain` or `feasibility_failed` decision with reasons. Tests demonstrate software correctness, not efficacy. By the 18 October deadline this complete CPU result plus honest report is the priority deliverable; a neural run is optional.

### Task 5: Conditional waveform measurement and cost gate

**Prerequisite:** Task 4 identifies a promising development lead, or a documented feature limitation with a specific waveform mechanism. A failed linear comparison alone is insufficient. This task reads raw signals only during a later authorized execution phase; no such reads occurred while writing this plan.

**Files:** Create `src/cardiotwin/precursor_neural.py`, `tests/test_precursor_neural.py`; extend conditional CLI commands in `scripts/run_precursor_cycle.py` and `scripts/kaggle_precursor_cycle.py`. Reuse `data.read_signal`, `features._qrs_band/_peaks`, `neural.rr_sequence/waveform_sequence/SequenceNormalizer` and existing tests. Do not introduce a new filter, gain conversion or detector before showing the diagnosed limitation.

**Interfaces:**

```python
def measure_waveform_contract(rows, records, roles, config, output, *, deadline) -> dict: ...
def hybrid_provider(records, config, *, source_contract, deadline, max_cache_bytes=0): ...
def probe_pair_runtime(rows, pairs, roles, provider, config, *, deadline) -> dict: ...
```

`hybrid_provider` is a context manager with `.get_batch(row_batch)` returning `(rr_batch, waveform_batch)` float32 arrays, shapes `[N,4,2400]` and `[N,3,60000]`, in input order. Zero default persistent cache avoids consuming local headroom; any opt-in RR cache must be within an explicit reservation. No overlapping waveform or learned-embedding cache.

- [ ] Specify a deterministic, bounded measurement panel: at most 12 fit-role histories per fold, drawn from at least six fit people where available, equally including paired positive/negative states and flagged quality/baseline extremes. Selection uses fit metadata only, never outer scores. Save selection IDs/rationale before reading. Produce numeric diagnostics and compact aggregate plots, no copied raw recordings. Record actual fs/lead units/gain metadata, finite fraction, mask edges, amplitude/DC quantiles, clipping, peak counts, lead disagreement and preprocessing differences. Absence of authoritative gain or beat labels remains an explicit unresolved measurement limit, not a guessed scale.

- [ ] Test the representation boundary independently. Change future samples after the decision and future annotation intervals; earlier arrays/features must be unchanged. Test acquisition-mask edges, two-day joins and shared-valid-mask behavior. Compare single-history versus batch provider outputs exactly within numerical tolerance. Show why filtering a physical chunk after its own restart differs from extracting it within a 600-s history; reject an embedding-reuse optimization unless a separate equivalence proof is supplied.

```python
def test_future_cannot_change_completed_representation():
    rng = np.random.default_rng(42)
    ecg = rng.normal(size=(120_200, 2))
    changed = ecg.copy()
    changed[120_000:] *= 1000
    np.testing.assert_allclose(waveform_sequence(ecg[:120_000], 200),
                               waveform_sequence(changed[:120_000], 200))
```

Retain the existing synthetic causal-filter and TCN tests. Add an end-to-end provider/read-window test because the slice-level test alone cannot catch an erroneous extraction boundary. Inject a low-amplitude atrial-like waveform into a synthetic ECG to measure transformation attenuation; this is a signal-processing check, not validation of biological P waves. Without beat adjudication, do not assert PAC/PVC sensitivity.

- [ ] Implement one owner-thread read of each completed 600-s history, derive both representations, then worker-safe transforms as the existing provider does. Ensure handles close on normal completion, timeout and transformation error. Forward-filling, mask semantics, causal filter restart and per-modality normalization remain the current recipe. Record any unresolved measurement failure and stop; an intended repair requires a fresh version/predeclaration before fitting.

- [ ] Run a separate private T4 probe with actual pair batch size 32 (64 histories), both arms and three-fold row counts. Warm up CUDA, synchronize timings, distinguish random versus chronological/cold/warm I/O, record peak allocated/reserved memory, GPU name/versions and source hashes. Include normalization passes, up to 30 fit epochs, stop-pair evaluation each epoch, selected-checkpoint natural stop scoring, all calibration/policy/outer rows, model saving and bootstrap/report costs. Charge distinct reads and forward/backward separately; use the larger observed throughput cost where necessary and the existing 1.5 safety multiplier.

```python
estimate = 1.5 * (
    normalization_histories * measured_normalization_seconds
    + fit_batches * (measured_random_read_seconds + measured_train_batch_seconds)
    + stop_batches * (measured_stop_read_seconds + measured_inference_batch_seconds)
    + score_batches * (measured_chronological_read_seconds + measured_inference_batch_seconds)
    + measured_non_gpu_overhead_seconds
)
passed = finite_positive_timings and estimate <= remaining_session_seconds - 300
```

Counts include **both** positive and negative histories per pair and every fold/arm actually scheduled. Estimate artifact space from actual tensor/optimizer sizes and a bounded sample of compressed row output, keeping best+last checkpoints per arm/fold. Refuse if any arm's complete three-fold workload or output reserve cannot fit; do not assume the old four-epoch 4h19m result predicts this recipe. A provider offering two T4s does not imply two GPUs are scheduled or useful; one verified device suffices.

```bash
.venv/bin/python -m pytest tests/test_precursor_neural.py tests/test_neural.py tests/test_measurement.py -q
.venv/bin/python -m ruff check src/cardiotwin/precursor_neural.py tests/test_precursor_neural.py
```

Future `measure` command requires `--data`, `--cache`, `--roles`, `--pairs`, `--config`, `--output`; `neural-probe` additionally requires `--arm rr_pair16` or `--arm rr_waveform_pair16` and `--max-seconds`. Paths are validated against the signed development index/role manifest; the probe never fits an outcome model or reads held-out people.

**Acceptance:** No unresolved coordinate/mask/causal failure, credible waveform information relevant to the feature limitation, measured feasible full workload and reserved storage. If the panel cannot establish biological morphology fidelity, say so; passing generic throughput tests alone does not authorize fusion training.

### Task 6: Conditional matched RR versus RR+waveform pair learning

**Prerequisite:** Recorded Task 5 measurement/runtime/storage pass and a new predeclaration approved for execution. No automatic launch from the CPU review. Prefer one combined job only if its complete conservative estimate fits; otherwise two explicitly identified arm jobs with identical recipe/roles/source and reported cache/hardware differences. Each arm must complete all three folds.

**Files:** Modify `src/cardiotwin/neural.py` only to expose encoder vectors without changing legacy forward behavior; extend `src/cardiotwin/precursor_neural.py`, conditional CLI/wrapper, `tests/test_precursor_neural.py`, `tests/test_neural.py`, `tests/test_kaggle_precursor_cycle.py`. Keep original model/checkpoint schemas supported and never overwrite submitted job sources.

**Interfaces:**

```python
# Existing RRTCN and WaveformCNN gain encode(x) -> Tensor[N,16].
class PairFusion(nn.Module):
    def __init__(self, *, use_waveform: bool): ...
    def forward(self, rr, waveform): ...       # one raw score per history

def fit_pair_network(rows, fit_pairs, stop_pairs, provider, config, output, *,
                     use_waveform, seed, deadline, max_epochs=30, patience=5) -> dict: ...
def save_pair_checkpoint(path, payload) -> dict: ...
def run_neural_comparison(cache, split_path, audit_path, probe_path, config, output, *,
                          arm, deadline) -> dict: ...
```

Arm IDs are `rr_pair16` and `rr_waveform_pair16`. Checkpoint result includes complete-best/last paths, SHA256s, selected epoch, representation/normalizer states, role/pair/source identities and actual device. The existing `run_neural_ladder` and its four/eight-epoch recipes are not repurposed.

- [ ] Test encoder equivalence first: save a seeded original model's `forward(x)`, expose `encode`, and require `model.output(model.encode(x)).squeeze(-1)` to reproduce it in eval mode. Preserve parameter names/state-dict loading for old models. Reuse the last causal token; do not pool across future tokens or change the CNN/residual blocks. The residual model is not introduced here.

- [ ] Implement the matched architecture and initialization. Build shared RR encoder and fusion head once from a seed, deep-copy them into both arms. Construct the optional waveform branch from a separate deterministic generator so its construction cannot alter shared initialization. Head is `Linear(32,16) → ReLU → Dropout(0.1) → Linear(16,1)`; RR-only uses zero waveform features.

```python
rr_hidden = self.rr.encode(rr)
wave_hidden = self.wave.encode(waveform) if self.use_waveform else torch.zeros_like(rr_hidden)
score = self.head(torch.cat([rr_hidden, wave_hidden], dim=1)).squeeze(-1)
```

Test exact equality of shared initial tensors, absence of waveform gradients in the control, and a real gradient path from each fusion modality. Feed identical histories to both arms, including waveform I/O in the matched accounting even when the RR-only arm ignores it. If a deliberately cheaper RR-only provider is used, declare the cost difference but preserve the scientific rows/weights/schedule.

- [ ] Precompute a seeded epoch schedule with one positive per paired event per epoch, cycling through that event's accepted positive decisions. A chosen positive must still have unique onset association and nonoverlap with its fixed earlier negative. Save all candidate row IDs and schedule hash before fitting; both arms use exactly the same schedule. Fit separate RR/waveform normalizers on the union of scheduled fit histories only; persist arrays/hashes. Stop pairs are the fixed Stage C choices. Do not use stop windows for normalization or add class weights.

```python
positive_score = model(rr_pos, wave_pos)
negative_score = model(rr_neg, wave_neg)
loss = (F.softplus(-(positive_score - negative_score)) * person_event_weights).mean()
optimizer.zero_grad(set_to_none=True)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

Use AdamW lr 0.001, weight_decay 0.0001, batch 32 pairs, 30-epoch ceiling and patience 5 on mean of each stop person's mean pair loss. Weight normalization is fixed over the full training pair pool, not independently per batch. Shuffle pair order deterministically per epoch; keep both arms' schedules equal. Record natural stop-timeline metrics only for the selected checkpoint. Do not change stopping weights mid-run or increase epochs because one arm is losing. A nonfinite loss, OOM, source drift or budget failure records an incomplete run; a batch-size retry needs a declared matched recipe, not an unnoticed change to one arm.

- [ ] Add atomic best and last-completed-epoch checkpoints with model/optimizer/Torch CPU+CUDA/NumPy RNG and schedule position, normalizer states, epoch completion marker, current/best metric, state-policy version, source/config/cache/roles/pairs/schedule hashes, actual environment/device and elapsed time. Write a temp file, fsync, then `os.replace`; hash only after replacement. A progress receipt may record an interrupted epoch but never overwrite the last completed marker. No automatic resume; any future recovery first verifies every identity and records a separate receipt.

```python
with temporary.open("wb") as handle:
    torch.save(payload, handle)
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, destination)
```

Test interruption before rename, corrupt/incompatible checkpoint refusal, matching epoch restoration and preservation of original errors when provider/cleanup fails. Legacy raw checkpoints do not acquire missing recovery fields by assumption.

- [ ] Calibrate natural timelines, freeze inner policies, score all outer rows and review using exactly Tasks 3–4. Neural raw-score adapters must provide ordinary logits without requiring pair labels. Record model parameter count and full measured cost; linear-to-neural results are exploratory because learner/exposure differ. Only RR-only versus fusion is the matched representation comparison. Apply the same advancement gates, person uncertainty and no-promotion rules; higher fit accuracy, selected-pair AUC or current-AF separation cannot win.

```bash
.venv/bin/python -m pytest tests/test_precursor_neural.py tests/test_neural.py tests/test_kaggle_precursor_cycle.py tests/test_runtime.py tests/test_evaluation_diagnostics.py -q
.venv/bin/python -m ruff check src/cardiotwin/neural.py src/cardiotwin/precursor_neural.py scripts/run_precursor_cycle.py scripts/kaggle_precursor_cycle.py tests/test_precursor_neural.py tests/test_kaggle_precursor_cycle.py
```

The future `neural` CLI requires `--cache`, `--split`, `--audit`, `--probe`, `--config`, `--data`, `--arm`, `--output`, `--max-seconds`. Its preflight binds both arm manifests and refuses a model/role/cost mismatch. Three fold checkpoints and full row streams are required even if the provider marks the job COMPLETE.

**Acceptance:** Complete, auditable, matched comparison under the fixed endpoint and burden contract. A failed or uncertain result closes this cycle with a negative finding. Do not automatically try GRU, residual CNN, hazard, SSL, new data or clinical-feature fusion. Those require a separately evidenced question and a new plan.

## Decision and deadline handoff

The first scientific action after implementation is **Task 1's metadata/pair-support audit**, followed by exactly the two-arm linear comparison if feasible. This is the cheapest missing-mechanism test. If labels disagree, repair that scoped defect under a new extraction identity before fitting. If pair coverage fails, publish why and revise the design explicitly. If complete continuous utility does not improve under the burden constraint, reject advancement rather than reporting a selected-window success.

Aim to finish Tasks 1–4 and their review before deciding whether sufficient time remains for Tasks 5–6 and final auditing by 18 October. Do not trade complete folds, provenance or reporting for an extra model. At final handoff, save a decision document and state with all attempted candidates, failures, source hashes, actual CPU/GPU/wall time, storage receipts, limitations and remaining confirmation needs. Preserve the user-facing result even if no model passes. A fresh clinically credible cohort and clinical claims are outside this cycle's demonstrated deliverables.

## Plan self-review record

- Spec stages A/B/C/execution map to Tasks 1/2/3/4; conditional D maps to Tasks 5/6.
- Existing caller review covers `alerts.advance_alert`, `evaluation.evaluate_stream`, both policy selectors, service warning updates/store JSON, default config compatibility and new checkpoint/search identities.
- Interface declarations use explicit names and return contracts; forward references are defined in the task that owns them. CLI option contracts and example paths are concrete.
- Review Focus cases each have a named owning test step. Legacy regressions run before any new scientific comparison.
- No execution or model-performance evidence is asserted by this plan. Remaining empirical gaps are gates, not silently satisfied prerequisites.
