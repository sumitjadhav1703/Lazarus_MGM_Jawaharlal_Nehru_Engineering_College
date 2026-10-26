# ECGFounder engineering resilience implementation plan

> **For agentic workers:** Use the existing delegated implementation and independent review workflow. Track execution in `council/work/ecgfounder-resume-20261021/state.json`; the user has explicitly requested autonomous diagnosis and implementation.

**Goal:** Repair the reproduced dense resource-monitoring bottleneck and preserve useful independent measurements without hiding failed or unsafe stages.

**Architecture:** Keep the existing serial process-group supervisor and four fixed engineering phases. Separate cheap per-inference RSS/CUDA checks from redundant full filesystem walks, then make stage prerequisites and terminal states explicit. Continue only after a supervisor-proven phase-local timeout with successful cleanup and fresh safety validation; unknown failures stop.

**Tech stack:** Existing Python standard library, NumPy/PyTorch, pytest and Ruff. No new dependency, workflow engine or model.

**Spec:** Latest direct user request; `recovery-21g/browser-terminal-observation.json`, both preterminal architecture audits, and `phase-dependency-science-ruling-2326.json` under `council/work/ecgfounder-resume-20261021/`.

## Global constraints

- Preserve exact failed 21-g packet, source, receipts and usage. No retry or new job until repair verification and a fresh declaration/review.
- Keep session41400/work41100/cleanup300/engineering3600 seconds; CPU1800/table900/dense600 seconds; RSS10GB/CUDA12GB/storage25GB/reserve250MB/diagnostics250MB.
- Keep 22 October18:00IST cutoff, all scientific contracts, checkpoint/selection/representation and original fixed workloads. No outcome fits or held-out reads.
- Keep the independent supervisor's five-second storage checks and process-group RSS/deadline enforcement. Do not cache a failed safety check or treat stale/missing receipts as completion.
- Parallel agents own disjoint production and test files. No Git repository exists here; preserve source snapshots and receipts instead of committing.

## Review focus

- Slow storage callbacks must not turn an inference loop into thousands of filesystem scans; cheap checks remain per callback.
- A global deadline/storage/RSS/integrity or cleanup failure must never become a recoverable local timeout.
- Failed stage receipt identity and original exception must survive outer orchestration.
- Independent later stages may run after a known local timeout; dependent work stays blocked, total clocks never restart, overall status stays partial/failed.
- Boundary tests must exercise periodic work and the composed phase loop, not only tiny helpers with free mocks.

### Task 1: Dense monitoring repair and scale regressions

**Owner/files:** implementation agent A only: `scripts/run_ecgfounder_engineering.py`, `tests/test_ecgfounder_engineering.py`.

**Interface:** `real_dense(declaration, *, guard, progress=None)` remains unchanged. Its per-inference `resource_check` continues to measure/check RSS and CUDA. Full storage guards remain before/after dense work and at slot boundaries; independent parent storage monitoring remains unchanged. The inner read-only chunk callback must not rewalk the whole project. Direct table allocation/write storage checks remain fresh.

- [x] Add failing generated tests reproducing the 94 callbacks per 11-row slot and proving bounded full scans while retaining each cheap RSS/CUDA check and safety failure propagation.
- [x] Add a4097-row actual table writer/reader boundary regression and a pure full-count size-plan check, with no real ECG/checkpoint/GPU or629444-row local allocation.
- [x] Make the narrow monitoring repair; record cumulative scan/check calls and elapsed scope where existing receipts can carry it.
- [x] Run focused engineering tests and Ruff; preserve RED/GREEN logs, source hashes and measured generated call counts.

### Task 2: Explicit phase states and bounded independent continuation

**Owner/files:** implementation agent B only: `scripts/kaggle_ecgfounder_engineering.py`, new `tests/test_ecgfounder_routing.py`.

**Interface:** preserve `supervise(...)` signature and existing successful return receipt. Add a narrowly typed supervisor-origin local-phase-timeout signal carrying the terminal receipt. Other failures retain fatal behavior. Extract only the fixed phase-routing loop needed for deterministic generated tests; no generic scheduler or parallel execution.

- [x] Add failing fault-matrix tests: success; CPU local timeout followed by table/dense; staging failure; global deadline; storage/RSS/integrity; child error; missing terminal receipt; cleanup/drain failure; expired remaining total budget.
- [x] Persist `pending/running/complete/failed/blocked` phase records, original error/cause, dependency identity, elapsed and reused producer identity when applicable. Do not synthesize prior success or add automatic reuse of unverified artifacts.
- [x] Distinguish phase-only exhaustion where global/session/calendar budgets remain positive. Cleanup/drain/post-failure safety failures override recoverability. Revalidate source and current global safety before continuing an independent node.
- [x] Retain current execution order and reservation semantics. Partial measurements leave full-run eligibility false and the overall attempt visibly non-successful.
- [x] Run focused supervisor/routing checks and Ruff; save RED/GREEN evidence. Existing callers must remain compatible.

### Task 3: Integration, root-cause report and eligible next measurement

**Owner:** parent plus independent technical/scientific reviewers.

- [x] Recover exact completed21-g receipts or explicitly mark browser-only evidence limits; independently classify terminal failure and reusable completed components.
- [ ] Review both units together against unchanged caps, original workload and failure matrix. Run affected suites, then one full suite after final production edits.
- [ ] Save a concise incident report covering prior distinct failures, test blind spots, repaired causes, stage graph and actual remaining gates. Update canonical JSON/AGENTS continuation status.
- [ ] Freeze a new prospective attempt only if current quota/storage/calendar and source applicability pass. Use the separately recorded22October scientific ruling: one existing four-phase integration validation is permitted because selective execution is not implemented; repeated CPU/table work is an explicit finite validation tradeoff and retains original receipts/usage. Do not add a selective-reuse framework just to skip that measurement.
- [ ] Obtain actual remote terminal evidence before saying the repair works on Kaggle. Publish only verified result claims to README; synthetic/software status cannot establish forecasting gain.
