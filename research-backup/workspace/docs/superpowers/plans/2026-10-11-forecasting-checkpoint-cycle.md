# Forecasting checkpoint cycle Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task by task. This workspace is not a Git repository; preserve source snapshots instead of commits.

**Goal:** Test the checkpoint-objective hypothesis without changing frozen models or assuming neural superiority.

**Architecture:** Reuse the existing causal models, full chronological evaluation, patient partitions, weights and calibrator. Add an opt-in stopping objective and allow one architecture per invocation so three folds can complete. Keep the existing stopping objective as default.

**Tech Stack:** Existing Python, PyTorch, NumPy, sklearn and pytest; no dependencies added.

**Spec:** reports/ml-investigation/next-cycle/state.json and training-review.md (independent evidence review).

## Global Constraints

- User removed experiment-hour limits after initial approval; use measured provider quota and bounded stages. Keep original accounting separate. Dataset cap remains25,000,000,000bytes.
- Use original 121 development groups only. Primary/LTAF are already examined and cannot provide untouched confirmation.
- Keep causal trailing 600 s, onset 120–600 s, cadence 30 s, minimum AF 30 s, washout 600 s.
- Fit/stop/calibration/policy/outer groups stay separate. Calibration uses natural prevalence. Complete three folds before pooled selection.
- Preserve frozen models, source snapshots, access ledgers, failed attempts and original receipts.

## Review Focus

- Rare positives: constant low scores can beat discriminative scores on natural raw log loss; opt-in objective must select a different checkpoint in a controlled test.
- Unequal group lengths: preserve independent-group weighting.
- Invalid objective: reject before provider reads or fresh output creation.
- Defaults: retain original natural group-macro selection and model order.
- OOM retries/artifacts: forward and record the selected objective through retries, per-fold trials, checkpoints and summary.

### Task 1: Pin checkpoint behavior

Files: tests/test_neural.py; src/cardiotwin/neural.py.

- [x] Add a controlled two-epoch model test: epoch1 predicts0.01, epoch2 predicts0.2 with100negative/1positive stop windows. Natural loss selects epoch1; patient/class loss selects epoch2. Assert restored weight, chosen epoch, objective and both reported losses.
- [x] Run the test and retain the expected TypeError for missing opt-in argument.
- [x] Add early_stop_weighting='patient_only' to _fit_network, fit_network and run_neural_ladder. Validate choices before work. For patient_class compute sklearn log_loss with existing training_weights; retain natural group-macro loss separately. Record objective, best_epoch and class exposure. Forward through OOM retry and persist mode in checkpoint/trial/summary.
- [x] Run focused neural tests and default-parity checks.

### Task 2: Restrict the candidate queue

Files: scripts/run_investigation.py; scripts/kaggle_investigation.py; tests/test_neural.py.

- [x] Extend CLI forwarding test with --models waveform16 and --early-stop-weighting patient_class; watch argparse reject before implementation.
- [x] Add constrained flags to both entrypoints; map rr16/waveform16/rr32 to existing specs and pass stopping mode. Defaults retain the original queue.
- [x] Verify CLI help, focused tests and Ruff. Review only actual changed code.

### Task 3: Evidence and bounded execution

Files: reports/ml-investigation/next-cycle/* (new artifacts only).

- [x] Validate original final-verification bindings; preserve original training recipe and source bytes.
- [x] Read-only reconcile quota, prior terminal job, dataset privacy and fresh submission identity before any new job.
- [x] Inspect development-only saved learning curves, calibration and policy evidence. Record hypotheses separately from demonstrated failures.
- [x] Declare matched cycle: natural-stop vs balanced-stop RR16, same seed/splits/windows/epochs; use existing classical baselines as historical references. If resources cannot complete matched folds, preserve partials and do not select from them.
- [x] Keep counts, UTC phase/device/utilization logs, exclusive intents, bounded runtime/storage and new immutable source manifest. No automatic resubmission.
- [x] Run meaningful checks, save results and update separate next-cycle state; leave pending access/budget dependencies explicit.

Execution note: preparation/submission tasks complete; remote three-fold matched experiment remains in progress. These checked preparation items do not claim a completed GPU result. Residual waveform candidate was subsequently integrated and verified separately from the immutable submitted RR snapshot.
