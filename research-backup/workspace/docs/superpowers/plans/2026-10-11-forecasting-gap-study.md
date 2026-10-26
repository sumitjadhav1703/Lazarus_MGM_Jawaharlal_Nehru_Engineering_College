# Forecasting gap study implementation plan

**Goal:** Find measured bottlenecks and prepare the next development cycle while the preserved Kaggle experiment runs.

**Architecture:** Read the development cache and immutable training snapshot. Produce independent audit receipts and a primary-source decision report; leave the running recipe and frozen evidence untouched.

**Tech Stack:** Existing Python, NumPy, sklearn and project readers. No new dependencies or data downloads.

**Spec:** User request in this chat; next-cycle/state.json and log-recovery/state.json establish scope and recovery history. Native execution in this chat is authorized.

## Constraints and review focus

- Original121developmentgroups only; primary/LTAF have already been examined.
- Same600second history and120–600second forecasting interval; event attribution uses offline labels only.
- No changes to submitted source, new Kaggle submission, cancellation, frozen models or ledgers.
- Retain25GB filesystem cap; only small audit outputs, no duplicate signal caches.
- Preserve provenance, group separation, empty-event cases, repeated-record identities, inclusive onset boundaries and original errors.
- Canonical work-state tools are unavailable; gap-study-20261011/state.json tracks this work and next-cycle/state.json links to it.

## Tasks

- [x] Reconcile exact recovery identity and capture bounded live evidence; distinguish CUDA placement, utilization and complete results.
- [x] Implement a read-only development exposure audit with self-checks: bind immutable sampling code, reconstruct three folds, attribute positives to eligible onsets, count events/groups before and after sampling, quantify weighted event concentration and role prevalence. Verify historical fitting counts6442/6441/6455.
- [x] Audit representation/throughput using small completed development histories: check ECG/lead/rate/mask provenance, RR retention and full-history model cost. Bound runtime and save reproducible selection receipts.
- [x] Review primary literature on imminent forecasting, event-level learning/evaluation, class weighting/calibration and transfer-learning feasibility. Record exact differences from this endpoint and practical actions.
- [x] Rank evidence-backed next steps; keep proposed experiments separate from implemented changes and achieved forecasting results. Verify scoped helpers, source preservation and storage; update handoff and state.

No production-training changes are part of this study. If an audit supports a new sampler, loss, backbone or policy, declare and test that one axis separately before its next run.
