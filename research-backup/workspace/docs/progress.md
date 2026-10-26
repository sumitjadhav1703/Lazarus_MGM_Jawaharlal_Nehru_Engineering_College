# CardioTwin execution ledger

Source: CardioTwin_Development_Plan.docx; ordered text docs/development-plan.txt. Updated 7 October 2026.

| Task / dependencies | Status | Evidence |
| --- | --- | --- |
| Contracts/setup | Done | Versioned config, package, lockfiles and shared interfaces |
| Data/labels/audit; depends setup | Done | Official metadata/annotation checksums, genuine ECG integration, conservative feasibility report and tests |
| Causal features/state; depends setup | Done | Prior baseline, shared quality/warnings, atomic bounded SQLite replay, future-invariance/restart checks |
| Models/evaluation; depends contracts/data/features | Done | Six grouped/calibrated comparisons, grouped early stopping, validation policy freeze, patient bootstrap, final synthetic replay |
| API/dashboard; depends state/model interfaces | Done | Built responsive clinician UI, available/unavailable quality forecasts, replay/reviews/evidence/evaluation |
| Delivery; depends implemented pipeline/UI | Done for local development | README, licenses, Docker/Compose/CI, reports, architecture/presentation PDFs and actual recording script |
| Verification/review; depends all software | Done | Independent initial review, five repairs, 38 final tests, Ruff/build/browser/clean-install checks |

User authorized useful parallel subagents and full implementation; team/college details and submission are deferred. Dataset references in the source plan were used. No existing repository/code was overwritten (the folder initially contained only the DOCX).

Provisional engineering defaults: 30-second AF episode, 600-second washout, 12 event subjects / 3 test event subjects, experimental signal quality. Operating budget/sensitivity remain unset by default; the synthetic demonstration explicitly uses 1 false warning/eligible hour and sensitivity 0.5. These choices are not clinician-approved.

Real-data forecasting acceptance: NOT PASSED. Strict uncertain-rhythm semantics leave four annotation candidates across four subjects; downloaded-signal audit has no quality-qualified events. Missing signals/complete duplicate-quality verification and alternative label justification remain explicit gates. Synthetic success cannot establish clinical fusion/personalization benefit, zero-error safety or Indian performance.

External delivery: pending actual team identity, public repository/link verification, genuine >=20-minute recording, organizer registration/submission. Optional TCN/hardware/advanced simulation deferred per plan. Docker execution unavailable locally; remote CI not run.

See [final verification and requirements coverage](../reports/verification.md), [review resolutions](review.md), and [submission checklist](submission.md).


## ML improvement iteration — 7 October 2026

Completed internet research, robust beat extraction, 17 RR feature additions, prior-only variance/zscores, audited-group/class balancing, group-macro CV coverage/convergence checks, elastic-net challengers and independent calibration diagnostics. Three regimes/ten models were developed under the original frozen patient splits. All three independent review defects were repaired and rechecked. C1 calibration was rejected on development results before test; the selected C100 core logistic package retains event-level synthetic behavior, but test Brier regressed versus v1. Enriched challengers improve synthetic probability fit without establishing clinical forecasting gains.

Current package: `artifacts/ml-v2-reviewed/patient_class`; current evaluation: `reports/ml-v2-evaluation`; current run commands are in README. Old reports, artifacts and delivery PDFs remain previous-release evidence. The fresh strict real-data audit still blocks clinical training. See [measured findings](ml-findings.md), [review closure](ml-code-review.md), and [verification](../reports/ml-improvements/verification.md).
