# CardioTwin evidence-driven ML investigation

> Delivery update, 19 October 2026: [Watch the Unlisted CardioTwin demo](https://youtu.be/4CJFCPdKEto) is uploaded; current publication and remaining platform actions are tracked in [the submission checklist](../../docs/submission.md). The dated research findings below are unchanged.

Some referenced artifacts (model binaries, caches, intermediate receipts) are not shipped in this repository.

Status: development extraction, audit and bounded classical comparisons are complete; the local neural screen ended at its three-hour cap, and the private Kaggle GPU comparison completed with a budget-exhausted screen. Interrupted runs are preserved, with verified warm RR-cache reuse for the local neural restart. Unrestricted and challenge-constrained selections are recorded, and both classical artifacts were actually frozen before held-out access. All four frozen primary/external evaluations are complete; all predeclared endpoint checks and independent final evidence verification are complete. No forecasting superiority is claimed.

## 1. Current System

The preserved v2 release uses six causal RR features, a logistic classifier, independently fitted sigmoid calibration (C100), prior-only patient state, and a shared persistence/hysteresis/cooldown warning policy. History: 600 s; new onset: 120–600 s ahead; cadence: 30 s. AF episodes must last at least 30 s; post-AF washout is 600 s. Original quality settings and target remain the forecasting control. The synthetic test has four people and three manufactured events and was previously observed.

Exact baseline reproduction passed: event sensitivity 3/3, zero assessed false warnings, zero unassessable warnings, warning duration 0.741667 h (15.89% of 4.666667 monitored hours), and Brier 0.0023698411679117736. Source/model/results snapshot and hashes: baseline-a/manifest.json; baseline-a/reproduction.json.

## 2. Problems Found

- Strict SHDB interpretation supplies zero eligible local forecast events; 128 headers / 122 subjects / 98 annotations, only one ECG signal locally.
- Synthetic generator explicitly changes rate/variability before designed onsets; its score ceiling is not real forecast evidence.
- V2 selected-model Brier regressed versus prior release 0.00043126.
- Linked-person aliases changed macro-Brier selection weighting: reproducible example 0.276667 versus independent-group 0.41. Group-macro selection repaired.
- Assessment excluded off-target/unknown warnings but model selection ignored their burden. Ranking and reason-specific reporting repaired without relabelling unknown outcomes as false.
- Frozen membership was not bound to embedded models. New cohort contracts and integrity checks added; historical artifacts disclose their weaker contract.
- Fresh evaluation outputs and immutable held-out access entries now preserve repeated-access history.
- Audit quality-only availability ignored runtime baseline warm-up. Audit now follows that state and preserves annotation-event sensitivity denominators.
- Common-mode Gaussian noise passed the prior quality gate. Optional QRS-band kurtosis/saturation checks reject this counterexample but do not certify ECG quality.

## 3. Internet Research

- [IRIDIA-AF release 1.0.1](https://zenodo.org/records/8405941): reviewed AF boundaries, 167 records / 152 people, 200 Hz two-lead ECG, paired age/sex; release API lists CC BY 4.0. [Dataset paper](https://www.nature.com/articles/s41597-023-02621-1) describes correction of trailing noise, so original clock duration differing from corrected samples is not proof of internal gaps.
- [LTAF](https://physionet.org/content/ltafdb/1.0.0/): 84 long two-lead 128 Hz recordings, reviewed atr beat/rhythm references. External transfer is reserved; recording IDs alone do not establish a verified cross-record patient map.
- [WARN2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11240177/): RR recurrence representations support investigating dynamics, but retrospective variability-defined pre-AF and AF-present positives differ from this onset target. Headline accuracy is not directly comparable.
- [Gadaleta2023](https://www.nature.com/articles/s41746-023-00966-w): predicts AF occurrence within a 14-day monitoring period from at-home single-lead ECG without observed arrhythmia, using demographics and morphology. This motivates testing waveform information; it does not validate our 2–10-minute continuous warning endpoint.
- [Clifford signal-quality research](https://pmc.ncbi.nlm.nih.gov/articles/PMC2259026/): kurtosis is one quality index, not standalone clinical certification; our QRS-band implementation is a separately tested engineering hypothesis.
- [Official noise protocol](https://physionet.org/content/nstdb/1.0.0/old/): clean first 5 min, then alternating 2 min noise/clean with known beat references and six SNR levels.
- [Model-selection bias](https://www.jmlr.org/beta/papers/v11/cawley10a.html) and [external validation guidance](https://www.bmj.com/content/384/bmj-2023-074820) motivate grouped separation, uncertainty and untouched final cohorts.

## 4. Candidate Approaches

The matched current recipe refits the existing six-feature logistic helper at 30 s cadence with its original C 0.1/1 search, group/class weighting, and C100 independent calibration. A 120 s fitting-cadence control isolates overlap/cost reduction; calibration, policy, and outer evaluation stay at 30 s. New representations receive at most six seeded random configurations inside fitting patients. Neural fitting uses a bounded group/class-balanced window sample, whereas early stopping, calibration, policy and outer monitoring preserve natural within-group prevalence; this is a disclosed training-recipe comparison, not an architecture-only ablation.

No-warning and development-prevalence controls; current core logistic; isolated robust/multiscale/nonlinear feature groups and ECG-only prior-baseline personalization; elastic net; shallow XGBoost; normalized interval motifs; RR sequence TCN 16/32; small two-lead waveform CNN; one complementary blend if justified. Large pretrained models remain deferred because input leads, tasks, provenance and cost may not match.

## 5. Experiments

Primary membership is fixed before fitting: 121 development groups / 31 held-out groups, seed 42. The predeclared engineering throughput probe patient 000 is reserved wholly for development. The initial untrained allocation was archived as superseded before model fitting, with the correction reason recorded; no outcome-based selection was made. Reviewed annotations provide 284 eligible events across 143 people, including one cadence-unavailable onset retained in sensitivity. Positive labeled windows are 0.7495% before quality screening. Secondary minimum-duration endpoints 60/300 s are predeclared in endpoint-sensitivity-predeclared.json; frozen model/policy remained unchanged.

Completed: exact archived baseline reproduction; full NST detector comparison; explicit correlated-noise/clipping/dropout engineering counterexamples. The complete development audit covers 132 recordings, 121 independent groups, 629,444 decision rows, 5,243.98 monitored hours and 3,734.85 eligible hours. 604,231 decisions are available; 3,613 accepted windows are positive. 233/241 events have at least one available eligible decision (96.68% availability ceiling, ignoring persistence/FAR/learnability). The first bounded classical run completed 13 model/control types in 2,394.23 s (including a soft-deadline overrun); elasticnet ended before any complete fold. The supplementary original weighting/scaling/missingness and isolated-feature run completed seven additional candidates in 1,242.77 s, including soft-deadline overrun; motif augmentation completed 1/3 folds and was excluded from that run. After the user requested parallel Mac work, a fresh unchanged three-fold motif run completed in 204.89 s under a 600 s cap. Its repeated fold 0 exactly matches the preserved original probabilities and decompressed prediction stream; both runs are charged, with no duplicate fold pooled as independent evidence. Local MPS RR16 now has all three completed outer folds, detecting 4/241 events with 6.25 warning hours across the same 5,243.98 monitored hours. This complete backend result is retained as fallback/portability evidence; the complete CUDA RR16 result takes precedence in unified selection. The local screen ended after 10,802.69 s: RR16 is complete, waveform CNN completed two of three folds and is excluded, and RR32 was not reached. The private CUDA comparison completed after 20,884.37 neural-screen seconds. RR16 has three complete folds; waveform has two and is excluded; RR32 and optional elastic net were not reached. The declared NN screen is 4 epochs / at most 10,000 fitting windows / batch 64; the original longer defaults are not the executed recipe. The first completed RR fold attempted four epochs and restored epoch 1 by independent-group early-stop loss; patience 5 cannot trigger within four epochs. Its 123.92 s fitting/normalization/stop timing is nested within 3,249.39 s full-fold cost, including 2,321.09 s outer prediction/IO; these times must not be added. Incremental independent review verifies disjoint 48 fit / 8 stop / 12 calibration / 12 policy / 41 outer groups, exact classical outer rows, causal inputs and checkpoint contracts (scientific-review-neural-incremental.json). Private CUDA is primary for complete neural candidates, with local MPS fallback and same-seed backend repeats disclosed (neural-backend-predeclaration.json). Full MIT-BIH comparison and local MPS neural engineering smoke are complete. Primary budget: 0.25 assessed false warnings / eligible hour; frontiers: 0.1/1; minimum sensitivity: 0.5 is provisional engineering acceptance, not clinical approval. Unknown-warning burden is reported separately.

## 6. Results

Completed bounded classical development comparison (9 October): nineteen candidates and two controls have all three outer folds after the original motif completion (classical-complete-pool-summary.json). Independent review reconciles pooled counts/Brier, checksums, group coverage and ranking for the preserved first two runs (scientific-review-resumed.json/.md); the motif completion has a separate review (scientific-review-motif-backfill.json/.md). The reproducible merge preserves original summaries and verifies duplicated controls exactly (merge-motif-backfill.py). Total recorded classical elapsed time is 3,841.88 s, including the repeated motif fold. Both neural screens ended at their caps; the final primary/external results appear below. Every row covers 121 development groups and 241 eligible events. Assessed FAR uses labeled forecast-eligible hours, including quality-rejected time; current AF, washout, gaps and unknown outcomes cannot dilute that denominator. Occupancy and time coverage use all monitored hours. Shared time coverage is 96.00%.

| Candidate | Detected/events | FAR/eligible h | All warnings | Occupancy | Group Brier | AUROC | PR-AUC | Model-fold wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no_warning | 0/241 | 0.0000 | 0 | 0.00% | 0.01232753 | 0.5000 | 0.008429 | 0.00 |
| training_prevalence | 0/241 | 0.0000 | 0 | 0.00% | 0.01219389 | 0.4336 | 0.007405 | 0.00 |
| logistic_current_recipe | 7/241 | 0.1186 | 624 | 21.84% | 0.01220163 | 0.4598 | 0.008476 | 94.49 |
| logistic_core | 7/241 | 0.1285 | 655 | 25.88% | 0.01220217 | 0.4599 | 0.008390 | 80.98 |
| logistic_engineered | 4/241 | 0.1449 | 656 | 34.66% | 0.01220901 | 0.4527 | 0.007986 | 198.57 |
| logistic_combined | 6/241 | 0.1706 | 761 | 40.78% | 0.01220871 | 0.4531 | 0.007979 | 182.53 |
| logistic_availability_control | 4/241 | 0.1449 | 656 | 34.66% | 0.01220901 | 0.4527 | 0.007986 | 173.16 |
| logistic_clinical | 3/241 | 0.0913 | 386 | 34.59% | 0.01220361 | 0.4631 | 0.007708 | 89.75 |
| xgboost_core | 2/241 | 0.0747 | 296 | 26.81% | 0.01220668 | 0.4486 | 0.007387 | 110.18 |
| xgboost_engineered | 1/241 | 0.0977 | 397 | 29.61% | 0.01220830 | 0.4492 | 0.007534 | 202.22 |
| xgboost_combined | 3/241 | 0.0993 | 425 | 25.16% | 0.01220618 | 0.4358 | 0.006802 | 309.09 |
| xgboost_personal | 2/241 | 0.0546 | 217 | 1.05% | 0.01220674 | 0.4635 | 0.007169 | 585.61 |
| blend_core_xgboost | 4/241 | 0.0739 | 362 | 28.08% | 0.01220304 | 0.4634 | 0.007714 | 843.16 |
| logistic_core_patient_only | 2/241 | 0.1098 | 483 | 35.73% | 0.01220841 | 0.4593 | 0.007569 | 130.56 |
| logistic_core_window_class | 5/241 | 0.0846 | 388 | 20.55% | 0.01220939 | 0.5227 | 0.010268 | 121.88 |
| logistic_core_robustscale | 7/241 | 0.1285 | 654 | 25.57% | 0.01220205 | 0.4601 | 0.008388 | 118.50 |
| logistic_core_no_missing | 7/241 | 0.1280 | 652 | 25.87% | 0.01220200 | 0.4599 | 0.008393 | 146.16 |
| logistic_dispersion | 3/241 | 0.0672 | 328 | 40.33% | 0.01220581 | 0.4736 | 0.008444 | 156.23 |
| logistic_dynamics | 6/241 | 0.0817 | 402 | 6.97% | 0.01220436 | 0.4869 | 0.008950 | 170.07 |
| logistic_nonlinear | 10/241 | 0.1009 | 545 | 13.26% | 0.01220478 | 0.4588 | 0.008289 | 180.33 |
| logistic_motifs | 7/241 | 0.1111 | 530 | 31.53% | 0.01221452 | 0.4809 | 0.008332 | 140.77 |

The prevalence control is constant within each fold, but differs across folds; its pooled AUROC of 0.4336 reflects cross-fold prevalence variation.

Model-fold wall costs sum the legacy `training_and_selection_seconds` field, which spans fitting, calibration, policy search, outer inference, diagnostic metrics and artifact writes. The separately reported inference timer is nested within that total and must not be added. Run-level setup, controls and pooling overhead are additional. The blend includes its two members, so model-row costs must not be summed as total investigation cost. No-warning/prevalence controls have no separately timed fit. Cached-feature prediction for the current recipe takes 12.32 s across 604,231 available decisions (mean 0.0204 ms/decision); this excludes ECG reads/feature extraction and differs from single-decision evidence-generation latency. Per-model fold inference timers are retained in the experiment logs. Neural full prediction timings include source reads and sequence construction, so they are not directly comparable to this cached-feature number.

The current recipe remains selected at 7/241 events (2.90%), 624 total warnings and 21.84% occupancy; every candidate fails the 50% provisional engineering gate. Prevalence-control group Brier is numerically lower than all completed candidates. A low assessed warning-start rate does not imply low overall burden. Probability-positive prevalence 0.8429% supplies the descriptive PR-AUC reference. These grouped outer results are development evidence; measurement results follow separately. A separate future-label oracle, explicitly nondeployable and excluded from selection, matches 232/241 events (96.27%) with zero false starts and 0.50% occupied time under the same availability/persistence/cooldown. Eight events have no available opportunity; one additional event is lost under this oracle policy. This diagnostic confirms the alert machinery can reward perfectly informed risks; learnable predictive information remains unestablished (development-oracle-diagnostic.json).


Development operating frontiers for the current recipe (thresholds chosen separately inside each fold):

| Policy-group FAR budget / h | Detected/events | Observed outer FAR / h | All warnings | Occupancy |
|---|---:|---:|---:|---:|
| 0.1 | 2/241 | 0.0115 | 50 | 0.87% |
| 0.25 | 7/241 | 0.1186 | 624 | 21.84% |
| 1 | 6/241 | 0.1906 | 865 | 62.96% |

These outer frontiers can be nonmonotonic because fitted risks, threshold persistence and warning starts change. The budget names specify policy-group selection constraints; observed outer FAR remains an evaluation result. Source: `classical-current-frontier.json`.

Local MPS fallback/portability result (CUDA precedence remains fixed before selection):

| Candidate | Detected/events | Assessed FAR / eligible h | All warnings | Occupancy / monitored time | Group Brier | AUROC | PR-AUC | Full candidate s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rr_tcn_16 | 4/241 | 0.0367 | 185 | 0.119% | 0.01220279 | 0.4847 | 0.009391 | 6192.99 |

Independent review reconciles all 629,444 decisions, 121 groups, probability arrays and counters; the 237 misses retain eight events without available opportunity. Of 185 warnings, 137 are assessed false, 44 occur in known excluded periods and none have unknown outcomes. Fold 2 selects threshold 1 and emits no warnings. The full RR candidate costs 6,192.99 s, including nested fitting/normalization/stopping of 527.17 s and outer prediction/IO of 4,225.75 s; those nested times must not be added. The complete local run also includes 3,620.12 s for two waveform folds, 872.13 s for the interrupted waveform fold and 117.45 s outside trials. Only RR16 is eligible for pooling (`scientific-review-neural-incremental.json`).

Completed CUDA primary result (same predeclared recipe; runtime/library differences prevent a pure hardware comparison):

| Candidate | Detected/events | Assessed FAR / eligible h | All warnings | Occupancy / monitored time | Group Brier | AUROC | PR-AUC | Full candidate s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rr_tcn_16 CUDA | 4/241 | 0.0300 | 154 | 0.088% | 0.01220082 | 0.4868 | 0.009565 | 9092.06 |

Of 154 warnings, 112 are assessed false, 37 occur in known excluded periods and one has an unknown outcome. Warning duration is 4.625 h. All 23 aggregate audit checks and all five completed folds pass (`scientific-review-gpu-final.json/.md`). Checkpoints intentionally serialize CPU state dictionaries; CUDA fit receipts and the verified placement code establish actual GPU training. The canonical merge preserves both runs and incomplete artifacts; complete CUDA RR16 is primary, with no backend fold mixing. Unrestricted unified selection retains `logistic_current_recipe`; the new, explicitly post-development challenge constraint selects `logistic_combined` among complete genuine clinical-plus-ECG candidates. Neither meets the engineering acceptance gate. The original selection remains unchanged (`unified-selection.json`; `challenge-selection.json`).

| System | Task | Sensitivity | Precision | Cost | Notes |
|---|---|---:|---:|---|---|
| V2 archived logistic | Manufactured AF events |1.00|1.00 assessed|See original evaluation|Three events; Brier 0.00236984; not clinical evidence |
| V2 beat detector | All NST clean/noise portions, both leads |0.96949|0.81895|0.68s detector-only total|1,558 misses; 10,944 extra beats |
| WFDB XQRS | Same NST signals/matching |0.96913|0.88215|31.63s detector-only total|1,576 misses; 6,611 extra beats; different cost |
| V2 beat detector | Full MIT-BIH, 48 records, both leads |0.96997|0.94748|2.68s detector-only total|6,560 misses; 11,745 extra beats |
| WFDB XQRS | Same full MIT-BIH signals/matching |0.98122|0.97967|128.54s detector-only total|4,103 misses; 4,447 extra beats |

Detector comparison figure (measurement/detector-comparison.png; image not shipped).

Counts pool repeated noise variants, not independent people; clean/noise and SNR strata are separately saved. Timing includes detection only and comes from this local run.

### Frozen primary and external results

Both artifacts were frozen before these once-only cohort accesses. Primary: 31 audited patient groups / 35 recordings, 1,446.4988 monitored and 1,211.7000 eligible hours. External: all 84 LTAF record groups, 1,960.6000 monitored and 483.3250 eligible hours. All 173,623 primary and 235,356 external chronological decisions are retained, including unavailable and excluded periods. Clinical metadata is present in all primary records and absent from all external records; this is a missing-modality transfer test.

| Frozen model / cohort | Detected/events | Sensitivity, 95% group interval | Assessed false / eligible h | All starts | Warning h / occupancy | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| Fusion / primary | 1/43 | 2.33% [0.00, 7.89] | 103 / 0.08500 | 113 | 24.457 / 1.69% | 92.89% |
| Fusion / external | 2/29 | 6.90% [0.00, 16.67] | 4 / 0.00828 | 398 | 120.783 / 6.16% | 89.37% |
| ECG reference / primary | 0/43 | 0.00% [0.00, 0.00] | 0 / 0.00000 | 0 | 0.000 / 0.00% | 92.89% |
| ECG reference / external | 0/29 | 0.00% [0.00, 0.00] | 0 / 0.00000 | 0 | 0.000 / 0.00% | 89.37% |

| Frozen model / cohort | Labeled available windows | Window Brier | Group Brier | AUROC | PR-AUC / prevalence reference | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Fusion / primary | 135188 | 0.00442000 | 0.00665748 | 0.6110 | 0.00625 / 0.00442 | 0.004227 |
| Fusion / external | 48479 | 0.00701545 | 0.01205468 | 0.6309 | 0.03531 / 0.00708 | 0.000973 |
| ECG reference / primary | 135188 | 0.00441734 | 0.00666074 | 0.5453 | 0.00542 / 0.00442 | 0.003208 |
| ECG reference / external | 48479 | 0.00700850 | 0.01205449 | 0.6681 | 0.01881 / 0.00708 | 0.000556 |

Intervals use 300 whole-group bootstrap draws; external groups are recording IDs, not verified independent people. The reference sensitivity interval [0,0] is a degenerate resampling result after zero observed detections, not proof of zero population sensitivity or precise safety. Only 46 external groups contribute labeled available windows to probability metrics. Separate intervals and three observed fusion matches do not establish a paired superiority claim. The held-out results do not promote or retune either artifact.

Fusion primary: 42 misses, including 3 with no available warning opportunity; 103 assessed false starts, 9 known-excluded starts (1 current AF, 8 washout), no unknown starts. Its single matched warning leads onset by 496.225 s. External: 27 misses, including 3 with no available opportunity; 4 assessed false starts, 299 known-excluded starts (222 current AF, 77 washout) and 93 unknown-outcome starts. Two matched external warnings have median lead 356.914 s, with only one of 29 total events receiving at least five minutes of lead. The 392 excluded/unknown starts must accompany the low external assessed FAR; they are not silently relabeled as false or omitted from burden.

| Frozen model / cohort | Predeclared budget | Detected/events | Observed assessed FAR | All starts | Warning h |
|---|---:|---:|---:|---:|---:|
| Fusion / primary | 0.1 | 0/43 | 0.00000 | 0 | 0.000 |
| Fusion / primary | 0.25 | 1/43 | 0.08500 | 113 | 24.457 |
| Fusion / primary | 1 | 0/43 | 0.13287 | 192 | 62.865 |
| Fusion / external | 0.1 | 0/29 | 0.00000 | 0 | 0.000 |
| Fusion / external | 0.25 | 2/29 | 0.00828 | 398 | 120.783 |
| Fusion / external | 1 | 2/29 | 0.02483 | 528 | 208.808 |
| ECG reference / primary | 0.1 | 0/43 | 0.00000 | 0 | 0.000 |
| ECG reference / primary | 0.25 | 0/43 | 0.00000 | 0 | 0.000 |
| ECG reference / primary | 1 | 3/43 | 0.20385 | 271 | 139.948 |
| ECG reference / external | 0.1 | 0/29 | 0.00000 | 0 | 0.000 |
| ECG reference / external | 0.25 | 0/29 | 0.00000 | 0 | 0.000 |
| ECG reference / external | 1 | 0/29 | 0.09931 | 280 | 445.550 |

These are policies selected on development policy groups, not thresholds optimized on these tests. The reference 1/hour frontier detects 3/43 primary events with observed FAR 0.20385/hour, but it cannot be adopted as a new 0.25/hour policy after seeing this test. The fused primary 1/hour frontier detects none despite more warnings; persistence and warning-start matching can produce nonmonotonic test sensitivity. No preferred endpoint/frontier was selected post hoc.

Completed source reports: [fusion primary](frozen-cohorts/primary-evaluation/evaluation.json), [fusion external](frozen-cohorts/external-evaluation/evaluation.json), [reference primary](reference-evaluations/primary-evaluation/evaluation.json), [reference external](reference-evaluations/external-evaluation/evaluation.json). Frozen failure analysis preserves denominators, source shift, every group and all count-maxima ties.

### Predeclared episode-duration sensitivity

The same frozen forecasts and warning starts were scored against the predeclared minimum AF episode durations; no fitting, policy changes or endpoint selection followed these results.

| Frozen model / cohort | At least 30 s | At least 60 s | At least 300 s | Unchanged warning starts / occupied h |
|---|---:|---:|---:|---:|
| Fusion / primary | 1/43 | 1/42 | 1/41 | 113 / 24.4567 |
| Reference / primary | 0/43 | 0/42 | 0/41 | 0 / 0 |
| Fusion / external | 2/29 | 0/19 | 0/8 | 398 / 120.7833 |
| Reference / external | 0/29 | 0/19 | 0/8 | 0 / 0 |

The two external matches do not survive the longer-duration definitions. Fusion primary assessed false-warning rates at 60/300 s are 0.085014/0.085023 per eligible hour; external rates are 0.008290/0.008315. Denominators and warning assessability change with endpoint labels: external unassessable starts rise from 392 to 394 while total starts and occupied time remain fixed. A low assessed FAR alone cannot summarize this burden. Each evaluation folder contains its bound `endpoint-sensitivity.json`.

## 7. Ablation Results

Quality-counterexample ablation: unchanged clean ECG accepted by both gates; duplicated Gaussian noise and severe clipping accepted by old gate/rejected by optional candidate; 20 s dropout rejected by both. NST candidate changes accepted windows 36/48 to 35/48, indicating limited broader benefit. On full MIT-BIH both quality settings accept 133/192 chunks. Neither gate has been promoted on forecasting evidence.

Development clinical ablation: adding age/sex to engineered ECG gives 6/241 events versus 4/241 for ECG alone, but both remain below the current recipe 7/241; fusion increases occupancy 34.66%→40.78% and FAR 0.1449→0.1706 / eligible hour. This does not establish fusion benefit. Both availability flags are constant 1, and fitted coefficients near 0 explain identical ECG/availability-control outcomes; metadata-missingness benefit is not estimable in this cohort. The current detector fails particularly on second leads of records 104/200/208; XQRS improves aggregate beat accuracy but also fails on some second leads and takes 48 times as much detector-only wall time. Record 201/202 shares a person; repeated NST variants are derived from 118/119. Full benchmark numbers are descriptive measurement evidence, not independent forecasting patients.

Weighting/scaling/missingness ablations do not support replacement under the declared selection rule. Nonlinear-only logistic detects 10/241 versus 7/241 events for the current recipe and reduces occupancy 21.84%→13.26%, but paired sensitivity CI spans −1.47 to +3.68 percentage points. The legacy `warning_occupancy_ci95` interval measures warning hours / eligible hours, whereas table occupancy measures warning hours / all monitored hours. Its lower warning-duration burden CI cannot override uncertain unequal sensitivity under the predeclared rule. Patient-only weighting detects 2 events, window/class weighting 5; robust scaling and removing missingness each detect 7 while having higher warning-start rates than the current recipe. Dynamics detects 6 with 6.97% occupancy. The completed motif-augmented model adds interval motifs to engineered ECG features, rather than using motifs alone. It ties the current recipe at 7/241 events and reduces starts to 530, but raises occupancy to 31.53%. Paired differences in sensitivity, assessed FAR, total-warning rate and group Brier all span zero; warning-duration burden per eligible hour is worse (95% CI +0.0154 to +0.2584 h/h). No supported tied-sensitivity secondary improvement justifies replacing the current recipe.

## 8. Best Architecture

The unrestricted development rule retains the six-feature ECG logistic reference. The challenge requires static clinical plus dynamic sensor fusion, so a separate recorded pre-held-out constraint selects engineered ECG plus genuinely paired age/sex logistic regression. This compliance-driven selection does not establish a fusion gain. Both artifacts were frozen before held-out access: the scientific reference at 2026-10-09 20:38:33 UTC and challenge fusion at 20:39:14 UTC (`freeze-finalists-receipt.json`). Clinical/ECG input dependence and future-perturbation replay checks pass; these software checks do not establish physiological causality or predictive benefit. The synthetic demo remains distinct and public deployment gates stay closed. See [official requirements audit](CHALLENGE_REQUIREMENTS.md).

## 9. Final Pipeline

Implemented pipeline: source-verified continuous ECG and paired baseline age/sex → past-only device-calibration masking, beat detection and signal quality → causal engineered ECG features and prior-only patient baseline → fitting-group standardization and logistic regression → independently fitted sigmoid calibration → frozen persistence/hysteresis/cooldown policy → complete chronological monitoring, retaining abstentions and excluded/unknown warning burden. The challenge artifact has 27 schema fields (54 matrix inputs including missingness indicators); the separate scientific reference has six schema fields. The twin baseline evolves as state, but baseline-deviation features are absent from the selected challenge predictor. No weights, calibration or policy are updated on primary or external cohorts. Missing external age/sex remains missing and follows the training-fitted preprocessing. Future ECG and annotation perturbations leave earlier replay outputs unchanged (`frozen-challenge-replay.json`); default serving correctly rejects the failed-gate research artifact.

## 10. Failure Analysis

The frozen ECG reference issues no warnings because its development-policy search selected threshold 1.0: all 35 budget-qualifying policy candidates detected zero of 24 policy-group events, so the declared burden/simplicity tie-break selected no warnings. Actual maximum calibrated risk is 0.06037 primary and 0.06234 external, both below 1.0 (`frozen-score-range-diagnostic.json`; `frozen-reference/policy_search.json`). This is a measured policy/development failure, not a GPU execution failure. Thresholds stay frozen. Fusion has 2,303 primary and 11,187 external available scores crossing its 0.013468 threshold; persistence and matching still leave only 1/43 and 2/29 useful forecasts.

Primary fusion misses 39 events despite an available opportunity and 3 without one; external fusion misses 24 despite an available opportunity and 3 without one. The metadata cannot distinguish zero eligible opportunities from opportunities that were all unavailable, so that finer count remains unestimated. Lead disagreement rejects 94.980 primary and 189.892 external hours. Primary group patient_001 misses all 4 eligible events, patient_039 supplies 28 of 103 false starts, and patient_028 has 18.793 rejected hours. External record120 misses all 4 events; record103 has 22.867 rejected hours and no eligible events. These are descriptive count maxima with denominators and ties saved, not physiological causes or inferential rankings. All age/sex values disappear at external transfer and fs changes 200 to 128 Hz; source/lead gain correspondence remains incompletely verified.

Current observed failures include noisy extra beats and label-coverage limitations. The retained classical recipe generates 443 assessed false warnings, 162 starts during current AF, 6 during washout, 2 during gaps and 4 unknown-outcome starts. Seven starts match eligible events; occupied warning time is 1,145.38 hours. Lead disagreement accounts for 182.26 of 209.995 rejected monitoring hours. Abstention-related misses remain in the 241-event denominator (classical-development-failure-analysis.json). Completed development folds already show weak/nontransportable score association: several independent calibration fits learn a negative slope and reverse raw ranking on outer patients. Raw weighted scores are poorly calibrated; C1/C100 calibration differences are small. These observations do not justify choosing a calibration variant post hoc on outer outcomes. Warning occupancy can be high while new starts and useful event matches remain scarce; a continuous warning begun outside 120–600 s does not retrospectively count as a forecast for a later event. Policy-group-selected frontiers can be nonmonotonic on outer patients. The baseline misses 234 events, including eight with no available opportunity, and has 1.56% assessed warning precision. Pooled window Brier is 0.008377, group Brier 0.012202, and 10-bin ECE 0.001242. All but three accepted eligible risks fall in the lowest probability bin (mean risk 0.9668%, observed prevalence 0.8429%); this rare-event aggregate does not establish useful event prediction or subject-level calibration. Raw/C1/C100 and fold-specific ranking diagnostics are retained in `classical-calibration-diagnostic.json` and the original experiment log. Final test failures will not be used for retuning.

## 11. Limitations

Paired group-bootstrap intervals are descriptive: shared training folds induce dependence, and repeated candidate searches are not corrected for multiplicity. Pooled outer AUROC also mixes three independently calibrated models; fold-specific calibration diagnostics are saved separately. Final held-out intervals must retain the independent-group and external identity limitations.

Full IRIDIA archive MD5 verification and extraction are complete (167 recordings, 289 daily files, 6,690.476 monitoring hours); all four primary/external forecasting evaluations are complete, with poor event detection and unchanged failed readiness gates. The native Kaggle connection now passes the actual authenticated quota tool (9 October, 10:16 UTC); the earlier cached connection had returned unauthenticated. The private uploaded dataset is ready; all 718 raw file names/sizes and 28 autoexpanded bundle members match the approved payload. GPU version 1 completed. Independent returned-artifact review confirms source/cache/raw checksums passed within the training bootstrap (kaggle-access-state.json; kaggle-upload/remote-reconciliation-20261009.json). Dual Tesla T4 inventory was verified; the submitted code schedules only one CUDA device and live utilization remains unmeasured. Current source now adds phase/counter logs and bounded NVIDIA sampling (GPU_DEBUGGING.md), but those changes are absent from the immutable version 1 bundle. The smoke session was stopped after 198.91 s. No forecasting GPU training result is inferred from that test. A real local MPS tensor operation passed outside the sandbox. Budgets: 8 local experiment hours, 6 GPU hours and 25 GB peak dataset storage; hardware: 16 GB RAM. Local prior/interrupted costs include estimates; exact partial neural cost is unavailable. Original path-sum storage samples double-counted 719 hard links used for the private upload; unique-inode allocated-block monitoring replaces those samples (storage-accounting-correction.json). This is filesystem storage, not process RAM. The completed classical process measured 2.673 GB peak RSS. Local optional dependencies are frozen in uv.lock. The Kaggle bootstrap uses bounded version ranges against its supplied runtime and records actual versions; it does not install the local lock verbatim. Backend repeats therefore confound hardware with library/runtime differences and are not pure hardware comparisons. Live GPU bootstrap logs confirm Torch 2.11.0+cu128/CUDA 12.8, NumPy 2.1.3 and SciPy 1.16.3; local versions are Torch 2.14.1, NumPy 2.5.3 and SciPy 1.18.1. Other logged scientific library versions match. Returned fold artifacts independently verify all five CUDA fits and the full cache/raw-source guard (`scientific-review-gpu-final.json`). Historical utilization percentages remain unavailable. Total GPU allocation including smoke was 21,212.2753 s (5.8923 h), within the six-hour allowance. RR16 outer scoring/source I/O occupied 6,105.15 of 9,092.06 trial seconds (67.15%); this is not kernel-only time. Optional elastic net did not start. Mixed Torch 2.14.1 / XGBoost 3.4.1 native OpenMP runtimes crashed during integrated testing on macOS; startup OMP_NUM_THREADS=1 passed both minimal import-order reproductions. [Upstream OpenMP report](https://github.com/pytorch/pytorch/issues/155201) documents the library conflict; our current-version crash direction differs, so we use the verified thread constraint rather than assume import order suffices. [Author plotting code](https://github.com/cedricgilon/iridia-af/blob/main/iridia_af/record.py) treats IRIDIA HDF5 values as mV; all 168 LTAF header channels specify mV, and WFDB performs physical gain conversion. No extra numeric scaling is justified. Independent IRIDIA export-gain validation and LTAF I/II lead correspondence remain unverified (waveform-unit-provenance.json). Dataset populations do not establish Indian clinical generalization.

Final experiment accounting totals **27,965.18 local seconds (7.7681 h)** of the eight-hour allowance, leaving an estimated **834.82 s**. This sums prior disclosed estimates and measured stage receipts once, including both failed transports, the final-file recovery and repeated motif fold; the inactive overnight gap is charged zero. Audit, documentation and PDF production are separate software work, not model-experiment time. Successful external/primary cache construction took 1,569.60/1,693.15 s; fused evaluations took 32.60/39.64 s. The reference evaluations plus all predeclared endpoint checks completed in 465.02 s under a 1,299.84 s outer watchdog (`reference-endpoints-watchdog-receipt.json`). Native fitting/source deadlines remain soft; outer watchdogs bound the owned processes. No further GPU allocation was used.

Corrected unique-device/inode storage samples reached **23,649,918,976 allocated bytes (23.650 decimal GB)**, below the 25 GB dataset allowance. This is a sampled peak, not a proof against brief unobserved peaks or process RAM. Original naive path sums are preserved and excluded because they double-count upload hard links. Independent final evidence verification checks model bytes, freeze/access ordering, exact cohort/cache contracts, all four compressed prediction streams and endpoint warning invariance; the independent audit passed, with its machine-readable receipt. Recorded diagnostics verification passed 178 tests; the bound core source is unchanged, while current one-off helpers receive fresh finite self-checks and focused lint checks.

## 12. Judge Verdict

These scores are subjective reviewer judgments on a ten-point scale, not an organizer-published rubric or clinical validation.

| Dimension | Score / 10 | Evidence and practical limit |
|---|---:|---|
| Technical quality | 8 | Causal/grouped contracts, calibrated policies and immutable held-out access; measurement remains imperfect. |
| Novelty | 3 | Standard logistic/TCN/CNN components; careful digital-twin engineering is the main contribution. |
| Performance | 1 | Fusion detects 1/43 and 2/29 held-out events; both development acceptance gates fail. |
| Data use | 6 | Real continuous ECG and genuinely paired age/sex; limited clinical fields and external missing-modality shift. |
| Robustness | 5 | Rejection, exclusions and transfer are measured; many misses and device/lead limitations remain. |
| Reproducibility | 8 | Saved identities, split/roles, runtime versions, source receipts and repeatable commands; compute and historic partial costs are disclosed. |
| Engineering | 8 | Persistent twin, bounded diagnostics, restart/leakage tests and serving gates; hospital deployment is outside verified scope. |
| Practicality | 3 | Local replay is usable and fast on its measured prefix; forecasting is insufficient for care. |
| Explainability | 7 | Logistic inputs, policy, abstention and warning evidence are inspectable; coefficients are not causal effects. |
| Evidence | 6 | Real frozen primary/external evaluation and independent audits; few matched events, unverified external identities and no prospective Indian validation. |

**“Is the current implementation genuinely close to the best solution we can reasonably build from the available data and resources, or are we leaving meaningful performance on the table?”**

It is not demonstrated to be close to the best reasonable forecasting solution. The measured result is a reproducible research prototype with poor event capture, not a successful clinical forecaster. Meaningful untested opportunities remain: measurement errors, training/sample/optimization limits, morphology, calibration transport and policy stability. Their gains are hypotheses, not achieved improvements. The four-epoch/10,000-window neural screen cannot rule out better deep learning; waveform completed only two folds, RR32 and later seeds were not reached, and elastic net remains incomplete. More GPU allocation alone will not fix CPU/input bottlenecks or missing predictive signal.

Keep this frozen study intact. Any next model/endpoint revision belongs to a newly declared development cycle with an appropriate fresh validation plan; these held-out failures cannot be used as a tuning set while retaining an untouched-test claim. Challenge delivery can demonstrate genuine clinical/sensor software fusion, causal historical replay and a conceptual doctor UI, while openly reporting weak forecasting and failed gates. Team identity, actual video, public publication, registration and submission remain pending external delivery steps.
