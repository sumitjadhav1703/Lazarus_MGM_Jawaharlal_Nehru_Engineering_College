# CardioTwin ML improvements: measured findings

**Current investigation:** The IRIDIA/LTAF investigation and frozen clinical-plus-ECG finalist supersede this earlier SHDB/synthetic iteration. See the [current report](../reports/ml-investigation/report.md), [frozen model card](../reports/ml-investigation/frozen-challenge-fusion/model_card.md) and [challenge requirements](../reports/ml-investigation/CHALLENGE_REQUIREMENTS.md). The dated findings below are preserved as historical evidence.

7 October 2026. Research and implementation are complete for this development iteration. Results distinguish detector accuracy, synthetic forecasting checks and unavailable clinical evidence. The selected v2 software release is `artifacts/ml-v2-reviewed/patient_class`, core ECG logistic model `cardiotwin-0.1-logistic_ecg-51c17e8a4b3d`. Its calibration C is 100. This package was selected before test evaluation; test comparisons below did not promote a different model.

## Findings that changed the implementation

The most concrete defect was in measurement, upstream of machine learning. The old detector based height/prominence partly on the largest filtered-energy sample in the entire trailing history. A large artifact in SHDB 005 raised the threshold above ordinary beats: the two leads yielded 907 and 45 peaks. More model complexity cannot recover RR information that this threshold erased. The causal filter is retained, but threshold scale now uses the 99th percentile. The lead-agreement check also matches beats symmetrically and one-to-one instead of allowing reused matches. Quality thresholds are unchanged.

On the genuine 2220–2820 second window, counts became 908/927 and disagreement fell from 0.98126 to 0.04423. The window now passes the same provisional quality gate. This establishes a repaired engineering behavior, not accurate QRS localization: SHDB's supplied QRS files lack independently reviewed reference beats. [Raw feature/quality measurements](../reports/ml-improvements/shdb-feature-check.json) include ten-run extraction timing: median 17.29 ms, p95 17.65 ms.

To obtain stronger beat ground truth, I used reviewed beat annotations from the separate [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/). This resource is more relevant for detector verification than unreviewed QRS annotations; it is not a replacement fused AF forecast cohort. The first 600 seconds of records 100 and 201 were compared with 150 ms one-to-one tolerance, excluding filter startup and the unresolved edge. The fixed detector parameters were shared across records. [Benchmark evidence and local file hashes](../reports/ml-improvements/beat-benchmark.json):

| Record / lead | Previous sensitivity | v2 sensitivity | Previous precision | v2 precision | Trade-off |
| --- | ---: | ---: | ---: | ---: | --- |
|100 /1 |100.00% |100.00% |100.00% |100.00% | No change |
|100 /2 |99.60% |99.60% |100.00% |100.00% | Three misses remain |
|201 /1 |98.68% |100.00% |100.00% |98.19% | Ten misses removed;14 extra beats |
|201 /2 |81.18% |99.87% |100.00% |100.00% |143 misses reduced to 1 |

The improvement is lead dependent. Percentile scaling can admit extra peaks, so this two-snippet result does not validate the detector across devices, artifact severity, ectopy, or the complete MIT cohort. No manual timing ground truth exists for the repaired SHDB window. A broader annotated development benchmark remains the next measurement priority.

## Implemented feature engineering

Seventeen new numeric ECG fields expand six core fields to 23:

- Robust RR median, interquartile range, median absolute deviation and coefficient of variation.
- Poincare SD1/SD2, lag-one RR correlation, and sample entropy.
- HR, RR CV and RMSSD over trailing 60/300 second windows, plus each short-minus-long difference.

These measure recent dynamics instead of relying only on whole-history averages. Personal variants also receive deviations and zscores from prior reliable baseline means/variances. At most 101 allowlisted original fields enter the enriched fusion/personal comparison, plus a missingness indicator for each. Clinical timing restrictions and exclusion of labels, filenames, IDs and future diagnoses are retained.

Missing primary-lead samples invalidate RR intervals crossing them, even when the resulting RR duration falls within plausibility bounds. Consecutive-pair metrics and entropy cannot bridge invalid gaps. Short windows require ten valid intervals and 80% plausibility. SampEn uses m=2 and tolerance 0.2 standard deviations; it requires 50 contiguous intervals and caps work at 256. Estimates without matches stay missing. The [original SampEn paper](https://journals.physiology.org/doi/10.1152/ajpheart.2000.278.6.h2039) supports the statistic, but neither our minimum nor cap is clinically validated. The implementation retains bounded exact O(n²) comparisons rather than adding a new dependency.

Baseline population variances and field counts update after prediction. Zscores require five earlier field updates and variance above 1e-12 squared feature units; this avoids enormous values from floating-point residual variance in an effectively constant baseline. Overlapping baseline windows still do not establish independent confidence. RR statistics are not certified normal-to-normal HRV, and no interval pattern is labelled a PAC without beat-morphology validation.

## Learning changes and controlled results

The candidate pool grows from six to ten, adding enriched ECG logistic regression, combined/personal elastic net, and enriched personal XGBoost. [Elastic net](https://hastie.su.domains/Papers/elasticnet.pdf) is a practical hypothesis for correlated RR/multiscale inputs. A small regularization search, fixed L1 ratio 0.5, grouped preprocessing and convergence checks constrain the search. XGBoost retains grouped early stopping and a training-selected fixed tree count. No neural model, SMOTE or online supervised adaptation was added: available independent events do not justify them.

Default training weights balance audited independent groups and their observed class cells. All linked patient IDs share one mass. Weights are normalized to mean one and recomputed within each fitting fold; global positive/negative mass is not necessarily equal. On these fixtures, each of 12 training groups has 69.75 total mass and weighted positive prevalence is 37.5%, versus natural training prevalence 146/837=17.44%. The four calibration groups retain natural prevalence 48/282=17.02%, with no balancing weights. Calibration learns on independent patients, consistent with [scikit-learn's calibration guidance](https://scikit-learn.org/stable/modules/calibration.html).

Audited train/calibration/validation/test assignments remain exactly the previous 12/4/4/4. Three declared development regimes each fit ten candidates: window-class/C100, group-balanced/C100, and group-balanced/C1. Every candidate converged, with complete OOF coverage of 12 groups and both classes; failures would remain archived and be excluded from export. Final-test features and labels were excluded from all these fits and decisions. Window-class is a v2 control, not an exact recreation of the previous source: all regimes share the new detector, features, macro tuning and scaled calibration.

The group-balanced/C100 development results follow. Lower CV log loss and Brier are better. Validation covers four subjects, three events and 282 available labeled windows; these are tiny, manufactured cohorts.

| Candidate | Fields | Training group-macro CV log loss | Validation calibrated Brier | Validation event sensitivity | False warnings / eligible hour |
| --- | ---: | ---: | ---: | ---: | ---: |
| logistic_clinical | 32 | 0.69441 | 0.14020361 | No policy | No policy |
| logistic_ecg | 6 | 0.05445 | 0.00004404 | 1.00 | 0.000 |
| xgboost_clinical | 32 | 0.54804 | 0.14969665 | No policy | No policy |
| xgboost_ecg | 6 | 0.05214 | 0.00025352 | 1.00 | 0.000 |
| xgboost_combined | 38 | 0.05177 | 0.00025872 | 1.00 | 0.000 |
| xgboost_personal | 45 | 0.05302 | 0.00196760 | 1.00 | 0.000 |
| logistic_ecg_engineered | 23 | 0.03742 | 0.00000062 | 1.00 | 0.000 |
| elasticnet_combined_engineered | 55 | 0.03915 | 0.00000002 | 1.00 | 0.000 |
| elasticnet_personal_engineered | 101 | 0.04033 | 0.00000010 | 1.00 | 0.000 |
| xgboost_personal_engineered | 101 | 0.05449 | 0.00000369 | 1.00 | 0.000 |

Core logistic's group-macro CV loss improved from 0.06567 under window-class weighting to 0.05445 under group balancing. Core ECG XGBoost's validation false-warning rate fell from 0.39735/hour to 0 at the selected policy. These are controlled synthetic development differences, not clinical effect estimates. The richer logistic feature set further reduced grouped CV loss to 0.03742 and validation Brier from 0.00004404 to 0.00000062. Static clinical-only candidates failed the operating constraint, illustrating that susceptibility fields do not automatically add short-term imminence information.

Every passing ECG/fusion challenger reached the same event-sensitivity/false-warning ceiling. The explicit tie rule therefore kept core ECG logistic: logistic before elastic net before XGBoost, then fewer fields. Better probability fit did not establish incremental event benefit. Personalization is not proven superior: fusion and baseline changes alter several inputs at once, and no clinical event cohort is available. No individual feature importance is presented as a causal or independently isolated effect.

## A rejected hypothesis: stronger sigmoid regularization

The calibrator now standardizes scalar scores using calibration patients only and archives slope/intercept, score range, groups, prevalence and fit diagnostics. Stronger C1 regularization was the initial hypothesis, but separate development validation showed poorer Brier than C100:

| Candidate | Validation Brier C100 | Validation Brier C1 |
| --- | ---: | ---: |
| logistic_ecg | 0.00004404 | 0.00618250 |
| logistic_ecg_engineered | 0.00000062 | 0.00021156 |
| elasticnet_personal_engineered | 0.00000010 | 0.00010454 |
| xgboost_personal_engineered | 0.00000369 | 0.00037016 |

For core logistic, C1's Brier 0.00618250 also exceeded its uncalibrated 0.00129403. I rejected C1 for the demonstrated package before evaluating test, retaining C100. The revised runner chooses calibration strength by validation patient-macro Brier among the two group-balanced regimes after model/policy qualification. The [original initial C1 protocol](../reports/ml-improvements/initial-c1-protocol.json), [revised development record](../reports/ml-improvements/development-ablations.json) and written plan preserve the change.

This is an additional development selection, not a pristine preregistered protocol or evidence that C100 is clinically optimal. The generic C1 config remains an experimental starting hypothesis; runtime loads the frozen artifact's C100 config. Reproduce the selected demo with the experiment runner rather than assuming the generic default was validated. Three regimes, ten models, two settings per family and nineteen thresholds create substantial selection uncertainty on four validation groups. [Cawley and Talbot](https://jmlr.org/papers/v11/cawley10a.html) explain why selecting many alternatives can overfit the selection criterion.

## Frozen synthetic test: gains and regressions

Only after the revised choice was recorded did a separate evaluation replay the frozen selected package and all ten comparisons on the same four held-out synthetic subjects. They provide software verification rather than a blinded clinical test. No model or policy was changed after seeing these results.

| Model | Test Brier | Matched events | Assessable false warnings | Total warnings | Unassessable warnings |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous release, core logistic | 0.00043126 | 3 | 0 | 3 | 0 |
| logistic_ecg | 0.00236984 | 3 | 0 | 3 | 0 |
| logistic_ecg_engineered | 0.00000582 | 3 | 0 | 5 | 2 |
| elasticnet_personal_engineered | 0.00001303 | 3 | 0 | 3 | 0 |
| xgboost_personal_engineered | 0.00000020 | 3 | 0 | 3 | 0 |

The selected core package still detected 3/3 events with three warnings and no assessable false warnings, over 4.667 recording hours and 2.517 eligible hours. Its test Brier 0.00236984 **regressed** from the previous release's 0.00043126. The combined v2 changes are therefore not an overall probability-performance win for the selected model. Calibration within v2 improved test Brier relative to raw 0.00577521, but that does not erase the release-to-release regression.

Enriched challengers achieved lower synthetic probability errors. Enriched logistic issued two additional unassessable warnings, so zero assessable false warnings does not mean zero extra burden. Those test results are descriptive and did not override the pre-test simplicity rule. PR-AUC/AUROC remain 1 on manufactured precursors; bootstrap sensitivity bounds collapse to1–1 and cannot bound future misses. A sensitivity ceiling prevents demonstrating further event gains. [Complete evaluation](../reports/ml-v2-evaluation/evaluation.json) includes per-patient outcomes, misses, exclusion time, prevalence, calibration bins and independent-group bootstrap intervals.

Median lead was 521 seconds, with quartiles 512/523.5 seconds; all eligible events had at least 300 seconds of warning lead. Forecast rejection remained 0.8333 hours. Selected prediction with evidence took median 0.50 ms / p95 0.72 ms; offline ECG I/O, feature extraction, labels and selected prediction took median 12.65 ms / p95 14.48 ms on this machine. Comparison inference and bootstrap are excluded. These timings do not prove raw-device or hospital throughput.

## What prevents a credible clinical improvement claim

The [fresh strict SHDB audit](../reports/ml-v2-shdb-audit/audit.json) retains 128 records/122 subjects/98 annotated records, one complete signal, four annotation candidates across four subjects, zero eligible forecast events and zero eligible labeled monitoring hours. Training remains blocked. The `(N` label is still unknown; more plausible ECG measurement cannot manufacture defensible outcome labels. An explicit alternative non-AF interpretation remains a separate clinical assumption, not a silently applied workaround.

A separate [signal-only full-record QA scan](../reports/ml-v2-shdb-audit/quality-coverage.json) accepted all 2851 cadence windows for downloaded 005 under the same thresholds. Its 23.758 cadence-equivalent hours represent overlapping histories on one record. They are neither label-qualified exposure nor clinical quality certification. Source zero-phase preprocessing still limits streaming claims. MIT's stronger beat ground truth addresses measurement only; unrelated clinical histories were never joined.

The highest-value next work is a defensible continuous onset cohort with enough independent event subjects, reviewed beat/quality development data across leads/devices, then untouched clinical external evaluation of the compact feature ladder. P-wave delineation, PAC morphology and prospective unlabeled adaptation require their own measurement validation. A grouped nested-development design, paired group-bootstrap differences, calibration slope/overall-bias uncertainty and robust feature ablations can follow once independent data support them. Changing test-derived thresholds, adding AF-present positives, downloading 8.3 GB without solving annotation ambiguity, or quoting detection-paper scores would not solve this project's forecast-validation gap.

## Verification and reproduction

The independent reviewer found three defects that existing tests missed: negative-only pooled OOF evaluation, alias-dependent balancing, and list-order complexity ties. All were repaired before completed experiments and independently rechecked; [review closure](ml-code-review.md) includes reproductions and verification. Source changes preserve causal trailing signals, annotation-free runtime, grouped transforms, independent calibration, shared warning semantics and legacy-artifact rejection.

Python tests, targeted experiment selection, Ruff, lockfile consistency, a clean installed wheel, React build and live application replay are verified in [verification](../reports/ml-improvements/verification.md). The active localhost app uses the new model/config and a fresh state database. Old artifacts, reports and source snapshot remain available under artifacts/baseline-source-v1.tar.gz and artifacts/baseline-features-v1.py; old models intentionally fail the new feature-version contract.

```bash
# Fresh output paths are required if reproducing an existing completed experiment.
uv run python scripts/run_ml_experiments.py --data-dir data/demo \
  --audit reports/ml-v2-audit --artifacts artifacts/new-ml-experiment \
  --output reports/new-ml-experiment.json --synthetic \
  --false-warning-budget 1 --minimum-sensitivity 0.5
uv run python scripts/benchmark_beats.py --data-dir data/beat-benchmark \
  --legacy-source artifacts/baseline-features-v1.py \
  --output reports/new-beat-benchmark.json
```

Read the runner's recommended_artifact, then evaluate that frozen package in a separate command. MIT files/license must come from the attributed official resource; stored hashes verify local reproducibility rather than published checksums. Dataset-specific calibration and operating numbers here remain provisional software assumptions.
