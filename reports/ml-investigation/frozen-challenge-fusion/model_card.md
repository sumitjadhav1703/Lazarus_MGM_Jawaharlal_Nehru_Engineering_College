# CardioTwin frozen challenge model card

Some referenced artifacts (model binaries, caches, intermediate receipts) are not shipped in this repository.

Prepared 10 October 2026. **Research only; primary and external frozen evaluations completed.** This card describes the actual frozen `logistic_combined` package, not the synthetic demo or the separate ECG scientific reference. This update reports the completed frozen evaluations; no model, calibration or policy was retuned.

## Identity and permitted use

Version: `investigation-logistic_combined-a5fbcb844aba`. Frozen at `2026-10-09T20:39:14Z` (10 October, 02:09:14 IST). Intended use is reproducible AF-onset forecasting research and a clearly labeled historical digital-twin demonstration with clinician review. It is not validated for diagnosis, treatment, heart-attack/stroke prediction, prospective wearable operation or Indian population care.

The actual manifest has `deployment_ready=false`, `development_acceptance_gate=false`, `final_policy_acceptance_gate=false` and `acceptance_gate=false`. The ordinary predictor loader refuses the package; an explicit research-only harness is required. No warning or low risk score establishes safety.

| Bound file | SHA256 |
| --- | --- |
| model.joblib | `55f602615f44a02fdd7f90647b08740bbeaa74ecaaff2dd2f8d85fcb05a79a02` |
| [manifest.json](manifest.json) | `3f6e3e31d042030def7399c3d8b5649488c93a37d4e19cd24507e3a042d5dfc8` |
| [feature_schema.json](feature_schema.json) | `d79f0adf2ae3bd4f5a010221d65f0f6652573cec3cb0c90f6f656eb47f3e0330` |
| [config.json](config.json) | `1bbfd79bef06304429f313cf24ddfbbe28d430afa0f30a968cdaa41e2d5ec8b9` |
| policy_search.json | `e07f1e8b224441d55a886c8c6f036c82ac30ab407942da54414be0088a598bde` |

The model binary embeds pipeline, calibration, schema, configuration and cohort contract. Sidecar/package equality and binary SHA were checked through the research loader. The selection semantic SHA is `bdd6fd1d357be588f30a209fe50bbae56f487f84a53495be43f8cd6118cfbeb7`; the primary split semantic SHA is `a5fbcb844abaeff2c30b1b16587f6fecbb057c81d2e80678c408158f0c903536`. Freeze receipt records this artifact and the independently preserved scientific reference.

## Inputs, target and fitted method

- IRIDIA-AF v1.0.1 supplies genuinely paired baseline age/sex and two-lead 200 Hz ECG. Only available clinical values are used; diagnoses/labs are not invented or joined from unrelated profiles.
- Target: new AF onset 120–600 seconds ahead, previous 600 seconds of history, 30-second decisions; minimum episode 30 seconds, washout 600 seconds. Daily device-calibration exclusions and quality/warm-up rules remain in the frozen configuration.
- The saved schema has **27 fields**: age, sex, their two availability flags and 23 causal ECG features. Matrix construction adds missingness indicators, giving 54 transformed inputs. Feature contributions explain the score rather than a physiological cause.
- Fitted method: median imputation, standard scaling and logistic regression (`C=100`, seed 42), audited-group/class training weights; fitting cadence 120 seconds. All 121 development groups are partitioned into 85 fitting, 18 calibration and 18 policy groups, disjoint from the 31 primary test groups.
- Calibration is a raw-score StandardScaler plus unweighted sigmoid logistic regression (`C=100`) on separate calibration groups at natural prevalence. The embedded sigmoid coefficient is 0.33845231717576435 and intercept is -4.745140789259675 **after its score scaling**; these are not clinical effect sizes.
- Prior-only patient baseline and persistent state evolve in the twin service, but baseline-deviation features are absent from this selected predictor. State adaptation must not be described as online weight learning or demonstrated personalization benefit.

## Frozen warning policy and development evidence

Primary policy: threshold `0.013467981195342779`, release threshold `0.010100985896507082`, persistence 60 seconds and cooldown 600 seconds. The assessed false-warning budget is 0.25 per eligible hour. The config's research policy-search minimum is zero; this does **not** waive the separate 50% development/final acceptance gates, both of which failed. Additional 0.1/1-hour policies are frozen in the cohort contract; none may be retuned from held-out results.

Three grouped outer development folds cover 132 recordings / 121 groups, 5243.98 monitored hours and 3734.85 eligible hours. The fused model detects **6/241 events (2.49%)**, misses 235, produces 637 assessed false warnings (0.1706/eligible hour) and 761 total starts. Warning occupancy is 40.78%; forecast coverage is 96.00%; independent-group Brier is 0.01220871. These are **development comparisons**, not results of the final refitted package on unseen people.

The preserved ECG reference detects 7/241 development events with lower warning burden. Clinical fusion benefit is not established. Challenge selection applies a new, explicitly post-development/pre-held-out fusion constraint to completed candidates; it does not overwrite the original scientific selection. The completed held-out results below retain the frozen artifacts and policies without retuning.

## Completed frozen primary and external evaluation

The [primary fused evaluation](../frozen-cohorts/primary-evaluation/evaluation.json) and [external fused evaluation](../frozen-cohorts/external-evaluation/evaluation.json) both bind model SHA `55f602615f44a02fdd7f90647b08740bbeaa74ecaaff2dd2f8d85fcb05a79a02` and the frozen primary split above. Metrics use the frozen 0.25-budget policy; its budget name is a development selection constraint, not a guarantee of measured test performance.

| Fused finalist metric | Primary IRIDIA test | External LTAF |
| --- | --- | --- |
| Resampling unit / groups / records | Audited patients / 31 / 35 | Record groups / 84 / 84 |
| Monitored / eligible hours | 1446.50 / 1211.700 | 1960.60 / 483.325 |
| Matched / eligible events; missed events | 1 / 43; 42 missed | 2 / 29; 27 missed |
| Event sensitivity [bootstrap 95% CI] | 2.33% [0.00, 7.89] | 6.90% [0.00, 16.67] |
| Assessed false warnings / eligible hour | 103 / 0.085005 | 4 / 0.008276 |
| All warning starts; warning hours; occupancy | 113; 24.4567 h; 1.69% | 398; 120.7833 h; 6.16% |
| Forecast coverage; events missed during abstention | 92.89%; 3 | 89.37%; 3 |
| Known-excluded / unknown-outcome starts | 9 / 0 | 299 / 93 |
| Group-macro Brier / window Brier | 0.00665748 / 0.00442000 | 0.01205468 / 0.00701545 |
| Expected calibration error / positive window prevalence | 0.00422748 / 0.4423% | 0.00097285 / 0.7075% |

Intervals use 300 whole-group bootstrap samples. Primary units are audited independent patients. LTAF uses record groups: cross-record patient identity and cross-dataset subject disjointness remain unverified. LTAF lacks paired clinical values, so external inference also tests missing-modality transfer. Calibration is descriptive on natural labeled prevalence; a small Brier/ECE does not establish useful forecasting.

The unchanged [primary ECG reference](../reference-evaluations/primary-evaluation/evaluation.json) detects 0/43 events and the [external ECG reference](../reference-evaluations/external-evaluation/evaluation.json) detects 0/29. Both issue zero warnings and have zero observed warning occupancy at their independently frozen threshold 1.0. Its model SHA is `888e6bc4a1bf6b6ea344398f678fc831d0bc01d41bb7e18d108dc1bcb17d281a`. The fused model detects one/two events with additional warning burden, including **299 known-excluded and 93 unknown-outcome external starts**; the low external assessed FAR describes only four assessed false warnings and must not hide the other 392 unassessable starts. Fusion improvement or clinical benefit is not established by these sparse counts. Reference 0–0 bootstrap sensitivity/FAR intervals describe this observed sample and cannot guarantee zero future error.

Neither held-out point sensitivity approaches the 50% engineering gate. **All frozen readiness/gate flags remain false.** No result authorized a threshold/model change or public clinical serving. Completed predeclared endpoint checks retain the frozen predictions and warnings: primary 1/42 and 1/41 at 60/300 s minimum AF duration; external 0/19 and 0/8. These analyses are not used to select a better-looking model or episode definition; see the [final report](../report.md).

## Verified software dependence and replay

Input-dependency receipt binds this model SHA and confirms nonzero coefficients and changed risks after artificial changes to age, sex and ECG heart rate on one accepted **development** row. Second dependency check independently records those changes. These establish actual software use of both streams; they are not physiological counterfactuals, treatment effects or clinical efficacy evidence.

[Research replay receipt](../frozen-challenge-replay.json) binds model, manifest, split and selection identities. Mutating only ECG/annotations after 1,800 seconds leaves earlier forecasts, features, baselines, warnings and timelines identical. There are 60 decisions per variant and 36 available per variant; persisted waveform keys are absent. Across 72 available steps, median latency is 25.17 ms and p95 is 27.38 ms, including ECG/features/prediction/alert/store/state response and excluding startup. This is one development-record prefix, not full-cohort throughput or clinical validation. Default-loader refusal remains intact. Restart/UI verification is not inferred from this causality receipt.

## Data rights and limitations

[IRIDIA-AF, Gilon et al. (2023)](https://zenodo.org/records/8405941) is CC BY 4.0; [LTAF, Petrutiu et al. (2007)](https://physionet.org/content/ltafdb/1.0.0/) is Open Data Commons Attribution 1.0. Code is MIT; retain [full dataset attribution](../../../DATA_LICENSE.md). Raw signals and local patient state remain outside public source delivery.

External LTAF clinical covariates remain missing and are never imputed from invented patient records. LTAF record groups do not prove unique people or cross-dataset subject disjointness. Small independent cohorts, rare events, calibration uncertainty, source/device differences and poor sensitivity limit interpretation. This artifact is a feasibility finding, not a clinical release. Team identity, public repository/video, registration and submission remain deferred.
