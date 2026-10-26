# Twenty-minute demonstration recording script

Record a real **20:00-minute** screen-and-voice demonstration and upload it as **unlisted YouTube**, with its link in README. This meets the common duration of the listing's minimum-twenty rule and the mandatory form's 15–20-minute rule; verify actual player duration. See the dated [requirements audit](../reports/ml-investigation/CHALLENGE_REQUIREMENTS.md). A script or ordinary slide deck is not a recorded demonstration.

The main pipeline is real IRIDIA paired age/sex plus ECG, with `logistic_combined` as the declared challenge finalist. Keep `logistic_current_recipe` as the separate unrestricted scientific reference. Development sensitivity is weak and the acceptance gate fails. Do not fabricate clinical success or convert synthetic software checks into real outcomes.

| Recording time | Duration | Show and explain |
| --- | --- | --- |
| 00:00–02:00 | 02:00 | Introduce CardioTwin, clinician review and the India-relevant cardiovascular use case. Name the precise outcome: new AF onset 120–600 seconds ahead, using only the previous 600 seconds, at a 30-second cadence. Label historical replay and research-only status; AF is different from heart attack or stroke. |
| 02:00–05:00 | 03:00 | Show IRIDIA source/attribution, paired patient age/sex, real ECG, independent patient grouping, repeated recordings and causal input rules. Explain that unavailable diagnoses/labs were not invented. LTAF is external transfer evidence; disclose its record-group identity limitation. Explain SHDB's earlier feasibility limitation briefly without presenting it as the final main cohort. |
| 05:00–08:00 | 03:00 | Show challenge selection, the separate scientific reference, actual frozen schema/model card, calibration and warning policy. Demonstrate clinical values and ECG entering the selected predictor. Explain that the challenge fusion constraint was declared after development and before held-out evaluation. Report development 6/241 detected events for combined logistic versus 7/241 for the ECG reference with greater fusion warning burden; this does not prove fusion benefit. Show the failed gate. |
| 08:00–14:00 | 06:00 | Show the artifact-bound real development-prefix research replay receipt and logs. Present start/pause/step, ECG/trends, clinical profile, baseline, quality, warnings and review timeline in the separately labeled manufactured-data dashboard as the conceptual doctor UI. The failed-gate frozen model is not publicly served in that dashboard. The baseline evolves as state; baseline-deviation inputs are absent from the chosen fusion schema. Show held-out warnings, misses or false warnings from completed evaluation artifacts only; offline evaluation is not interactive held-out UI replay. Show real abstention if present; any synthetic quality example must carry a software-example label. Keep failed-gate artifacts research-only and preserve the normal serving refusal. If real replay remains unverified, state that limitation explicitly. |
| 14:00–17:00 | 03:00 | Show completed primary/external evidence from the final investigation report using the fields below. Compare the frozen fused finalist with the scientific reference without retuning. Explain sensitivity, false warnings per eligible hour, total starts/occupancy, missed events, unknown/excluded warnings, coverage and group uncertainty. If a result is still pending, label it pending; do not substitute development or synthetic numbers. |
| 17:00–20:00 | 03:00 | Show reproducible install, code/data licenses, leakage/state tests, measured replay latency and timestamped phase/GPU diagnostics. State observed limitations, lack of demonstrated clinical usefulness and absent Indian-population validation. Explain the future ethically approved Indian partner study and prospective silent evaluation. End with the actual repository/artifact links and research-only conclusion. |

Durations total **20:00**. Rehearse to these boundaries before recording; inspect the exported/uploaded video's duration, not just this schedule.

The results card below is filled from the completed, model-bound evaluation JSONs and must agree with the [final investigation report](../reports/ml-investigation/report.md). **F** means the clinical-plus-ECG challenge finalist; **R** means the separately preserved ECG scientific reference. These are offline frozen-cohort outcomes, not the manufactured-data dashboard's predictions.

| Metric | Primary F | Primary R | External F | External R |
| --- | --- | --- | --- | --- |
| Groups / recordings | 31 / 35 | 31 / 35 | 84 / 84 | 84 / 84 |
| Monitored / eligible hours | 1446.50 / 1211.700 | 1446.50 / 1211.700 | 1960.60 / 483.325 | 1960.60 / 483.325 |
| Matched / eligible events; missed | 1 / 43; 42 missed | 0 / 43; 43 missed | 2 / 29; 27 missed | 0 / 29; 29 missed |
| Sensitivity % [95% CI] | 2.33% [0.00, 7.89] | 0.00% [0.00, 0.00] | 6.90% [0.00, 16.67] | 0.00% [0.00, 0.00] |
| Assessed false starts; FAR / eligible h | 103; 0.085005 | 0; 0.000000 | 4; 0.008276 | 0; 0.000000 |
| All starts; warning hours; occupancy | 113; 24.4567 h; 1.69% | 0; 0.0000 h; 0.00% | 398; 120.7833 h; 6.16% | 0; 0.0000 h; 0.00% |
| Coverage %; abstention-missed events | 92.89%; 3 | 92.89%; 3 | 89.37%; 3 | 89.37%; 3 |
| Known-excluded / unknown starts | 9 / 0 | 0 / 0 | 299 / 93 | 0 / 0 |
| Group-macro / window Brier | 0.00665748 / 0.00442000 | 0.00666074 / 0.00441734 | 0.01205468 / 0.00701545 | 0.01205449 / 0.00700850 |
| ECE / positive-window prevalence % | 0.00422748 / 0.4423% | 0.00320787 / 0.4423% | 0.00097285 / 0.7075% | 0.00055566 / 0.7075% |

Source JSONs: [primary F](../reports/ml-investigation/frozen-cohorts/primary-evaluation/evaluation.json), [primary R](../reports/ml-investigation/reference-evaluations/primary-evaluation/evaluation.json), [external F](../reports/ml-investigation/frozen-cohorts/external-evaluation/evaluation.json), [external R](../reports/ml-investigation/reference-evaluations/external-evaluation/evaluation.json). F model SHA: `55f602615f44a02fdd7f90647b08740bbeaa74ecaaff2dd2f8d85fcb05a79a02`; R: `888e6bc4a1bf6b6ea344398f678fc831d0bc01d41bb7e18d108dc1bcb17d281a`.

Both use their unchanged frozen 0.25-budget policies with 60-second persistence / 600-second cooldown. F threshold/release: 0.013467981195342779 / 0.010100985896507082; R: 1.0 / approximately 0.75. No held-out retuning occurred.

Sensitivity intervals use **300 whole-group bootstrap samples**: primary resamples 31 audited patient groups; external resamples 84 recording groups, whose patient identity/disjointness is unverified. Reference 0–0 intervals do not bound unseen errors. FAR uses labeled eligible hours, including rejected windows; occupancy uses all monitored hours. Coverage is available/monitored time. Brier/ECE are dimensionless probability diagnostics; rare-event prevalence can make them small despite poor event sensitivity.

Narrate the failure plainly: F detects 1/43 primary and 2/29 external events; R detects none and emits no warnings. Additional F warning burden is real, especially the external **398 starts, including 299 known-excluded and 93 unknown-outcome starts**. Only four external starts are assessed false, so its low FAR must not conceal these other starts. These results do not establish clinical usefulness or fusion benefit, and all readiness gates remain false. External clinical inputs are missing and remain missing. Completed predeclared endpoint checks retain every frozen warning: F detects 1/42 and 1/41 primary events at 60/300 s minimum duration, and 0/19 and 0/8 external events. Keep the declared 30 s endpoint as the headline; these checks do not select a favorable definition.

Copy measured research-replay latency and verified future-perturbation/restart outcomes separately. Do not claim a numerical outcome from a planned command or passing synthetic fixture.

At the final delivery stage, add true team/college details, remove tokens/contact screens and local identifiers, check audio/readability and all public links, and verify the unlisted video works without login. Actual recording, upload and public playback verification remain pending.
