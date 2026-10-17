# Source verification and implementation implications

**Current investigation:** The IRIDIA/LTAF investigation and frozen clinical-plus-ECG finalist supersede this earlier SHDB/synthetic iteration. See the [current report](../reports/ml-investigation/report.md), [frozen model card](../reports/ml-investigation/frozen-challenge-fusion/model_card.md) and [challenge requirements](../reports/ml-investigation/CHALLENGE_REQUIREMENTS.md). The dated findings below are preserved as historical evidence.

Verified on 6 October 2026. The development plan is a specification, not evidence of completed training. Its forecasts, sensitivity targets, and clinical acceptance decisions remain provisional until measured. This implementation must separate software completion from empirical feasibility and external delivery.

## Requirements and dependency gates from the development plan

| Task | Depends on | Required evidence |
| --- | --- | --- |
| Ingestion and cohort audit | Authorized SHDB-AF files | Subject/record manifest, duration/sample-rate checks, duplicate/overlap audit, rhythm exclusions, clinical availability audit |
| Forecast contract | Audit | Versioned history, gap, horizon, cadence, episode-duration and washout rules; eligible events, event-bearing subjects and monitoring hours |
| Causal features | Contract and raw ECG | Quality and R-peak checks, HR/RR/valid HRV, trends, prior-only patient baselines; prefix invariance |
| Grouped model development | Feasible cohort and features | Untouched subject test manifest; training-only transforms; clinical/ECG/combined/personalized comparisons; modest regularization/early-stopping search |
| Calibration and operating policy | Held-out development patients | Probability calibration, persistence/hysteresis/cooldown, false-warning budget and sensitivity target selected before final test |
| Continuous evaluation | Frozen model and policy | One-to-one warning/event matching, burden/abstention denominators, patient bootstrap intervals, calibration and latency |
| Twin API and clinician dashboard | Stable feature/model/state contracts | Bounded persistent state, deterministic restart/replay, historical-simulation labeling, evidence/review timeline and unavailable reasons |
| Verification and release | Integrated pipeline | Separation, causality, labels, metadata timing, alert/restart and end-to-end checks; versions, schema, split manifest and model card |
| Challenge delivery | Verified implementation and real measured results | README/team details, architecture and presentation PDF/PPT, public repository links and genuine minimum-20-minute demo |

TCN, hardware, advanced simulation and the PAF supplementary benchmark are optional. Indian-population validation and prospective silent evaluation are later work, not claims established by this cohort. Registration, public publication, recording a genuine demo and clinician agreement require real people/accounts; files may prepare those actions but cannot manufacture them.

## SHDB-AF integration facts

The [versioned resource](https://physionet.org/content/shdb-af/1.0.1/) contains 128 recordings from 122 subjects, with 98 rhythm-annotated recordings from 93 subjects; 30 are unannotated. Files are flat under the [versioned file directory](https://physionet.org/files/shdb-af/1.0.1/). Rhythm `.atr` `aux_note` values start intervals: `(AFIB`, `(AFL`, `(AT`, `(PAT`, `(NOD`, `(N`. `(N` means other/unlabelled; it does not certify sinus rhythm. The published 809 AF intervals are not independent eligible forecasting events. Data were zero-phase filtered and resampled from 125 to 200 Hz before release; therefore implementation causality cannot erase acquisition preprocessing leakage. Full download size is approximately 8.3 GB uncompressed, 7.3 GB ZIP. Removed duplicates include `016`/`030`. Cite the dataset DOI [10.13026/n6yq-fq90](https://doi.org/10.13026/n6yq-fq90) and publication DOI [10.1038/s41597-025-04777-4](https://doi.org/10.1038/s41597-025-04777-4).

[Header 001](https://physionet.org/files/shdb-af/1.0.1/001.hea) specifies two channels, 200 Hz, 17,220,000 samples: 86,100 seconds. Read each header instead of assuming exact 24-hour coverage. [ANNOTATORS](https://physionet.org/files/shdb-af/1.0.1/ANNOTATORS) explicitly describes `.qrs` as unaudited beat annotations. Use independently causal ECG beat extraction at runtime; annotation files belong only to the audit/evaluator.

## Direct metadata inspection

These counts were computed from the complete official [AdditionalData.csv](https://physionet.org/files/shdb-af/1.0.1/AdditionalData.csv), downloaded on the verification date. Join zero-padded `Data_ID` to record filenames and group by `Subject_ID`, never filename. Actual beta-blocker column: `Antiarrhythmic_Drug_BB`. Sex values are `M`/`F`; booleans are strings such as `True`/`False` and some numeric 0.0/1.0 fields. Missing values must remain missing.

| Audit observation | Computed result | Consequence |
| --- | --- | --- |
| Rows / unique subjects | 128 / 122 | The grouping unit is the subject |
| Annotated rows / unique annotated subjects | 98 / 93 | Unannotated records cannot be supervised negatives |
| Repeated subjects | `005/020`, `015/047`, `035/036`, `052/128`, `066/118`, `129/133` | Keep each pair in one split and inspect temporal overlap |
| First AF/AFL diagnosis date | All 69 populated dates occur after `Date_Holter` | Exclude date and dependent `AF_Duration_Months` until provenance/time consistency is resolved |
| Echo date | 125 populated; 31 after Holter, 15 same-day, 79 strictly before | Exclude after and same-day values absent proof of prior availability |
| First ablation date | 61 populated; 20 after Holter | Procedure details need prior-date proof |
| Redo ablation date | 32 populated; 20 after Holter | Do not allow post-recording procedures into features |
| Pacemaker date | 7 populated; 6 after Holter | Future indication/date fields are unsafe |
| Missing core data | BMI: 4, CHF: 1, age/sex: 0 | Training-only imputation plus missingness indicators |

`AF_Type` is a final Holter diagnosis and must be excluded. Comments and undated free text require exclusion. Static prior-diagnosis and prior-medication fields need their own documented availability rule, not blanket acceptance of every column. The [bundled README](https://physionet.org/files/shdb-af/1.0.1/README.md) has stale subject counts and old `UID`/`Study ID` names; the actual current CSV/schema and versioned landing page take precedence.

## Minimum trustworthy forecasting audit

This is an implementation interpretation of the plan's gates, not a claim that the dataset already passes:

1. Verify file checksums from [SHA256SUMS.txt](https://physionet.org/files/shdb-af/1.0.1/SHA256SUMS.txt), record list, sample rates, signals, clinical joins and annotation-presence flags. Record missing/corrupt files without silently dropping them.
2. Normalize rhythm notes, collapse consecutive duplicate rhythm labels, preserve unknown intervals and do not fill an unlabeled initial span as negative. Audit annotation coverage at recording edges and fragmentation.
3. Freeze AF duration/washout before counting events; require non-AF history and adequate future coverage. Distinguish raw AF interval count, duration-qualified transitions and onsets reachable by an eligible prediction window.
4. Count event-bearing **subjects**, not overlapping positive windows. Verify subject and byte-identical/temporally overlapping record separation before choosing train/development/test sizes.
5. Audit raw ECG quality across the required historical span; quantify eligible/rejected monitoring hours and events hidden by rejection. Metadata-only feasibility is preliminary until ECG quality is measured.
6. Expose uncertain rhythms, same-day metadata without times, zero-phase preprocessing, clinical sampling limits and clinician-unreviewed provisional operating targets in reports. Halt clinical-performance claims if the cohort or operating-point gate fails.

## Reuse terms

Dataset license: [Open Data Commons Attribution 1.0](https://physionet.org/files/shdb-af/1.0.1/LICENSE.txt). Keep dataset attribution and license notices in derived outputs; link them from the demo/model card. Repository source-code licensing is separate. Do not label data under the repository code license. Prefer official download instructions to committing multi-gigabyte signals.

## Challenge requirements verified against the organizer listing

The [official Unstop challenge listing](https://unstop.com/hackathons/crp-digital-twin-challenge-2026-happiest-health-1757873/amp) specifies 1–4 full-time students, historical/EHR and dynamic-sensor fusion, a predicted specific adverse event and a doctor-facing dashboard. Anonymous, open or synthetic data are permitted. Registration: **18 October 2026, 19:00 IST**; prototype submission: **20 October 2026, 19:00 IST**. Required public GitHub README contents: team/college, project/problem/use case, stack/model, license, minimum-20-minute explanatory demo video, architecture PDF/PPT and presentation PDF/PPT. Folder convention: `Team Name_College Name`. Submit team leader name/contact and public repository link through the platform. Recheck before submission; listing sections conflict on initial shortlist size (five versus ten).

## Supplementary benchmark boundary

The [PAF Prediction Challenge database](https://physionet.org/content/afpdb/1.0.0/) uses paired 30-minute pre-event/distant segments and 5-minute continuations; 50 learning sets represent 48 subjects. Its task construction differs from continuous SHDB-AF monitoring, and some distant records may contain short unnoticed AF bursts. It can provide a separate ECG benchmark, but cannot supply matched SHDB clinical histories or establish external validation of the same fused forecast.


## XGBoost implementation reference

Grouped training-fold early stopping uses the installed XGBoost 3.4.1 sklearn interface, verified against the [official early-stopping documentation](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html). The holdout evaluation matrix is transformed using that fold's training-only imputer; its labels select tree count, never preprocessing. Final fits use the median best tree count for the chosen parameter setting, with calibration and validation patients kept separate.
