# Data and forecasting labels

The implementation pins [SHDB-AF v1.0.1](https://physionet.org/content/shdb-af/1.0.1/), with dataset DOI [10.13026/n6yq-fq90](https://doi.org/10.13026/n6yq-fq90). Downloaded data retain the source `LICENSE.txt` (Open Data Commons Attribution License v1.0); code licensing is separate. Cite the dataset and [associated publication](https://doi.org/10.1038/s41597-025-04777-4) in derivative work.

```sh
# Download approximately 20 MB of headers/annotations for all 128 records first.
uv run python scripts/download_shdb.py --output data/shdb-af --metadata-only
# One full approximately 66 MB ECG, or selected records; omission downloads the full ~8.3 GB cohort.
uv run python scripts/download_shdb.py --output data/shdb-af --records 001
uv run cardiotwin audit --data-dir data/shdb-af --output reports/shdb-audit
```

Downloads use the flat official file index, atomic temporary files and source SHA256 verification. They reuse intact files and replace corrupt ones. Four small metadata requests can run concurrently; large signals download sequentially. The optional `--source-url https://physionet-open.s3.amazonaws.com/shdb-af/1.0.1/` uses the official public S3 mirror when the primary origin is slow. Missing `.dat` files remain explicit audit exclusions, and missing/unannotated rhythm data never become negatives. Metadata-only counts cannot pass the signal-qualified feasibility gate.

`Data_ID` identifies zero-padded record filenames; `Subject_ID` groups all recordings from one subject and is hashed to a local pseudonym. Header sampling rate and duration determine chunk boundaries. Signals are read as physical units and two-channel `(samples, channels)` arrays over `[start, end)`. Published files are already resampled to 200 Hz after zero-phase filtering; a causal implementation downstream cannot remove that upstream information. This limits any claim about raw-device streaming.

Rhythm labels come from nonempty `.atr` auxiliary notes, independently of annotation symbols (actual SHDB-AF symbols are `"`). `(AFIB` is AF; `(AFL`, `(AT`, `(PAT`, `(NOD` are explicitly labelled other rhythms. `(N` is other/unlabelled, **not verified sinus rhythm**. It defaults to `unknown`; `audit.n_rhythm_is_other=true` permits a separately documented AF-exclusion assumption. The audit always reports annotation-only candidate counts under that alternate assumption; it does not silently use them to pass feasibility. Uncovered leading samples and unannotated recordings remain unknown.

The provisional target uses 600 seconds of history, a 120-second gap and inclusive onset interval `[t+120, t+600]`. Current AF, incomplete history/follow-up, unknown labels, onset inside the gap, short/ineligible AF and washout are excluded. Adjacent AF fragments merge before onset counting. An eligible episode must last at least 30 seconds, admit a valid forecast time with continuous labelled non-AF history, and respect a 600-second washout measured from the prior AF end to prediction time. Audit events additionally require at least one quality-accepted **cadence** window before that onset. These are engineering defaults requiring clinical review, not validated timing recommendations.

The clinical allowlist contains numeric demographics, supplied baseline comorbidity flags, explicitly prior history, listed medication availability and strictly earlier echo measurements. `Antiarrhythmic_Drug_BB` is the beta-blocker source. Missing medication entries stay missing, not absent. Same-day echo dates are excluded when exact timing is unavailable. Positive prior-ablation/pacemaker flags require a strictly earlier procedure date. Final `AF_Type`, AF duration, first-diagnosis dates with unverified temporal availability, subsequent procedures and free text are excluded. Every field has a source and an availability/exclusion reason. Supplied comorbidity flags lack independently dated EHR evidence; their baseline interpretation is disclosed in the audit.

Audit writes `records.json`, `cohort.csv`, detailed `cohort.json`, `field_audit.json`, `audit.json`, `config.json` and `splits.json`. It reports labelled hours, AF fragmentation, short episodes, candidate/quality-eligible onsets, event-bearing groups, monitoring hours and exclusion reasons. Content hashes join duplicate subjects before splitting; duplicate and date-overlapping recordings are excluded from supervised counts. Overlap checks require available shifted date/start time. The flat patient split maps to train/calibration/validation/test, stratifies event-bearing groups once from the seed and reuses an existing split manifest. New duplicate links conflicting with frozen partitions fail rather than moving patients. Real training must enforce `included_record_ids` and `feasible`; the full input fingerprint binds data, annotations and clinical metadata to the artifact.

```sh
uv run cardiotwin demo --output data/demo
uv run cardiotwin audit --data-dir data/demo --output reports/demo-audit
```

The deterministic demo contains 24 independent **manufactured** patients, 18 designed episodes, independent ECG noise, a deliberately constructed precursor pattern, numeric metadata and one missing-signal period. It exists to verify training/calibration/replay/abstention plumbing. Its labels, rhythms and metric values establish no clinical or Indian-population performance. Synthetic provenance is explicit in the manifest and audit and must match the training command's `--synthetic` flag.

## Verified local checkpoint (6 October 2026)

All 128 headers and 98 annotations were downloaded and checksum-verified. The conservative target admits only **4 annotation-qualified onsets in 4 subjects**, so even complete signals cannot meet the default minimum of 12 event-bearing subjects. The alternate explicit `(N)`-as-non-AF interpretation yields **103 annotation-qualified onsets in 54 subjects** and requires source clarification before use. Neither count includes a whole-cohort signal-quality acceptance claim. Merged intervals contain 794 AF episodes; 310 are shorter than the provisional 30-second minimum. Published 809 raw AF intervals differ because adjacent AF marks are merged.

One genuine complete ECG, record `005`, was downloaded through the official S3 mirror and checksum-verified. Its 200 Hz, 86,700-second signal produced a 120,000×2 trailing-window chunk with exactly matching separately requested 10-second prefix. At t=2,820 seconds, the default offline label excludes unknown rhythm; an alternate `(N)` interpretation produces a positive label. The shared quality gate rejected that window for lead disagreement (0.9813). No model was applied. `reports/shdb-integration.json` contains the measured engineering check; `reports/shdb-preliminary` is the original metadata-only snapshot, and `reports/shdb-audit` includes the one downloaded ECG. Missing signals keep duplicate-content checks incomplete.
