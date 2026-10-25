# ECGFounder: generated pipeline measurement

**Measured and verified: 25 October 2026**

**Generated pipeline checks passed. Actual AF-onset forecasting remains unmeasured, and the full comparison has not been released.** This run completed five generated-data stages on a Kaggle Tesla T4 allocation. It made **zero real ECG reads, real encoder calls or real-outcome fits**.

| Stage | Measured wall time | Completed work |
| --- | ---: | --- |
| A: scheduler boundaries | 4.58 s | 5,676 generated rows across 132 record fixtures |
| B: small complete workflow | 144.75 s | 1,232 generated rows; 66 synthetic heads, 6 calibrators and 66 independent no-fit replay packages |
| C: diagnostic adapters | 44.32 s | 629,444 generated rows; 4,946 generated drift pairs |
| D: prediction and report codecs | 116.81 s | 629,444 prediction rows written, decoded and compared; 316,050 generated warning objects across 10 policies |
| E: bundle and transfer-payload preparation | 51.88 s | Generated bundle checks and a 200,000,000-byte payload |

The outer runner took **437.26 seconds**, including **70.12 seconds** for dependencies. The worker supervisor took **366.88 seconds**, with peak process-group RSS **832,061,440 bytes**. The Kaggle UI reported **443.7 seconds** under a different provider clock. Component and operation timers contain nested work; they must not be added again to the outer time.

The generated 200 MB payload was subsequently retrieved and verified against its original SHA-256 in **41.88 seconds** of transfer time. This checks one actual transfer; it does not certify full-study bundle publication or future transfer speed. Runtime package versions, single-thread numerical pools, source hashes and nested terminal receipts were verified.

## What this establishes

The generated scheduler, fitting/calibration/replay workflow, diagnostic adapters, prediction/report serialization and payload retrieval completed within their declared bounds. The synthetic heads and calibrators test software behavior; their outputs are not forecasting results. The [earlier bounded real-ECG engineering report](ecgfounder-engineering-20261022.md) remains separate evidence for 396 selected dense rows and 36 parity checks.

## What remains unresolved

**All eleven accepted intended-worker cost upper bounds remain unavailable; full-run eligibility is false.** The proposed cost calculation was not accepted: native-access applicability remains unresolved and even its partial projection exceeds the allowance. This is a conservative planning calculation, not measured full-study runtime. The remaining launch window is also insufficient under the retained queue and review reservations.

The diagnostic fixture produced 4,946 generated drift pairs. The existing 5,244-pair upper bound is conditional on the original complete chronology; this generated run did not verify those conditions. Native I/O and uncontrolled cache applicability, full chronological coverage, original-pair support, report/bundle scaling and final publication proof remain unresolved. The earlier dense helper measurement excludes the full chronological scheduler. These gaps do not establish physical impossibility or a negative forecasting result.

No sensitivity, AUROC improvement or clinical benefit was measured. Existing failed attempts and resource accounting remain preserved. No model was promoted.

[Aggregate machine-readable receipt](ecgfounder-measurement-20261025.json) records counts, timing scopes and limits without private inputs or source paths.
