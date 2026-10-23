# ECGFounder: completed engineering validation

**Run: 22 October 2026 · Receipts verified: 23 October 2026**

The repaired research runner completed all four bounded engineering stages on a Tesla T4. It processed **396/396 selected dense rows** and passed **36/36 comparisons against direct recomputation**. **Outcome fits remain zero; AF-onset forecasting performance is unmeasured.**

## Measured results

| Stage | Supervisor wall time | Result |
| --- | ---: | --- |
| Input staging and identity checks | 486.33 s | Complete |
| Fixed synthetic CPU benchmark | 1,554.30 s | Complete |
| Full-size synthetic table | 389.06 s | Complete; 629,444 rows |
| Bounded real-ECG dense probe | 115.20 s | Complete; 396 rows across 36 selected slots |

The bootstrap took **2,609.72 seconds**; the Kaggle UI reports **2,616.0 seconds** under its provider timing scope. Phase times above include their supervisors; the dense child itself took 107.50 seconds. These timing scopes are different and should not be added or substituted indiscriminately.

The dense probe encoded 3,240 new chunks, with another 2,160 chunks used for direct parity checks. All 36 endpoint comparisons passed; maximum absolute difference in the delta representation was approximately **3.13 × 10⁻⁷**. Actual device placement was `cuda:0`, Tesla T4, FP32. This does not demonstrate use of both GPUs shown in the Kaggle allocation.

## Why earlier runs stopped

Earlier attempts encountered different engineering limits: pip temporary files were charged to the diagnostic allocation; repeated filesystem walks inflated staging; warning-state copying slowed the CPU benchmark; and the previous dense probe exhausted its 600-second cap while repeatedly walking the filesystem.

In the last failed dense run, its final partial snapshot recorded **1,508 storage scans / 560.25 seconds** before a **601.46-second** supervisor stop. The repaired completed dense run recorded **111 scans / 43.18 seconds**, while retaining 3,449 per-call resource checks. This supports the monitoring-overhead diagnosis. The failed and completed runs cover different amounts of finished work, so this is **not an equal-work throughput or model-quality comparison**.

The repair retains storage checks at slot/parity boundaries and the independent parent monitor, plus per-call RSS/CUDA checks. The fixed phase ledger records failures before routing. Only a verified local timeout with successful cleanup and fresh global/source/storage checks can allow independent later stages; unknown errors and shared safety failures stop execution. Generated tests cover those failure branches; the completed cloud run exercised the successful four-stage path.

Prelaunch verification passed **1,036 tests, zero failures** in the private development workspace on 22 October (266.82 seconds), with independent source, capsule and dispatch review. This is separate from the public repository's CI figures in the README.

## Interpretation and remaining work

This validates the bounded runner, native ECG loading, frozen encoder execution, incremental chunk reuse, table handling and numerical consistency for the selected panel. The dense measurement covers the chunk helper/native I/O; it excludes the full chronological scheduler. A checksum pass preceded signal reads, and OS cache state was uncontrolled.

It does **not** measure forecasting accuracy or full-population coverage. No real-outcome head was fitted, no sensitivity/AUROC improvement was established, and the model was not promoted. All eleven accepted full-cost upper bounds remain unavailable. Full chronological coverage, original-pair support and the predeclared comparison gates must pass before an outcome study. The previous calendar window has expired; this result did not authorize another job or extend any limit.

[Aggregate machine-readable receipt](ecgfounder-engineering-20261022.json) includes source identity, original receipt hashes, counts, timing scopes and unchanged limits. It omits subject identifiers, raw ECG, credentials and private source-file paths. Full private research inputs and receipts are not distributed here. This is a research engineering result, not clinical validation.
