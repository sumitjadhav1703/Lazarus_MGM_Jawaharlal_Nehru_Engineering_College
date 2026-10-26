# Independent ML code review

Reviewed 7 October 2026 against `docs/superpowers/plans/2026-10-07-ml-improvements.md` and the learning contract. Scope: features, modeling, training, evaluation, configuration, experiment runner, beat benchmark, and associated tests. This is a read-only source review; no training artifacts or final-test results were used for selection. Line references describe the reviewed snapshot and can shift after repairs.

## Actionable findings

### P2 — Grouped tuning can succeed without any positive out-of-fold assessment

Location: `src/cardiotwin/train.py:173–175`, `213–236`.

Only the fitting side of a fold is checked for both classes. When all event windows belong to one independent group, the fold holding that group is skipped because its training set contains only negatives. Remaining folds evaluate negative-only patients. Their finite predictions are pooled, counted as valid folds, and used to select regularization, even though no positive forecast has been assessed out of fold.

Reproduction, executed successfully in the project environment:

```python
from cardiotwin.train import _fit_model

rows = []
for patient in range(3):
    for i in range(12):
        label = int(patient == 0 and i >= 6)
        rows.append({
            "patient_id": str(patient), "accepted": True, "label": label,
            "features": {"ecg_hr": 60 + 20 * label + i * .01},
        })
_, _, details = _fit_model(rows, "logistic", "ecg", 42)
print([(s["valid_folds"], [f["hold_positive_windows"]
       for f in s["folds"] if f["status"] == "evaluated"])
       for s in details["search"]])
```

Observed: `[(2, [0, 0]), (2, [0, 0])]`; tuning nevertheless succeeds. The two candidate macro losses were approximately 0.1829 and 0.0505.

Repair: reject a candidate when pooled usable OOF labels lack either class. Archive covered/uncovered windows, independent groups, positive windows and event groups. Do not require each holdout fold to contain both classes: negative-only holdout patients remain informative when the pooled independent OOF assessment includes positives. Root accepted this finding and requested a learning-agent repair.

### P2 — Audited patient aliases receive multiple shares of training and tuning weight

Location: `src/cardiotwin/train.py:136–137`, `177`, `191`, `209`, `215`, `244`.

Grouped folds correctly use `group_id`, while balancing and macro tuning use `patient_id`. The audit joins duplicate-linked IDs into one independent group; retained nonduplicate records under those aliases can still enter fitting. Two aliases of one person then receive twice the aggregate mass of a person represented by one ID, despite remaining together during splitting and bootstrap. XGBoost early-stopping weights and macro selection have the same inconsistency.

Minimal weight demonstration: `training_weights([0,1,0,1,0,1], ['a','a','alias','alias','b','b'])` returns six equal weights. If `a` and `alias` share an audited group, that independent person receives 2/3 of total mass and `b` receives 1/3.

Repair: use audited independent `group_id` for the declared balancing unit, macro selection and early-stopping weighting. Preserve original patient-ID counts separately so reporting does not erase aliases. Record both independent-group and patient-ID mass, and test invariance to splitting one person's identical rows between aliases. Root accepted this finding.

### P2 — New candidate ordering can select boosting over a tied logistic baseline

Location: `src/cardiotwin/train.py:497–502`, `529–536`.

The selection comment promises simpler models win ties, but selection resolves ties through iteration order alone. Original XGBoost candidates precede the new engineered logistic model. If core logistic models have lower sensitivity, and engineered logistic ties an earlier boosted candidate on event sensitivity and false-warning rate, boosting wins. This contradicts the learning contract's simpler-model tie rule and makes an unsupported complexity choice at the synthetic ceiling.

Repair: add an explicit, prespecified complexity/family tie priority, or order all applicable logistic baselines before boosted challengers. Test a constructed tied candidate table rather than depending on measured synthetic results. This finding was sent to root for disposition.

## Verified safeguards and remaining limits

- Feature extraction receives only the requested trailing signal; filtering uses `sosfilt`, missing samples are forward filled, and baselines update after the current row is constructed. These paths contain no future waveform or rhythm-annotation access.
- Invalid RR intervals do not bridge RMSSD, Poincare pairs or entropy templates. The sample prefix mask also excludes plausible-duration intervals containing missing samples. Entropy is bounded in cost and remains unavailable for a short valid tail.
- Feature allowlists exclude identifiers, timestamps and outcomes. Explicit missingness indicators cover missing fields first encountered after training. The current feature-version guard rejects legacy configurations before deserialization; embedded schema/config and model checksum checks remain present.
- Training preprocessing stays inside fitting folds. Calibration uses separate patients, its own scalar score scaling, natural prevalence and no class/patient weights. Convergence failures block export and candidate failure isolation retains a successful baseline.
- Validation chooses policy, model and threshold; final-test records are skipped before signal/label extraction. The experiment runner uses only development training and validation. Its failed-candidate summary field was corrected from `failure_reason` to `reason` while review was underway.
- Bootstrap groups linked patient IDs together. Probability references are explicitly descriptive cohort prevalence, and macro Brier supplements the pooled result. Warning burden retains abstentions and annotation exclusions; sensitivity includes eligible events missed during unavailable opportunities.
- The beat benchmark declares reference symbols, timing tolerance, snippets and hashes. Its two-record localization results cannot validate AF forecasting or general detector accuracy. Neither those results nor a synthetic ceiling establish clinical usefulness.

Verification command:

```sh
.venv/bin/pytest -q tests/test_features.py tests/test_learning.py tests/test_evaluation_diagnostics.py
```

Observed before repairs: **42 passed in 1.40 seconds**. Passing tests did not cover the three scenarios above. Re-run these checks and the full suite after repairs, with targeted regression cases, before freezing a release.

## Repair verification and closure

Independent verification on 7 October 2026 closes all three findings. No additional actionable defect was identified in the changed repair paths. This verifies implementation behavior on constructed examples, not clinical forecasting performance. Source files were not edited during review, and final-test data and experiment results were not inspected.

1. **OOF assessment coverage — closed.** `train.py:133–152` records assessed classes, windows and independent positive groups, including unassessed positive groups. Eligibility at `268–277` now requires both pooled OOF classes. Re-running the original three-group reproduction raises the intended error; both candidates archive only class `[0]` as assessed. The new `test_grouped_cv_rejects_holdouts_without_any_positive_assessment` checks the exact failure, two otherwise usable folds, 2/3 window coverage, and omitted positive group. Negative-only individual holdouts remain allowed when aggregate OOF assessment covers both outcomes.

2. **Audited aliases — closed.** Fold fitting weights, XGBoost early-stopping weights, fold/pooled macro loss and final fitting now use audited groups (`train.py:208`, `222`, `240`, `246`, `280`). Reporting retains original patient counts and patient-ID mass separately. The new `test_fitting_weights_and_macro_loss_use_independent_groups_not_patient_aliases` tests equal group mass and invariant fitting/selection after alias relabeling. An additional independent reproduction compared both logistic and XGBoost models against canonical group IDs: maximum probability difference and selected macro-loss difference were exactly zero; all six group masses were 8.0. The XGBoost comparison also exercises its early-stopping path.

3. **Model simplicity on ties — closed.** `_candidate_rank` at `train.py:480–489`, used at `578`, explicitly prioritizes sensitivity, lower burden, family simplicity, then fewer features. The new `test_validation_ties_choose_model_simplicity_before_candidate_order` verifies engineered logistic beats earlier boosting/elastic-net candidates and core logistic wins a same-family tie. An independent constructed comparison also verified that higher event sensitivity still outranks simplicity.

Focused repair-verification command:

```sh
.venv/bin/pytest -q tests/test_learning.py -k 'pooled or independent_groups_not_patient_aliases or validation_ties or grouped_cv or patient_class_mass or xgboost_stops'
```

Observed: **6 passed, 25 deselected in 1.14 seconds**. The original failure reproduction and additional logistic/XGBoost alias and ranking checks also passed. No full suite was rerun by this reviewer; root is responsible for the final full-suite verification.
