# CardioTwin: research priorities for credible AF onset forecasting

Research review: 7 October 2026. This report combines primary literature with the current implementation and recorded data audits. Recommendations are development hypotheses, not demonstrated improvements. The implemented subset and measured results are recorded separately in [ML findings](ml-findings.md); remaining hypotheses below are future work. Final test patients must remain unavailable for feature, model, calibration, threshold, or protocol selection.

## 1. Establish the task and the evidence ceiling

CardioTwin predicts an annotated AF **onset between 2 and 10 minutes ahead**, using the preceding 10 minutes and a 30-second decision cadence. Detecting AF already present in an ECG, predicting eventual AF diagnosis, and forecasting a near-term onset are different outcomes. Published detection scores cannot validate this forecast. Neither a low score nor an unavailable forecast establishes absence of a heart problem.

The [SHDB-AF dataset](https://physionet.org/content/shdb-af/1.0.1/) was developed for AF detection benchmarking. It contains 128 recordings from 122 subjects, with rhythm annotations in 98 recordings. Its source processing includes zero-phase filtering; historical replay of that signal therefore cannot establish fully causal acquisition performance. The dataset does not explicitly annotate every normal-sinus interval: `(N` denotes other rhythm, including unlabelled normal sinus rhythm. A conservative unknown label is defensible; an alternative AF-exclusion interpretation requires a documented clinical assumption, separate audit, and sensitivity analysis. The algorithm-generated beat annotations are not independently reviewed reference beats.

The recorded [real-data audit](../reports/shdb-audit/audit.json) currently has one available signal record and zero eligible events under the conservative contract. Annotation-only candidate events are not evaluable signal events. Consequently, clinical training feasibility has not been established. The [synthetic evaluation](../reports/demo-evaluation/evaluation.json) is a software check: its four held-out subjects and three events test pipeline behavior, not clinical forecasting accuracy. Perfect synthetic results and degenerate bootstrap intervals provide no evidence of generalization.

## 2. What primary forecasting studies actually support

The [PhysioNet 2001 PAF Prediction Challenge](https://physionet.org/content/challenge-2001/1.0.0/) used paired half-hour records near an onset and distant from an onset. The [challenge report](https://www.cinc.org/archives/2001/pdf/113.pdf) motivates premature-beat and atrial-morphology hypotheses. Its selected-record discrimination setting does not reproduce continuous monitoring, CardioTwin's two-minute exclusion gap, or warning burden per hour. Related [P-wave time-course research](https://pubmed.ncbi.nlm.nih.gov/23138002/) supports investigating changes in atrial morphology before PAF, conditional on reliable delineation. It does not establish the validity of every P-wave statistic on these leads.

[WARN, published in Patterns in 2024](https://doi.org/10.1016/j.patter.2024.100970), investigated early warning from short RR sequences. Its cohort, prediction formulation, and reported warning timing differ from this project's fixed future interval. It supports RR dynamics as a candidate information source, not transferring its headline performance or interpreting a retrospective selected-window score as operational warning performance.

A [June 2026 primary manuscript](https://arxiv.org/html/2606.10900v1) studies 60-second ECG inputs and a five-minute pre-onset task, with a one-second gap. It reports candidate HR/RMSSD changes and benefits from supervised patient adaptation. Its training adds AF-containing segments to the positive class; patient-specific operating thresholds are selected on patient test data and explicitly treated as descriptive. CardioTwin should copy neither practice. AF-containing positives can teach detection, and test-selected thresholds cannot estimate prospective performance. Supervised adaptation using earlier labelled episodes also differs from an unlabeled, prior-only physiological baseline. Record-level splitting is insufficient where several records belong to one person.

## 3. Prioritized feature engineering

**Priority 0: repair and validate signal measurement before adding predictors.** The real-record integration check rejected a window for 98.1% lead disagreement. A read-only reproduction found 907 peaks in one lead and 45 in the other. The current detector's threshold includes a fraction of the maximum energy over the entire window; unusually large peaks can suppress ordinary beats. This is a plausible mechanism requiring reviewed snippets, not measured detector sensitivity. Relaxing the agreement threshold would conceal the measurement problem.

Compare adaptive, per-lead detection with the installed [WFDB XQRS and annotation-comparison tools](https://wfdb.readthedocs.io/en/latest/processing.html) offline. Validate missed/double beats, polarity, ectopy, artifacts, and inter-lead delays. Audit preprocessing causality before a comparator becomes the streaming detector. [Clifford et al.'s primary signal-quality study](https://pubmed.ncbi.nlm.nih.gov/22902749/) motivates combining spectral, morphological, and detector-agreement information; its reported accuracy on another setting is not transferable. Select a usable lead transparently and abstain when none qualifies.

**Priority 1: a compact, interpretable RR feature ladder.** Add robust RR median, IQR/MAD, normalized dispersion, and short-versus-long trailing changes in HR and variability. Prespecify minimum beat counts and durations; unavailable measurements remain missing with availability indicators. All computation must end at the decision timestamp. RR plausibility filtering alone does not produce normal-to-normal intervals: label resulting statistics RR variability unless normal-beat classification is independently validated.

**Priority 2: contextualize deviations.** Extend the prior-only baseline with spread, count, and age so a deviation from a stable baseline differs from one supported by few observations. Report absolute and relative changes, with numerical floors and unavailable states. Avoid updating a baseline with rejected windows. Baseline evolution changes patient state; it must not silently retrain model weights or use future annotations.

**Priority 3: atrial and ectopic hypotheses after measurement validation.** Short-long RR patterns and premature-beat fractions may be useful proxies. Calling them PAC counts requires validated atrial/beat morphology. P-wave template correlation, amplitude variability, and duration need delineation and their own quality gate. Avoid a large collection of redundant entropy/spectral features or a transformer before independent event counts support that complexity.

Static clinical fields may capture susceptibility rather than imminence. Audit availability at the decision time, retain missingness, and exclude final Holter diagnoses such as `AF_Type`, later investigations, and outcome-derived text. Require incremental forecasting benefit over ECG alone and report within-patient warning behavior.

## 4. Learning with a tiny grouped cohort

Start with regularized logistic models and a small prespecified comparison ladder: clinical, ECG, combined, then personal deviations. Consider boosting only when grouped development evaluation improves event sensitivity at the declared false-warning budget. Thousands of overlapping windows do not create thousands of independent patients. [Riley et al.'s sample-size methods](https://www.bmj.com/content/368/bmj.m441) explain why a universal “ten events per predictor” rule is inadequate: required size also depends on prevalence, parameter count, and anticipated model fit.

Maintain person-level grouping across records. Fit imputation, scaling, selection, and early stopping only inside training folds. Report independent event subjects and valid folds, and fail explicitly when no fold supports both classes. If four separate development partitions are too sparse, prespecify grouped nested validation or cross-fitting before accessing the final test. Do not redesign the split after observing test performance.

Calibration should use independent predictions at natural monitoring prevalence, not balanced training weights. Report calibration patient/event counts, score variation, separation, and instability. Sparse data can make even a two-parameter sigmoid unreliable. Thresholds, persistence, release rules, and warning budgets belong to development validation; freeze their version before final evaluation. These controls align with [TRIPOD+AI](https://www.bmj.com/content/385/bmj-2023-078378) and [PROBAST+AI](https://www.bmj.com/content/388/bmj-2024-082505).

## 5. Reporting improvements that expose failure modes

Retain event-level misses, chronological abstentions, one-to-one event matching, and patient bootstrap uncertainty. Add:

- Per-patient event sensitivity, eligible hours, false warnings, warning duration, and lead-time distributions alongside pooled values; show counts rather than hiding small denominators.
- Positive eligible-window prevalence and a constant-prevalence Brier reference. Interpret PR-AUC against prevalence. Show calibration-bin counts and contributing patients.
- Calibration-in-the-large, slope, and observed-to-expected ratio, with patient-clustered uncertainty where estimable. Mark unstable estimates unavailable. [Primary external-validation guidance](https://www.bmj.com/content/bmj/384/bmj-2023-074820.full.pdf) supports evaluating calibration beyond a single summary error.
- Lead-time quartiles and prespecified lead-time bands, including all eligible events in detection fractions. Split misses into available-but-unwarned and wholly/partly unavailable opportunities.
- Rejection hours by signal, annotation, and baseline reason; labelled monitoring and all-recording denominators side by side. Unknown rhythm must not dilute false-warning rates.
- Paired patient-bootstrap differences for incremental feature comparisons; separate confidence intervals do not establish superiority. Report valid resamples and degenerate intervals.
- End-to-end latency including loading and feature extraction, separately from prediction latency, plus prespecified lead/device robustness analyses.

The immediate sequence is therefore: complete real signals and defensible labels; validate peaks and quality; add a few causal features; run grouped development ablations; freeze calibration and policy; evaluate the untouched test once. Until a viable clinical cohort exists, stronger claims remain outside the evidence supported by this repository.


## Implementation-specific sources and limits

[Sample entropy's original paper](https://journals.physiology.org/doi/10.1152/ajpheart.2000.278.6.h2039) supports the statistic; it does not clinically validate this implementation's 50-interval minimum or 256-interval computational cap. [Elgendi's detector study](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0073557) motivates testing threshold behavior; CardioTwin's percentile repair is an engineering change, not an implementation of that published algorithm. The [separate MIT-BIH annotated beat resource](https://physionet.org/content/mitdb/1.0.0/) tests localization on two snippets without providing fused AF forecasting evidence.

[Zou and Hastie's elastic-net paper](https://hastie.su.domains/Papers/elasticnet.pdf) motivates combined L1/L2 regularization for correlated inputs. [Scikit-learn calibration guidance](https://scikit-learn.org/stable/modules/calibration.html) motivates independent sigmoid calibration; a small patient count remains a limitation even with many overlapping windows. [Cawley and Talbot](https://jmlr.org/papers/v11/cawley10a.html) describe selection bias and overfitting from model search. Ten models across three prespecified regimes increase this uncertainty; no small validation advantage is treated as clinical superiority.
