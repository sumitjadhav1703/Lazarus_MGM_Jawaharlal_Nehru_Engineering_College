"""Development-only training. Final-test signals are opened only by evaluation."""

import hashlib
import json
import time
import warnings
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from .evaluation import evaluate_stream, probability_metrics
from .features import FEATURE_VERSION
from .modeling import Predictor, build_pipeline, dependency_versions, feature_schema, matrix, raw_score


def validate_splits(splits, patients, group_ids=None):
    expected = {"train", "calibration", "validation", "test"}
    if set(patients) != set(splits):
        raise ValueError("Unknown or missing patients in split manifest")
    if set(splits.values()) != expected:
        raise ValueError("Every train/calibration/validation/test partition must be present")
    if group_ids is not None and set(group_ids) != set(splits):
        raise ValueError("Unknown or missing audited patient groups")
    assigned = {}
    for patient in patients:
        group = group_ids.get(patient, patient) if group_ids else patient
        if group in assigned and assigned[group] != splits[patient]:
            raise ValueError("An audited group crosses development partitions")
        assigned[group] = splits[patient]


def cohort_contract(splits, group_ids, included_record_ids):
    """Content identity for frozen cohort decisions, independent of JSON ordering."""
    if len(set(included_record_ids)) != len(included_record_ids):
        raise ValueError("Duplicate records in frozen cohort contract")
    payload = {
        "splits": dict(splits),
        "patient_group_ids": dict(group_ids),
        "included_record_ids": sorted(included_record_ids),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {**payload, "sha256": digest}


def validate_cohort_contract(manifest, splits, group_ids, included_record_ids):
    """Legacy releases remain readable; new contracts cannot silently change cohorts."""
    stored = manifest.get("cohort_contract")
    if stored is None:
        return False
    expected = cohort_contract(splits, group_ids, included_record_ids)
    if stored != expected or manifest.get("cohort_contract_sha256") != expected["sha256"]:
        raise ValueError("Frozen cohort contract differs from split/group/inclusion inputs")
    return True


def iter_record_rows(record, config, *, reader=None):
    from .data import read_intervals, read_signal
    from .features import feature_row, update_baseline
    from .labels import label_at

    reader = reader or read_signal
    intervals = read_intervals(record, n_is_other=config["audit"].get("n_rhythm_is_other", False))
    history, cadence = config["target"]["history_seconds"], config["target"]["cadence_seconds"]
    baseline = {"count": 0, "means": {}}
    timestamps = np.arange(0, record.duration_seconds + 1e-9, cadence)
    if timestamps[-1] < record.duration_seconds:
        timestamps = np.append(timestamps, record.duration_seconds)
    for t in timestamps:
        t = float(t)
        label, label_reason = label_at(t, intervals, record.duration_seconds, config)
        features, accepted, reason = {}, False, "insufficient_history"
        if t >= history:
            signal = reader(record, t - history, t)
            features, quality = feature_row(signal, record.fs, record.clinical, baseline, config)
            accepted = quality["accepted"] and baseline["count"] >= config["baseline"]["minimum_updates"]
            reason = None if accepted else (quality.get("reason") or "baseline_warmup")
            if quality["accepted"]:
                baseline = update_baseline(baseline, features, config)
        yield {
            "patient_id": record.patient_id,
            "group_id": record.patient_id,
            "record_id": record.record_id,
            "timestamp": t,
            "duration_seconds": record.duration_seconds,
            "features": features,
            "accepted": bool(accepted),
            "reason": reason,
            "label": label,
            "label_reason": label_reason,
        }


def training_weights(labels, patients, mode="patient_class"):
    """Mean-one weights over supplied independent IDs (audited groups during fitting)."""
    labels, patients = np.asarray(labels), np.asarray(patients)
    if not len(labels) or labels.shape != patients.shape or not set(labels) <= {0, 1}:
        raise ValueError("Training weights require aligned nonempty binary labels and patient IDs")
    counts = Counter(patients)
    if mode == "window_class":
        weights = compute_sample_weight("balanced", labels)
    elif mode == "patient_only":
        weights = np.asarray([1 / counts[patient] for patient in patients])
    elif mode == "patient_class":
        cells = Counter(zip(patients, labels, strict=True))
        observed = Counter(patient for patient, _ in cells)
        weights = np.asarray(
            [
                1 / (observed[patient] * cells[(patient, label)])
                for patient, label in zip(patients, labels, strict=True)
            ]
        )
    else:
        raise ValueError(f"Unknown sample weighting: {mode}")
    return weights / weights.mean()


def patient_macro_log_loss(labels, risks, patients):
    """Equal-unit loss; retained helper name accepts audited group IDs during fitting."""
    labels, risks, patients = np.asarray(labels), np.asarray(risks), np.asarray(patients)
    return float(
        np.mean(
            [
                log_loss(labels[patients == patient], risks[patients == patient], labels=[0, 1])
                for patient in np.unique(patients)
            ]
        )
    )


def _weight_summary(labels, groups, patients, weights, mode):
    group_mass, patient_mass, class_mass = defaultdict(float), defaultdict(float), defaultdict(float)
    for group, patient, label, weight in zip(groups, patients, labels, weights, strict=True):
        group_mass[str(group)] += float(weight)
        patient_mass[str(patient)] += float(weight)
        class_mass[str(label)] += float(weight)
    return {
        "mode": mode,
        "normalization": "mean_one",
        "independent_unit": "audited_group_id",
        "group_mass": dict(group_mass),
        "patient_mass": dict(patient_mass),
        "class_mass": dict(class_mass),
        "weighted_positive_prevalence": float(weights[np.asarray(labels) == 1].sum() / weights.sum()),
        "interpretation": "patient_class balances total independent-group mass and observed within-group "
        "classes; global class masses need not be 50/50. Original patient-ID masses are descriptive.",
    }


def _oof_coverage(labels, groups, predicted):
    positive_groups = set(groups[labels == 1])
    assessed_positive_groups = set(groups[predicted & (labels == 1)])
    return {
        "windows": int(predicted.sum()),
        "total_windows": len(labels),
        "window_fraction": float(predicted.mean()),
        "groups": len(set(groups[predicted])),
        "total_groups": len(set(groups)),
        "group_fraction": len(set(groups[predicted])) / len(set(groups)),
        "positive_windows": int((labels[predicted] == 1).sum()),
        "negative_windows": int((labels[predicted] == 0).sum()),
        "observed_classes": sorted(set(labels[predicted].tolist())),
        "has_both_classes": len(set(labels[predicted])) == 2,
        "positive_groups": len(assessed_positive_groups),
        "total_positive_groups": len(positive_groups),
        "positive_group_ids": sorted(assessed_positive_groups),
        "total_positive_group_ids": sorted(positive_groups),
        "unassessed_positive_group_ids": sorted(positive_groups - assessed_positive_groups),
        "positive_group_definition": "Groups containing eligible positive forecast labels, not unique onset counts.",
    }


def _fit_model(train_rows, family, variant, seed, sample_weighting="patient_class"):
    usable = [row for row in train_rows if row["accepted"] and row["label"] in (0, 1)]
    features = [row["features"] for row in usable]
    schema = feature_schema(features, variant)
    if not schema:
        raise ValueError(f"No audited {variant} features available")
    x, y = matrix(features, schema), np.asarray([row["label"] for row in usable])
    patients = np.asarray([row["patient_id"] for row in usable])
    groups = np.asarray([row.get("group_id", row["patient_id"]) for row in usable])
    if len(np.unique(y)) != 2:
        raise ValueError("Training needs both positive and negative eligible windows")
    candidates = (
        [{"C": 0.1}, {"C": 1}]
        if family == "logistic"
        else [{"C": 0.1, "l1_ratio": 0.5}, {"C": 1, "l1_ratio": 0.5}]
        if family == "elasticnet"
        else [{"max_depth": 2, "reg_lambda": 3}, {"max_depth": 3, "reg_lambda": 10}]
    )
    folds = min(3, len(np.unique(groups)))
    splits = (
        list(StratifiedGroupKFold(folds, shuffle=True, random_state=seed).split(x, y, groups))
        if folds >= 2
        else []
    )
    scores = []
    for params in candidates:
        fold_details, tree_counts = [], []
        out_of_fold = np.full(len(y), np.nan)
        converged = True
        for fold_number, (fit, hold) in enumerate(splits):
            fit_groups, hold_groups = set(groups[fit]), set(groups[hold])
            if fit_groups & hold_groups:
                raise ValueError("A grouped tuning fold contains shared fit/hold groups")
            fold = {
                "fold": fold_number,
                "fit_groups": sorted(fit_groups),
                "hold_groups": sorted(hold_groups),
                "fit_windows": len(fit),
                "hold_windows": len(hold),
                "fit_patients": len(np.unique(patients[fit])),
                "hold_patients": len(np.unique(patients[hold])),
                "fit_positive_windows": int(y[fit].sum()),
                "hold_positive_windows": int(y[hold].sum()),
                "fit_negative_windows": int((y[fit] == 0).sum()),
                "hold_negative_windows": int((y[hold] == 0).sum()),
                "fit_positive_groups": sorted(set(groups[fit][y[fit] == 1])),
                "hold_positive_groups": sorted(set(groups[hold][y[hold] == 1])),
            }
            if len(np.unique(y[fit])) != 2:
                fold_details.append(dict(fold, status="skipped", reason="single_class_fit", converged=None))
                continue
            pipeline = build_pipeline(family, params, seed)
            weights = training_weights(y[fit], groups[fit], sample_weighting)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                if family == "xgboost":
                    # Every preprocessing statistic comes from fold-training patients only.
                    preprocess = pipeline[:-1]
                    fit_x = preprocess.fit_transform(x[fit], y[fit])
                    hold_x = preprocess.transform(x[hold])
                    pipeline["model"].set_params(early_stopping_rounds=10)
                    pipeline["model"].fit(
                        fit_x,
                        y[fit],
                        sample_weight=weights,
                        eval_set=[(hold_x, y[hold])],
                        sample_weight_eval_set=[training_weights(y[hold], groups[hold], "patient_only")],
                        verbose=False,
                    )
                    tree_counts.append(pipeline["model"].best_iteration + 1)
                else:
                    pipeline.fit(x[fit], y[fit], model__sample_weight=weights, scale__sample_weight=weights)
            fold_converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
            converged &= fold_converged
            risks = pipeline.predict_proba(x[hold])[:, 1]
            out_of_fold[hold] = risks
            fold_details.append(
                {
                    **fold,
                    "status": "evaluated",
                    "converged": fold_converged,
                    "convergence_warnings": [
                        str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)
                    ],
                    "group_macro_log_loss": patient_macro_log_loss(y[hold], risks, groups[hold]),
                    "window_log_loss": float(log_loss(y[hold], risks, labels=[0, 1])),
                }
            )
        predicted = np.isfinite(out_of_fold)
        macro_loss = (
            patient_macro_log_loss(y[predicted], out_of_fold[predicted], groups[predicted])
            if predicted.any()
            else None
        )
        scores.append(
            {
                "parameters": {
                    **params,
                    **({"n_estimators": int(np.median(tree_counts))} if tree_counts else {}),
                },
                "grouped_cv_log_loss": macro_loss,
                "group_macro_log_loss": macro_loss,
                "window_log_loss": float(log_loss(y[predicted], out_of_fold[predicted], labels=[0, 1]))
                if predicted.any()
                else None,
                "valid_folds": sum(fold["status"] == "evaluated" for fold in fold_details),
                "converged": bool(converged),
                "oof_coverage": _oof_coverage(y, groups, predicted),
                "folds": fold_details,
                "fold_tree_counts": tree_counts,
            }
        )
    eligible = [
        score
        for score in scores
        if score["valid_folds"] and score["converged"] and score["oof_coverage"]["has_both_classes"]
    ]
    if not eligible:
        raise ValueError(
            "No valid converged grouped tuning folds with both classes assessed; more independent groups required",
            {"search": scores},
        )
    best = min(eligible, key=lambda score: score["group_macro_log_loss"])
    pipeline = build_pipeline(family, best["parameters"], seed)
    weights = training_weights(y, groups, sample_weighting)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        fit_kwargs = {"model__sample_weight": weights}
        if family != "xgboost":
            fit_kwargs["scale__sample_weight"] = weights
        pipeline.fit(x, y, **fit_kwargs)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise ValueError(
            "Final training model did not converge; model export is blocked",
            {
                "search": scores,
                "chosen": best,
                "converged": False,
                "convergence_warnings": [
                    str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)
                ],
            },
        )
    return (
        pipeline,
        schema,
        {
            "search": scores,
            "chosen": best,
            "training_windows": len(y),
            "training_patients": len(np.unique(patients)),
            "training_groups": len(np.unique(groups)),
            "training_positive_windows": int(y.sum()),
            "weighting": _weight_summary(y, groups, patients, weights, sample_weighting),
            "cv_selection_metric": "group_macro_log_loss",
            "converged": True,
        },
    )


def _calibrate(pipeline, schema, rows, seed, calibration_c=1.0):
    usable = [row for row in rows if row["accepted"] and row["label"] in (0, 1)]
    y = np.asarray([row["label"] for row in usable])
    if len(np.unique(y)) != 2:
        raise ValueError("Independent calibration patients need positive and negative eligible windows")
    scores = raw_score(pipeline, matrix([row["features"] for row in usable], schema))
    calibrator = Pipeline(
        [
            ("scale", StandardScaler()),
            ("sigmoid", LogisticRegression(C=calibration_c, max_iter=1000, random_state=seed)),
        ]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        calibrator.fit(scores.reshape(-1, 1), y)  # Natural prevalence, no patient/class weights.
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise ValueError("Independent sigmoid calibration did not converge; export is blocked")
    # Scores are risk-oriented: a non-positive slope reverses the ranking.
    if not calibrator.named_steps["sigmoid"].coef_[0][0] > 0:
        raise ValueError("Independent sigmoid calibration slope is not positive; export is blocked")
    return calibrator


def calibration_diagnostics(pipeline, schema, calibrator, rows):
    usable = [row for row in rows if row["accepted"] and row["label"] in (0, 1)]
    x = matrix([row["features"] for row in usable], schema)
    labels = np.asarray([row["label"] for row in usable])
    scores = raw_score(pipeline, x)
    raw_risk, calibrated_risk = (
        pipeline.predict_proba(x)[:, 1],
        calibrator.predict_proba(scores.reshape(-1, 1))[:, 1],
    )
    slope = float(calibrator["sigmoid"].coef_[0, 0] / calibrator["scale"].scale_[0])
    intercept = float(calibrator["sigmoid"].intercept_[0] - slope * calibrator["scale"].mean_[0])

    def metrics(risks):
        return probability_metrics(
            [dict(row, risk=float(risk)) for row, risk in zip(usable, risks, strict=True)]
        )

    return {
        "scope": "Calibration-fit diagnostic; evaluate calibrated performance on separate validation patients.",
        "windows": len(usable),
        "patients": len({row["patient_id"] for row in usable}),
        "groups": len({row.get("group_id", row["patient_id"]) for row in usable}),
        "positive_windows": int(labels.sum()),
        "positive_prevalence": float(labels.mean()),
        "sample_weighting": "none_natural_monitoring_prevalence",
        "converged": True,
        "calibration_c": calibrator["sigmoid"].C,
        "raw_score_slope": slope,
        "raw_score_intercept": intercept,
        "standardized_score_slope": float(calibrator["sigmoid"].coef_[0, 0]),
        "raw_score_min": float(scores.min()),
        "raw_score_max": float(scores.max()),
        "raw_probability": metrics(raw_risk),
        "calibrated_probability": metrics(calibrated_risk),
        "limitations": ["Overlapping calibration windows are not independent patients."]
        + (
            ["Nonpositive calibration slope: inspect weak or inverted development scores."]
            if slope <= 0
            else []
        ),
    }


def _select_policy(rows, events, config):
    from .evaluation import policy_constraints

    finite = config["alert"].get("policy_version") == "finite-validity-v1"
    if finite and ({r["label"] for r in rows if r.get("accepted") and r.get("risk") is not None and r.get("label") in (0, 1)} != {0, 1} or not events):
        raise ValueError("Finite policy role requires both classes and eligible events")
    budget = config["alert"].get("max_false_warnings_per_hour")
    sensitivity = config["alert"].get("min_event_sensitivity")
    if budget is None or sensitivity is None:
        raise ValueError(
            "Pass explicit provisional false-warning budget and minimum sensitivity before training"
        )
    ratio = (
        config["alert"]["release_threshold"] / config["alert"]["threshold"]
        if config["alert"]["threshold"]
        else 0
    )
    evaluated = []
    for threshold in np.linspace(0.05, 0.95, 19):
        selected = deepcopy(config)
        selected["alert"].update(threshold=float(threshold), release_threshold=float(threshold * ratio))
        report = evaluate_stream(rows, events, selected, bootstrap_samples=0)
        constraints = policy_constraints(report, selected, budget=budget, minimum_sensitivity=sensitivity)
        passed = constraints["passed"]
        evaluated.append(
            {
                "threshold": float(threshold),
                "release_threshold": float(threshold * ratio),
                "meets_constraint": bool(passed),
                "event_sensitivity": report["event_sensitivity"],
                "false_warnings_per_hour": report["false_warnings_per_hour"],
                "warning_precision": report["warning_precision"],
                "warnings": report["warnings"],
                "unassessable_warnings": report.get("unassessable_warnings", 0),
                "known_excluded_warnings": report["known_excluded_warnings"],
                "unknown_outcome_warnings": report["unknown_outcome_warnings"],
                "warning_hours": report["warning_hours"],
                "eligible_events": report["eligible_events"],
                "monitored_hours": report["monitored_hours"],
                "probability": report["probability"],
            }
        )
        if finite:
            evaluated[-1].update(**constraints, **{k: report[k] for k in ("burden_seconds", "burden_occupancy", "eligible_negative_burden_occupancy")})
    passing = [r for r in evaluated if r["meets_constraint"]]
    best = (
        max(
            passing,
            key=lambda r: (
                r["event_sensitivity"],
                -r["false_warnings_per_hour"],
                -r["warnings"],
                -r["burden_seconds"] if finite else -r["warning_hours"],
                r["threshold"],
            ),
        )
        if passing
        else None
    )
    return best, evaluated


def _develop_candidate(
    train_rows, calibration_rows, validation_rows, events, config, family, variant, synthetic=False
):
    name = f"{family}_{variant}"
    started = time.perf_counter()
    try:
        pipeline, schema, details = _fit_model(
            train_rows, family, variant, config["seed"], config["training"]["sample_weighting"]
        )
        fit_seconds = time.perf_counter() - started
        calibrator = _calibrate(
            pipeline, schema, calibration_rows, config["seed"], config["training"]["calibration_c"]
        )
        predictor = Predictor(pipeline, calibrator, schema, config, {"version": name, "synthetic": synthetic})
        validation = [
            dict(row, risk=float(predictor.probabilities([row["features"]])[0]) if row["accepted"] else None)
            for row in validation_rows
        ]
        best, thresholds = _select_policy(validation, events, config)
        raw_validation = [
            dict(
                row,
                risk=float(pipeline.predict_proba(matrix([row["features"]], schema))[0, 1])
                if row["accepted"]
                else None,
            )
            for row in validation_rows
        ]
        return {
            "model": name,
            "status": "developed",
            "training_seconds": fit_seconds,
            "development_seconds": time.perf_counter() - started,
            **details,
            "selected_policy": best,
            "threshold_search": thresholds,
            "feature_variant": variant,
            "feature_count": len(schema),
            "features": schema,
            "calibration": calibration_diagnostics(pipeline, schema, calibrator, calibration_rows),
            "validation_probability": {
                "raw": probability_metrics(raw_validation),
                "calibrated": probability_metrics(validation),
            },
        }, predictor
    except ValueError as error:
        # A failed challenger remains visible; successful baselines can still be assessed and exported.
        return {
            "model": name,
            "status": "failed",
            "training_seconds": time.perf_counter() - started,
            "feature_variant": variant,
            "reason": str(error.args[0]),
            "failure_diagnostics": error.args[1] if len(error.args) > 1 else {},
            "selected_policy": None,
            "threshold_search": [],
        }, None


def _candidate_rank(candidate):
    policy = candidate["selected_policy"]
    family_priority = {"logistic": 0, "elasticnet": 1, "xgboost": 2}
    family = candidate["model"].split("_", 1)[0]
    return (
        policy["event_sensitivity"],
        -policy["false_warnings_per_hour"],
        -policy.get("warnings", 0),
        -policy.get("warning_hours", 0),
        -policy.get("probability", {}).get("group_macro_brier", float("inf")),
        -candidate.get("training_seconds", 0),
        -family_priority.get(family, 3),
        -candidate["feature_count"],
    )


def train_dataset(data_dir, audit_dir, output_dir, config, synthetic=False):
    from .config import validate_config
    from .data import dataset_fingerprint, list_records, read_intervals
    from .labels import eligible_onsets

    data_dir, audit_dir, output_dir = Path(data_dir), Path(audit_dir), Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("Training requires a fresh empty output directory; existing artifacts are preserved")
    audit = json.loads((audit_dir / "audit.json").read_text())
    if bool(audit.get("synthetic")) != bool(synthetic):
        raise ValueError("Explicit synthetic flag must match audited dataset provenance")
    if not audit.get("feasible"):
        raise ValueError(
            "Audit infeasible: real-data training is blocked; revise dataset/target and re-audit"
        )
    if dataset_fingerprint(data_dir) != audit["dataset_fingerprint"]:
        raise ValueError("Dataset differs from the audited inputs; re-audit required")
    validate_config(config)
    if (
        config["alert"].get("max_false_warnings_per_hour") is None
        or config["alert"].get("min_event_sensitivity") is None
    ):
        raise ValueError(
            "Pass explicit provisional false-warning budget and minimum sensitivity before training"
        )
    audited_config = json.loads((audit_dir / "config.json").read_text())
    for section in ["feature_version", "target", "quality", "baseline", "audit", "seed"]:
        if config.get(section) != audited_config.get(section):
            raise ValueError(f"Training {section} differs from audit; re-audit required")
    splits = json.loads((audit_dir / "splits.json").read_text())
    records = list_records(data_dir)
    group_ids = audit.get("patient_group_ids")
    if group_ids is None:
        raise ValueError("Audit missing independent patient groups; re-audit required")
    validate_splits(splits, [record.patient_id for record in records], group_ids)
    rows, events = {name: [] for name in ["train", "calibration", "validation"]}, []
    for record in records:
        partition = splits[record.patient_id]
        if partition == "test" or record.record_id not in audit["included_record_ids"]:
            continue  # Neither final-test signals nor labels are read for model/policy decisions.
        rows[partition].extend(
            dict(row, group_id=group_ids[record.patient_id]) for row in iter_record_rows(record, config)
        )
        if partition == "validation":
            events.extend(
                {"patient_id": record.patient_id, "record_id": record.record_id, "onset": onset}
                for onset in eligible_onsets(
                    read_intervals(record, n_is_other=config["audit"].get("n_rhythm_is_other", False)),
                    record.duration_seconds,
                    config,
                )
            )
    comparisons, fitted = [], {}
    for family, variants in [
        ("logistic", ["clinical", "ecg"]),
        ("xgboost", ["clinical", "ecg", "combined", "personal"]),
        ("logistic", ["ecg_engineered"]),
        ("elasticnet", ["combined_engineered", "personal_engineered"]),
        ("xgboost", ["personal_engineered"]),
    ]:
        for variant in variants:
            comparison, predictor = _develop_candidate(
                rows["train"],
                rows["calibration"],
                rows["validation"],
                events,
                config,
                family,
                variant,
                synthetic,
            )
            comparisons.append(comparison)
            if predictor is not None:
                fitted[comparison["model"]] = predictor
    passing = [comparison for comparison in comparisons if comparison["selected_policy"]]
    output_dir.mkdir(parents=True, exist_ok=True)
    if not passing:
        result = {
            "status": "constraint_not_met",
            "synthetic": synthetic,
            "comparisons": comparisons,
            "limitation": "No model met the explicit validation operating constraint; no model frozen.",
        }
        (output_dir / "training_report.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        return result
    # Explicit simplicity order wins ties; max retains earlier order when every criterion ties.
    chosen = max(passing, key=_candidate_rank)
    predictor = fitted[chosen["model"]]
    frozen_config = deepcopy(config)
    frozen_config["alert"].update(
        {key: chosen["selected_policy"][key] for key in ["threshold", "release_threshold"]}
    )
    baseline = [r for r in passing if r["model"].startswith("logistic")]
    baseline_sensitivity = max((r["selected_policy"]["event_sensitivity"] for r in baseline), default=None)
    improved = (
        baseline_sensitivity is not None
        and chosen["selected_policy"]["event_sensitivity"] > baseline_sensitivity
    )
    limitations = [
        "Research prototype; no treatment decisions or clinical readiness claim.",
        "Small patient cohort and overlapping windows limit inference; test evaluation is a separate command.",
        "SHDB-AF population is not Indian external validation; source zero-phase filtering limits streaming claims.",
        "Operating targets are explicitly supplied provisional engineering targets, not clinician approval.",
    ]
    if synthetic:
        limitations.insert(0, "All results are SYNTHETIC plumbing checks, not clinical forecasting evidence.")
        limitations.insert(
            1, "Synthetic fixtures may be replayed for software corrections; no blinded clinical test claim."
        )
    manifest = {
        "version": "cardiotwin-0.1-" + chosen["model"],
        "frozen": True,
        "synthetic": bool(synthetic),
        "selected_model": chosen["model"],
        "seed": config["seed"],
        "feature_version": FEATURE_VERSION,
        "training": deepcopy(config["training"]),
        "patient_group_ids": group_ids,
        "cohort_contract": cohort_contract(splits, group_ids, audit["included_record_ids"]),
        "cohort_contract_sha256": cohort_contract(splits, group_ids, audit["included_record_ids"])["sha256"],
        "comparison_failures": [
            {"model": c["model"], "reason": c["reason"]} for c in comparisons if c["status"] == "failed"
        ],
        "dependencies": dependency_versions(),
        "dataset_fingerprint": audit["dataset_fingerprint"],
        "included_record_ids": audit["included_record_ids"],
        "limitations": limitations,
        "acceptance": {
            "meets_validation_operating_constraint": True,
            "beats_logistic_sensitivity": bool(improved),
            "clinical_readiness": False,
        },
    }
    manifest = _save_predictor(predictor, output_dir, frozen_config, manifest, splits)
    for comparison in comparisons:
        name = comparison["model"]
        if name not in fitted:
            continue
        comparison_config = deepcopy(config)
        policy = comparison["selected_policy"]
        if policy:
            comparison_config["alert"].update(
                {key: policy[key] for key in ["threshold", "release_threshold"]}
            )
        comparison_manifest = dict(
            manifest,
            version="cardiotwin-0.1-" + name,
            selected_model=name,
            acceptance={"meets_validation_operating_constraint": bool(policy), "clinical_readiness": False},
        )
        _save_predictor(
            fitted[name], output_dir / "comparisons" / name, comparison_config, comparison_manifest, splits
        )
    result = {
        "status": "frozen",
        "synthetic": bool(synthetic),
        "selected_model": chosen["model"],
        "validation_policy": chosen["selected_policy"],
        "comparisons": comparisons,
        "acceptance": manifest["acceptance"],
        "test_opened": False,
        "cohort_contract_sha256": manifest["cohort_contract_sha256"],
        "development_rows": {key: len(value) for key, value in rows.items()},
    }
    (output_dir / "training_report.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    card = (
        f"# CardioTwin model card\n\nModel: {manifest['version']}\n\n"
        f"Provenance: {'SYNTHETIC demonstration' if synthetic else 'audited research dataset'}.\n\n"
        "Forecast contract: trailing history and inclusive future gap/horizon are frozen in config.json. "
        "Current AF, unknown labels, insufficient history/follow-up and washout are excluded from labels. "
        "Runtime never accesses rhythm labels and can abstain for signal quality or baseline warmup.\n\n"
        f"Validation policy: {json.dumps(chosen['selected_policy'])}\n\n"
        "Imputation/scaling/model weights and grouped hyperparameter selection use training patients only. "
        "An independent, unweighted sigmoid calibrator uses calibration patients. Validation patients select "
        "threshold and candidate; final test remains unopened until evaluate.\n\n"
        "Evidence contributions are raw score explanations, not calibrated probability attribution or causality.\n\n"
        + "\n".join(f"- {text}" for text in limitations)
        + "\n"
    )
    (output_dir / "model_card.md").write_text(card)
    return result


def _save_predictor(predictor, path, config, manifest, splits):
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": predictor.pipeline,
            "calibrator": predictor.calibrator,
            "schema": predictor.schema,
            "config": config,
            "cohort_contract": manifest.get("cohort_contract"),
        },
        path / "model.joblib",
    )
    with (path / "model.joblib").open("rb") as stream:
        model_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = dict(manifest, model_sha256=model_hash, version=manifest["version"] + "-" + model_hash[:12])
    for filename, payload in [
        ("config.json", config),
        ("manifest.json", manifest),
        (
            "feature_schema.json",
            {
                "features": predictor.schema,
                "missing_indicators": ["missingindicator_" + key for key in predictor.schema],
            },
        ),
        ("splits.json", splits),
    ]:
        (path / filename).write_text(json.dumps(payload, indent=2, allow_nan=False))

    return manifest
