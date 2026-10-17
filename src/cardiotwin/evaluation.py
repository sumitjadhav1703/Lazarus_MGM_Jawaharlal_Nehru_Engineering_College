"""Chronological monitoring evaluation using the exact runtime warning policy."""

import json
import math
import platform
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def match_warnings(warnings, onsets, gap, horizon):
    """Earliest warning first, earliest eligible unmatched event; inclusive interval."""
    remaining = sorted(float(onset) for onset in onsets)
    matches, false = [], []
    for warning in sorted(float(t) for t in warnings):
        eligible = next((event for event in remaining if gap <= event - warning <= horizon), None)
        if eligible is None:
            false.append(warning)
        else:
            remaining.remove(eligible)
            matches.append({"warning": warning, "onset": eligible, "lead_seconds": eligible - warning})
    return {"matches": matches, "false_warnings": false, "missed_events": remaining}


def probability_metrics(rows):
    labeled = [r for r in rows if r.get("label") in (0, 1) and r.get("risk") is not None]
    if not labeled:
        return {
            "labeled_windows": 0,
            "brier": None,
            "pr_auc": None,
            "auroc": None,
            "log_loss": None,
            "calibration": [],
            "expected_calibration_error": None,
            "positive_prevalence": None,
            "pr_auc_reference": None,
            "cohort_prevalence_brier_reference": None,
            "patient_macro_brier": None,
            "group_macro_brier": None,
            "probability_groups": 0,
            "probability_patients": 0,
        }
    y = np.asarray([r["label"] for r in labeled])
    p = np.asarray([r["risk"] for r in labeled])
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Forecast probabilities must be finite and in [0,1]")
    patient_errors, group_errors, patient_groups = defaultdict(list), defaultdict(list), {}
    for row in labeled:
        patient, group = row["patient_id"], row.get("group_id", row["patient_id"])
        if patient in patient_groups and patient_groups[patient] != group:
            raise ValueError("A patient cannot belong to multiple probability groups")
        patient_groups[patient] = group
        squared_error = (row["risk"] - row["label"]) ** 2
        patient_errors[patient].append(squared_error)
        group_errors[group].append(squared_error)
    prevalence = float(y.mean())
    bins, error = [], 0.0
    indices = np.minimum((p * 10).astype(int), 9)
    for i in range(10):
        selected = indices == i
        if selected.any():
            mean_p, rate = float(p[selected].mean()), float(y[selected].mean())
            bins.append(
                {
                    "lower": i / 10,
                    "upper": (i + 1) / 10,
                    "count": int(selected.sum()),
                    "mean_probability": mean_p,
                    "event_rate": rate,
                }
            )
            error += selected.mean() * abs(mean_p - rate)
    both = len(np.unique(y)) == 2
    return {
        "labeled_windows": len(y),
        "brier": float(brier_score_loss(y, p)),
        "pr_auc": float(average_precision_score(y, p)) if both else None,
        "auroc": float(roc_auc_score(y, p)) if both else None,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "calibration": bins,
        "expected_calibration_error": float(error),
        "positive_prevalence": prevalence,
        "pr_auc_reference": prevalence,
        "cohort_prevalence_brier_reference": prevalence * (1 - prevalence),
        "reference_note": "Observed cohort prevalence is a descriptive reference, not a fitted deployable baseline.",
        "patient_macro_brier": float(np.mean([np.mean(v) for v in patient_errors.values()])),
        "probability_patients": len(patient_errors),
        "group_macro_brier": float(np.mean([np.mean(v) for v in group_errors.values()])),
        "probability_groups": len(group_errors),
        "macro_metric_note": "Selection uses equal audited independent groups; patient_macro_brier retains original-ID descriptive aggregation.",
    }


def _rates(counts):
    hours = counts["monitored_seconds"] / 3600
    eligible_hours = counts["eligible_seconds"] / 3600
    assessed_warnings = counts["warnings"] - counts["unassessable_warnings"]
    rates = {
        "event_sensitivity": counts["matched_events"] / counts["eligible_events"]
        if counts["eligible_events"]
        else None,
        "warning_precision": counts["matched_events"] / assessed_warnings if assessed_warnings else None,
        "false_warnings_per_hour": counts["false_warnings"] / eligible_hours if eligible_hours else None,
        "false_warnings_per_patient_day": counts["false_warnings"] / (eligible_hours / 24)
        if eligible_hours
        else None,
        "false_warnings_per_all_recording_hour": counts["false_warnings"] / hours if hours else None,
        "false_warnings_per_eligible_hour": counts["false_warnings"] / eligible_hours
        if eligible_hours
        else None,
        "forecast_coverage": counts["available_seconds"] / counts["monitored_seconds"]
        if counts["monitored_seconds"]
        else None,
    }
    if "burden_seconds" in counts:
        rates.update(burden_occupancy=counts["burden_seconds"] / counts["monitored_seconds"] if counts["monitored_seconds"] else None,
                     eligible_negative_burden_occupancy=counts["eligible_negative_burden_seconds"] / counts["eligible_negative_seconds"] if counts["eligible_negative_seconds"] else None)
    return rates


def policy_constraints(report, config, *, budget, minimum_sensitivity=0.0):
    failures = []
    for key, limit, lower in (("false_warnings_per_hour", budget, False),
                              ("event_sensitivity", minimum_sensitivity, True)):
        value = report.get(key)
        if value is None or not math.isfinite(value) or (value < limit if lower else value > limit):
            failures.append(key)
    if config["alert"].get("policy_version") == "finite-validity-v1":
        for key in ("burden_occupancy", "eligible_negative_burden_occupancy"):
            value = report.get(key)
            if value is None or not math.isfinite(value) or value > config["alert"]["max_" + key]:
                failures.append(key)
    return {"passed": not failures, "failed_criteria": failures}


def evaluate_stream(rows, events, config, bootstrap_samples=300, *, include_probability=True):
    if not include_probability and bootstrap_samples:
        raise ValueError("Probability-free policy evaluation requires bootstrap_samples=0")
    from .alerts import advance_alert, alert_interval_seconds

    groups, event_groups = defaultdict(list), defaultdict(list)
    for row in rows:
        groups[(row["patient_id"], row["record_id"])].append(row)
    for event in events:
        event_groups[(event["patient_id"], event["record_id"])].append(float(event["onset"]))
    patients = defaultdict(Counter)
    exclusions, rejections, warning_exclusions = Counter(), Counter(), Counter()
    warning_details, missed_details, leads = [], [], []
    known_excluded_reasons, unknown_outcome_reasons = Counter(), Counter()
    known_reasons = {"current_AF", "washout", "gap", "ineligible_event"}
    cadence = config["target"]["cadence_seconds"]
    finite = config["alert"].get("policy_version") == "finite-validity-v1"
    finite_fields = ("burden_seconds", "unresolved_elevated_seconds", "rearm_wait_seconds",
                     "unavailable_rearm_seconds", "missing_cadence_seconds", "eligible_negative_seconds",
                     "eligible_negative_burden_seconds")
    burden_categories = Counter()
    gap, horizon = config["target"]["gap_seconds"], config["target"]["horizon_seconds"]
    for (patient, record), record_rows in groups.items():
        record_rows = sorted(record_rows, key=lambda r: r["timestamp"])
        counts = patients[patient]
        if finite:
            for key in finite_fields:
                counts.setdefault(key, 0.)
        state, issued = {}, {}
        for i, row in enumerate(record_rows):
            t = float(row["timestamp"])
            next_t = (
                float(record_rows[i + 1]["timestamp"])
                if i + 1 < len(record_rows)
                else float(row.get("duration_seconds", t + cadence))
            )
            dt = max(0.0, next_t - t)
            if finite:
                next_t = min(next_t, float(row.get("duration_seconds", next_t)))
                dt = max(0., next_t - t)
            available = bool(row.get("accepted") and row.get("risk") is not None)
            state, warning = advance_alert(state, t, row.get("risk"), available, config)
            if warning:
                issued[float(warning["timestamp"])] = dict(warning)
            counts["monitored_seconds"] += dt
            supported = min(dt, cadence) if finite else dt
            if finite:
                durations = alert_interval_seconds(state, t, t + dt, config)
                counts.update(durations)
                counts["rejected_seconds"] += dt - durations["available_seconds"]
                counts["missing_cadence_seconds"] += dt - supported
                exclusions["missing_cadence"] += dt - supported
                rejections["missing_cadence"] += dt - supported
                category = ("eligible_negative" if row.get("label") == 0 else "eligible_positive" if row.get("label") == 1
                            else row.get("label_reason") or "unknown")
                burden_categories[category] += durations["burden_seconds"]
                counts["burden_" + category + "_seconds"] += durations["burden_seconds"]
                if row.get("label") == 0:
                    counts["eligible_negative_seconds"] += supported
                    counts["eligible_negative_burden_seconds"] += durations["burden_seconds"]
            else:
                counts["available_seconds"] += dt if available else 0
                counts["rejected_seconds"] += 0 if available else dt
                counts["warning_seconds"] += dt if state.get("active") else 0
            counts["eligible_seconds"] += supported if row.get("label") in (0, 1) else 0
            if not available:
                rejections[row.get("reason") or "forecast_unavailable"] += supported
            if row.get("label") is None:
                exclusions[row.get("label_reason") or "unlabeled"] += supported
        counts["cooldown_suppressed_updates"] += state.get("suppressed_count", 0)
        at_time = {float(row["timestamp"]): row for row in record_rows}
        unknown = [t for t in issued if at_time[t].get("label") not in (0, 1)]
        assessable = [t for t in issued if t not in unknown]
        result = match_warnings(assessable, event_groups[(patient, record)], gap, horizon)
        warning_exclusions.update(at_time[t].get("label_reason") or "unlabeled" for t in unknown)
        for t in unknown:
            reason = at_time[t].get("label_reason") or "unlabeled"
            (known_excluded_reasons if reason in known_reasons else unknown_outcome_reasons)[reason] += 1
        known_false = result["false_warnings"]
        matched_times = {match["warning"] for match in result["matches"]}
        counts["warnings"] += len(issued)
        counts["eligible_events"] += len(event_groups[(patient, record)])
        counts["matched_events"] += len(result["matches"])
        counts["false_warnings"] += len(known_false)
        counts["unassessable_warnings"] += len(unknown)
        if finite:
            counts["assessable_warnings"] += len(assessable)
            counts["known_excluded_warnings"] += sum(at_time[t].get("label_reason") in known_reasons for t in unknown)
            counts["unknown_outcome_warnings"] += sum(at_time[t].get("label_reason") not in known_reasons for t in unknown)
        counts["missed_events"] += len(result["missed_events"])
        leads.extend(match["lead_seconds"] for match in result["matches"])
        for onset in result["missed_events"]:
            opportunities = [
                r
                for r in record_rows
                if gap <= onset - r["timestamp"] <= horizon and r.get("label") in (0, 1)
            ]
            abstention = not any(r.get("accepted") and r.get("risk") is not None for r in opportunities)
            counts["events_missed_during_abstention"] += int(abstention)
            missed_details.append(
                {
                    "patient_id": patient,
                    "record_id": record,
                    "onset": onset,
                    "all_warning_opportunities_unavailable": abstention,
                }
            )
        end = record_rows[-1].get("duration_seconds", record_rows[-1]["timestamp"])
        for t, warning in issued.items():
            warning_end = warning.get("end_time") or end
            if finite:
                warning_end = min(warning_end, t + config["alert"]["max_duration_seconds"],
                                  record_rows[-1]["timestamp"] + cadence, end)
            warning_details.append(
                {
                    **warning,
                    "patient_id": patient,
                    "record_id": record,
                    "end_time": warning_end,
                    "matched": t in matched_times,
                    "outcome": "unassessable"
                    if t in unknown
                    else ("false" if t in known_false else "matched"),
                    "annotation_exclusion": at_time[t].get("label_reason"),
                    "assessment_category": "known_excluded"
                    if t in unknown and at_time[t].get("label_reason") in known_reasons
                    else "unknown_outcome"
                    if t in unknown
                    else "assessable",
                }
            )
    totals = Counter()
    for counts in patients.values():
        totals.update(counts)
    for key in [
        "monitored_seconds",
        "eligible_seconds",
        "available_seconds",
        "rejected_seconds",
        "warning_seconds",
        "eligible_events",
        "matched_events",
        "warnings",
        "false_warnings",
        "missed_events",
        "events_missed_during_abstention",
        "cooldown_suppressed_updates",
        "unassessable_warnings",
    ]:
        totals.setdefault(key, 0)
    report = {
        **dict(totals),
        **_rates(totals),
        "patients": len(patients),
        "records": len(groups),
        "monitored_hours": totals["monitored_seconds"] / 3600,
        "eligible_hours": totals["eligible_seconds"] / 3600,
        "rejected_hours": totals["rejected_seconds"] / 3600,
        "warning_hours": totals["warning_seconds"] / 3600,
        "exclusion_hours": {key: value / 3600 for key, value in exclusions.items()},
        "rejection_hours": {key: value / 3600 for key, value in rejections.items()},
        "unassessable_warning_reasons": dict(warning_exclusions),
        "known_excluded_warnings": sum(known_excluded_reasons.values()),
        "unknown_outcome_warnings": sum(unknown_outcome_reasons.values()),
        "known_excluded_warning_reasons": dict(known_excluded_reasons),
        "unknown_outcome_warning_reasons": dict(unknown_outcome_reasons),
        "lead_seconds": leads,
        "warning_details": warning_details,
        "missed_event_details": missed_details,
        "probability": probability_metrics(rows) if include_probability else None,
        "per_patient": [
            {
                "patient_id": patient,
                **dict(counts),
                **_rates(counts),
                "eligible_hours": counts["eligible_seconds"] / 3600,
                "monitored_hours": counts["monitored_seconds"] / 3600,
                "warning_hours": counts["warning_seconds"] / 3600,
            }
            for patient, counts in sorted(patients.items())
        ],
        "lead_time_summary": {
            "p25_seconds": float(np.percentile(leads, 25)) if leads else None,
            "median_seconds": float(np.median(leads)) if leads else None,
            "p75_seconds": float(np.percentile(leads, 75)) if leads else None,
            "event_fraction_with_5min_lead": sum(t >= 300 for t in leads) / totals["eligible_events"]
            if totals["eligible_events"]
            else None,
        },
        "denominator": "Canonical false-warning rate uses adequately labeled forecast-eligible time, "
        "including quality-rejected windows. Unknown/current-AF/excluded time cannot dilute "
        "the operating budget. Total monitored time/coverage include all recording time, "
        "and false_warnings_per_all_recording_hour is a companion rate. Every eligible "
        "onset remains in event sensitivity, including events missed during abstention.",
    }
    report["confidence_intervals"] = patient_bootstrap(
        patients, rows, config.get("seed", 42), bootstrap_samples
    )
    if finite:
        durations = [w["end_time"] - w["timestamp"] for w in warning_details]
        report.update(policy_version="finite-validity-v1", burden_seconds_by_label=dict(burden_categories),
                      warning_duration_summary={name: float(np.percentile(durations, q)) if durations else None
                                                for name, q in (("p25_seconds", 25), ("median_seconds", 50),
                                                                ("p75_seconds", 75), ("max_seconds", 100))})
        report["denominator"] += " Finite policy: labels and availability support at most one cadence; missing-cadence time stays monitored but is excluded from eligible hours. Burden is active plus unresolved elevated time."
    return report


def patient_bootstrap(patients, rows, seed=42, samples=300):
    """Resample whole patients, including all their records and overlapping windows."""
    patient_groups = {}
    for row in rows:
        patient, group = row["patient_id"], row.get("group_id", row["patient_id"])
        if patient in patient_groups and patient_groups[patient] != group:
            raise ValueError("A patient cannot belong to multiple bootstrap groups")
        patient_groups[patient] = group
    grouped_counts = defaultdict(Counter)
    for patient, counts in patients.items():
        grouped_counts[patient_groups.get(patient, patient)].update(counts)
    ids = list(grouped_counts)
    if len(ids) < 2 or not samples:
        return {
            "method": "patient bootstrap",
            "samples": samples,
            "independent_groups": len(ids),
            "intervals": {},
            "limitation": "At least two independent patient groups required for bootstrap intervals.",
        }
    rng, estimates = np.random.default_rng(seed), defaultdict(list)
    # Multiplicity weights exactly equal copying every window once per sampled
    # group. Retain numeric arrays instead of hundreds of thousands of new dicts.
    codes = {group: number for number, group in enumerate(ids)}
    labeled = [
        row
        for row in rows
        if row.get("label") in (0, 1)
        and row.get("risk") is not None
        and patient_groups[row["patient_id"]] in codes
    ]
    y = np.asarray([row["label"] for row in labeled], dtype=np.int8)
    p = np.asarray([row["risk"] for row in labeled], dtype=float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Forecast probabilities must be finite and in [0,1]")
    row_groups = np.asarray([codes[patient_groups[row["patient_id"]]] for row in labeled], dtype=int)
    errors = (p - y) ** 2
    group_count = np.bincount(row_groups, minlength=len(ids))
    group_error = np.bincount(row_groups, weights=errors, minlength=len(ids))
    group_mean = np.divide(group_error, group_count, out=np.zeros(len(ids)), where=group_count > 0)
    bins = np.minimum((p * 10).astype(int), 9)
    for _ in range(samples):
        selected = rng.choice(ids, size=len(ids), replace=True)
        multiplicity = np.zeros(len(ids), dtype=int)
        total = Counter()
        for group in selected:
            total.update(grouped_counts[group])
            multiplicity[codes[group]] += 1
        metrics = _rates(total)
        weights = multiplicity[row_groups]
        labeled_count = weights.sum()
        if labeled_count:
            # Weighted sklearn ranking metrics preserve ties and class semantics.
            present = np.bincount(y, weights=weights, minlength=2) > 0
            both = bool(present.all())
            bin_error = np.bincount(bins, weights=weights * (p - y), minlength=10)
            macro_weights = multiplicity * (group_count > 0)
            metrics.update(
                brier=float(np.dot(errors, weights) / labeled_count),
                pr_auc=float(average_precision_score(y, p, sample_weight=weights)) if both else None,
                auroc=float(roc_auc_score(y, p, sample_weight=weights)) if both else None,
                expected_calibration_error=float(np.abs(bin_error).sum() / labeled_count),
                group_macro_brier=float(np.dot(group_mean, macro_weights) / macro_weights.sum()),
            )
        for key, value in metrics.items():
            if value is not None:
                estimates[key].append(value)
    return {
        "method": "patient bootstrap percentile 95%",
        "samples": samples,
        "independent_groups": len(ids),
        "intervals": {
            key: {
                "lower": float(np.percentile(values, 2.5)),
                "upper": float(np.percentile(values, 97.5)),
                "valid_samples": len(values),
            }
            for key, values in estimates.items()
        },
        "limitation": "Small patient cohorts produce unstable intervals. All-correct events can yield a "
        "degenerate interval; bootstrap cannot bound unobserved misses or false warnings. "
        "Synthetic intervals are not clinical evidence.",
    }


def evaluate_artifact(data_dir, audit_dir, artifact_dir, output_dir):
    from .data import dataset_fingerprint, list_records, read_intervals
    from .labels import eligible_onsets
    from .modeling import Predictor, matrix
    from .train import cohort_contract, iter_record_rows, validate_cohort_contract, validate_splits

    artifact_dir, audit_dir, output_dir = Path(artifact_dir), Path(audit_dir), Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("Evaluation requires a fresh empty output directory; previous reports are preserved")
    predictor = Predictor.load(artifact_dir)
    if not predictor.manifest.get("frozen"):
        raise ValueError("Evaluation requires a frozen artifact")
    splits = json.loads((artifact_dir / "splits.json").read_text())
    records = list_records(Path(data_dir))
    group_ids = predictor.manifest.get("patient_group_ids", {r.patient_id: r.patient_id for r in records})
    validate_splits(splits, [r.patient_id for r in records], group_ids)
    bound_contract = validate_cohort_contract(
        predictor.manifest, splits, group_ids, predictor.manifest["included_record_ids"]
    )
    if dataset_fingerprint(Path(data_dir)) != predictor.manifest["dataset_fingerprint"]:
        raise ValueError("Dataset differs from frozen artifact; re-audit required")
    if splits != json.loads((audit_dir / "splits.json").read_text()):
        raise ValueError("Audit split differs from frozen artifact")
    audit = json.loads((audit_dir / "audit.json").read_text())
    if bound_contract:
        validate_cohort_contract(
            predictor.manifest,
            splits,
            audit.get("patient_group_ids", {}),
            audit.get("included_record_ids", []),
        )
    comparisons = {
        path.name: Predictor.load(path)
        for path in sorted((artifact_dir / "comparisons").glob("*"))
        if path.is_dir()
    }
    for model in comparisons.values():
        validate_cohort_contract(model.manifest, splits, group_ids, predictor.manifest["included_record_ids"])
    cohort_identity = cohort_contract(splits, group_ids, predictor.manifest["included_record_ids"])["sha256"]
    consumption_dir = audit_dir / "test-consumption"
    consumption_dir.mkdir(exist_ok=True)
    earlier = []
    for path in consumption_dir.glob("*.json"):
        entry = json.loads(path.read_text())
        if (
            entry.get("dataset_fingerprint") == predictor.manifest["dataset_fingerprint"]
            and entry.get("cohort_contract_sha256") == cohort_identity
            and entry.get("status") == "started"
        ):
            earlier.append(entry["evaluation_id"])
    evaluation_id = str(uuid4())
    consumption = {
        "evaluation_id": evaluation_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_fingerprint": predictor.manifest["dataset_fingerprint"],
        "cohort_contract_sha256": cohort_identity,
        "model_sha256": predictor.manifest.get("model_sha256"),
        "model_version": predictor.version,
        "output_dir": str(output_dir.resolve()),
        "test_patient_ids": sorted(p for p in splits if splits[p] == "test"),
        "previous_evaluation_ids": sorted(earlier),
        "previously_observed": bool(earlier) or predictor.manifest["synthetic"] or not bound_contract,
        "history_note": "Ledger records accesses through this evaluator; absence of entries cannot certify historical blinding.",
        "status": "started",
    }
    with (consumption_dir / (evaluation_id + "-started.json")).open("x") as stream:
        json.dump(consumption, stream, indent=2, allow_nan=False)
    rows, events, latency, replay_latency = [], [], [], []
    for record in records:
        if (
            splits[record.patient_id] != "test"
            or record.record_id not in predictor.manifest["included_record_ids"]
        ):
            continue
        iterator = iter(iter_record_rows(record, predictor.config))
        while True:
            replay_start = time.perf_counter()
            try:
                row = next(iterator)
            except StopIteration:
                break
            row["group_id"] = predictor.manifest.get("patient_group_ids", {}).get(
                record.patient_id, record.patient_id
            )
            if row["accepted"]:
                start = time.perf_counter()
                row["risk"] = predictor.predict(row["features"])["risk"]
                latency.append((time.perf_counter() - start) * 1000)
            else:
                row["risk"] = None
            if row["timestamp"] >= predictor.config["target"]["history_seconds"]:
                replay_latency.append((time.perf_counter() - replay_start) * 1000)
            row["uncalibrated_risk"] = (
                float(predictor.pipeline.predict_proba(matrix([row["features"]], predictor.schema))[0, 1])
                if row["accepted"]
                else None
            )
            row["comparison_risks"] = {
                name: float(model.probabilities([row["features"]])[0]) if row["accepted"] else None
                for name, model in comparisons.items()
            }
            rows.append(row)
        events += [
            {"patient_id": record.patient_id, "record_id": record.record_id, "onset": onset}
            for onset in eligible_onsets(
                read_intervals(record, n_is_other=predictor.config["audit"].get("n_rhythm_is_other", False)),
                record.duration_seconds,
                predictor.config,
            )
        ]
    report = evaluate_stream(rows, events, predictor.config)
    report["probability_before_calibration"] = probability_metrics(
        [dict(row, risk=row["uncalibrated_risk"]) for row in rows]
    )
    report["comparisons"] = {
        name: {
            **evaluate_stream(
                [dict(row, risk=row["comparison_risks"][name]) for row in rows], events, model.config
            ),
            "frozen_alert_policy": model.config["alert"],
            "validation_acceptance": model.manifest["acceptance"],
        }
        for name, model in comparisons.items()
    }
    report.update(
        {
            "status": "evaluated",
            "partition": "synthetic_test" if predictor.manifest["synthetic"] else "held_out_test",
            "test_consumption": consumption,
            "cohort_integrity": "bound" if bound_contract else "legacy_unbound",
            "model_version": predictor.version,
            "synthetic": predictor.manifest["synthetic"],
            "alert_policy": predictor.config["alert"],
            "inference_latency_ms": {
                "median": float(np.median(latency)) if latency else None,
                "p95": float(np.percentile(latency, 95)) if latency else None,
                "calls": len(latency),
                "scope": "predict with feature evidence; excludes signal I/O and extraction",
            },
            "replay_pipeline_latency_ms": {
                "median": float(np.median(replay_latency)) if replay_latency else None,
                "p95": float(np.percentile(replay_latency, 95)) if replay_latency else None,
                "calls": len(replay_latency),
                "scope": "Offline row generation, ECG I/O, features, labels and selected predictor/evidence; excludes comparison inference and bootstrap",
            },
            "hardware": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python": platform.python_version(),
            },
            "limitations": predictor.manifest["limitations"],
        }
    )
    policy = predictor.config["alert"]
    sensitivity = report["event_sensitivity"]
    report["operating_constraint_met_on_test"] = (
        sensitivity is not None
        and report["false_warnings_per_hour"] is not None
        and sensitivity >= policy["min_event_sensitivity"]
        and report["false_warnings_per_hour"] <= policy["max_false_warnings_per_hour"]
    )
    bounds = report["confidence_intervals"]["intervals"].get("event_sensitivity")
    report["uncertainty_flags"] = []
    if not bound_contract:
        report["uncertainty_flags"].append(
            "Legacy artifact does not bind split/group/inclusion decisions; membership cannot be authenticated retrospectively."
        )
    report["test_protocol"] = (
        "Held-out membership is frozen for new contracts; access ledger does not certify historical blinding. No test-based model selection is permitted."
    )
    if not bounds or bounds["upper"] - bounds["lower"] > 0.2:
        report["uncertainty_flags"].append(
            "Event-sensitivity interval unavailable or wider than 0.20; more patients needed."
        )
    elif bounds["upper"] == bounds["lower"]:
        report["uncertainty_flags"].append(
            "Degenerate patient-bootstrap sensitivity interval; observed uniform "
            "outcomes do not bound unobserved misses or future false warnings."
        )
    if sensitivity is None or sensitivity < policy["min_event_sensitivity"]:
        report["uncertainty_flags"].append(
            "Test event sensitivity failed the frozen minimum; no clinical readiness claim."
        )
    if report["synthetic"]:
        report["uncertainty_flags"].append(
            "Synthetic outcomes and bootstrap intervals cannot establish clinical performance."
        )
        report["test_protocol"] = (
            "Synthetic fixtures can be replayed for software corrections; this is not a blinded clinical test."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    with (output_dir / "predictions.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    with (consumption_dir / (evaluation_id + "-completed.json")).open("x") as stream:
        json.dump(
            {**consumption, "status": "completed", "timestamp_utc": datetime.now(timezone.utc).isoformat()},
            stream,
            indent=2,
            allow_nan=False,
        )
    return report
