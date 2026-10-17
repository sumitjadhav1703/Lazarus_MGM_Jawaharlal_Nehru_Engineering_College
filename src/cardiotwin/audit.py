"""Offline cohort/target feasibility; signal quality and patient grouping precede training."""

import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

from .data import (
    dataset_fingerprint,
    list_records,
    read_intervals,
    read_signal,
    record_signal_hash,
    signal_available,
)
from .features import assess_quality, extract_features
from .labels import eligible_onsets, label_at


def _splits(records, event_patients, parent, config, existing=None):
    def root(patient):
        while parent[patient] != patient:
            patient = parent[patient]
        return patient

    groups = defaultdict(list)
    for record in records:
        if record.patient_id not in groups[root(record.patient_id)]:
            groups[root(record.patient_id)].append(record.patient_id)
    positive, negative = [], []
    for group, patients in sorted(groups.items()):
        (positive if any(p in event_patients for p in patients) else negative).append(group)
    rng = random.Random(config.get("seed", 42))
    result = {}
    for cohort in [positive, negative]:
        rng.shuffle(cohort)
        n = len(cohort)
        test_n = max(
            round(n * config["audit"].get("test_fraction", 0.15)),
            config["audit"]["min_test_event_subjects"] if cohort is positive else 0,
        )
        test_n = min(test_n, max(0, n - 3))
        calibration_n = min(max(1, round(n * 0.15)), max(0, n - test_n - 2))
        validation_n = min(max(1, round(n * 0.15)), max(0, n - test_n - calibration_n - 1))
        lengths = [("test", test_n), ("calibration", calibration_n), ("validation", validation_n)]
        cursor = 0
        for name, count in lengths + [("train", n - test_n - calibration_n - validation_n)]:
            for group in cohort[cursor : cursor + count]:
                result.update({patient: name for patient in groups[group]})
            cursor += count
    if existing:
        for patients in groups.values():
            frozen = {existing[p] for p in patients if p in existing}
            if len(frozen) > 1:
                raise ValueError(
                    "Duplicate/overlap links conflict with frozen patient partitions; use a new audit directory"
                )
            if frozen:
                partition = frozen.pop()
                if partition not in {"train", "calibration", "validation", "test"}:
                    raise ValueError("Invalid frozen split partition")
                result.update({patient: partition for patient in patients})
    return result


def audit_dataset(data_dir, output_dir, config, annotation_only=False):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = list_records(data_dir)
    synthetic = (data_dir / "manifest.json").exists() and bool(
        json.loads((data_dir / "manifest.json").read_text()).get("synthetic")
    )
    n_is_other = bool(config.get("audit", {}).get("n_rhythm_is_other", False))
    parent = {r.patient_id: r.patient_id for r in records}

    def root(patient):
        while parent[patient] != patient:
            patient = parent[patient]
        return patient

    def join(a, b):
        parent[root(b)] = root(a)

    exclusions, field_counts = Counter(), defaultdict(Counter)
    duplicate_ids, overlap_ids, seen_hashes = set(), set(), {}
    for record in records:
        if signal_available(record) and not annotation_only:
            record.signal_hash = record_signal_hash(record)
            if record.signal_hash in seen_hashes:
                original = seen_hashes[record.signal_hash]
                join(record.patient_id, original.patient_id)
                duplicate_ids.add(record.record_id)
            else:
                seen_hashes[record.signal_hash] = record
        for field in record.metadata_audit:
            source = field.get("field") or field["source"]
            field_counts[source]["available" if field["available"] else "unavailable"] += 1
            field_counts[source][field["reason"]] += 1
    by_patient = defaultdict(list)
    for record in records:
        by_patient[root(record.patient_id)].append(record)
    for grouped in by_patient.values():
        with_dates = sorted([r for r in grouped if r.start_time], key=lambda r: r.start_time)
        retained = []
        for record in with_dates:
            start = datetime.fromisoformat(record.start_time)
            end = start + timedelta(seconds=record.duration_seconds)
            if any(
                start < previous_end and previous_start < end for previous_start, previous_end in retained
            ):
                overlap_ids.add(record.record_id)
            else:
                retained.append((start, end))
    cohort, event_patients, total_events, monitoring_seconds, candidates = [], set(), 0, 0.0, 0
    candidate_subjects = set()
    n_as_other_candidates = 0
    alternate_event_subjects = set()
    rhythm_seconds, interval_counts = Counter(), Counter()
    short_af_intervals = 0
    history, cadence = (config["target"][key] for key in ["history_seconds", "cadence_seconds"])
    for record in records:
        intervals = read_intervals(record, n_is_other=n_is_other)
        annotated_onsets = eligible_onsets(intervals, record.duration_seconds, config)
        for interval in intervals:
            rhythm_seconds[interval.rhythm] += interval.end - interval.start
            interval_counts[interval.rhythm] += 1
            if (
                interval.rhythm == "AF"
                and interval.end - interval.start < config["target"]["min_episode_seconds"]
            ):
                short_af_intervals += 1
        if not synthetic and getattr(record, "source", "shdb-af") == "shdb-af":
            alternate = read_intervals(record, n_is_other=True)
            alternative_onsets = eligible_onsets(alternate, record.duration_seconds, config)
            n_as_other_candidates += len(alternative_onsets)
            if alternative_onsets:
                alternate_event_subjects.add(root(record.patient_id))
        candidates += len(annotated_onsets)
        if annotated_onsets:
            candidate_subjects.add(root(record.patient_id))
        has_signal = signal_available(record)
        reason = (
            "duplicate"
            if record.record_id in duplicate_ids
            else "overlap"
            if record.record_id in overlap_ids
            else "unannotated"
            if not record.annotated
            else "signal_not_downloaded"
            if not has_signal
            else None
        )
        accepted_onsets, counts, hours = set(), Counter(), 0.0
        quality_updates = 0
        warmup_updates = config.get("baseline", {}).get("minimum_updates", 0)
        if reason:
            exclusions[reason] += 1
        else:
            t = float(history)
            while t + config["target"]["horizon_seconds"] <= record.duration_seconds:
                label, exclusion = label_at(t, intervals, record.duration_seconds, config)
                if annotation_only:
                    if label is not None:
                        counts["annotation_positive" if label else "annotation_negative"] += 1
                    else:
                        counts[exclusion] += 1
                    t += cadence
                    continue
                # Runtime updates its prior baseline at every quality-accepted cadence,
                # including annotation-excluded periods; labels cannot alter warm-up.
                signal = read_signal(record, t - history, t)
                quality_accepted = assess_quality(extract_features(signal, record.fs), config)["accepted"]
                available = quality_accepted and quality_updates >= warmup_updates
                if quality_accepted:
                    quality_updates += 1
                if label is None:
                    counts[exclusion] += 1
                elif not available:
                    counts["baseline_warmup" if quality_accepted else "signal_quality"] += 1
                else:
                    counts["positive" if label else "negative"] += 1
                    hours += cadence / 3600
                    if label:
                        accepted_onsets.update(
                            onset
                            for onset in annotated_onsets
                            if t + config["target"]["gap_seconds"]
                            <= onset
                            <= t + config["target"]["horizon_seconds"]
                        )
                t += cadence
            exclusions.update(
                {key: value for key, value in counts.items() if key not in {"positive", "negative"}}
            )
        if accepted_onsets:
            event_patients.add(record.patient_id)
        total_events += len(accepted_onsets)
        monitoring_seconds += hours * 3600
        cohort.append(
            {
                "record_id": record.record_id,
                "patient_id": record.patient_id,
                "duration_seconds": record.duration_seconds,
                "annotated": record.annotated,
                "signal_available": has_signal,
                "exclusion": reason,
                "annotation_candidate_events": len(annotated_onsets),
                "eligible_events": len(accepted_onsets),
                "eligible_monitoring_hours": hours,
                "window_counts": dict(counts),
                "eligible_onsets": sorted(accepted_onsets),
            }
        )
    split_path = output_dir / "splits.json"
    existing = json.loads(split_path.read_text()) if split_path.exists() else None
    previous_config = output_dir / "config.json"
    if existing and previous_config.exists():
        previous_seed = json.loads(previous_config.read_text()).get("seed", 42)
        if previous_seed != config.get("seed", 42):
            raise ValueError("Seed conflicts with frozen partitions; use a new audit directory")
    splits = _splits(records, event_patients, parent, config, existing)
    split_counts = {}
    for partition in ["train", "calibration", "validation", "test"]:
        patients = {p for p, name in splits.items() if name == partition}
        split_counts[partition] = {
            "subjects": len(patients),
            "event_subjects": len({root(p) for p in patients & event_patients}),
            "events": sum(r["eligible_events"] for r in cohort if r["patient_id"] in patients),
        }
    independent_event_groups = {root(patient) for patient in event_patients}
    feasible = (
        not annotation_only
        and len(independent_event_groups) >= config["audit"]["min_event_subjects"]
        and split_counts["test"]["event_subjects"] >= config["audit"]["min_test_event_subjects"]
        and all(split_counts[p]["event_subjects"] > 0 for p in ["train", "calibration", "validation"])
    )
    limitations = [
        "Provisional target, episode duration, washout and quality settings require clinical review.",
        "Available audit events require a labeled quality-accepted window after runtime baseline warm-up. Evaluation retains all annotation-eligible onsets, including unavailable events.",
        "No prospective, clinical-readiness or Indian-population performance established.",
    ]
    if synthetic:
        limitations.append(
            "All ECGs, clinical profiles and precursor patterns are manufactured; metrics test plumbing only."
        )
    else:
        sources = {getattr(record, "source", "shdb-af") for record in records}
        if "shdb-af" in sources:
            limitations.extend(
                [
                    "SHDB-AF signals use zero-phase filtering; causal features do not undo upstream future information.",
                    "SHDB-AF (N means other/unlabelled rhythm; default unknown prevents assuming unlabelled negative periods.",
                    "SHDB-AF baseline comorbidity flags are supplied at Holter; independently dated EHR validation is absent.",
                ]
            )
        limitations.append(
            "Annotation semantics and static-field timing are source-specific; no cross-source clinical equivalence is assumed."
        )
    missing_signals = sum(not r["signal_available"] for r in cohort)
    if missing_signals:
        limitations.append(
            f"Content duplicate checks are incomplete: {missing_signals} recordings lack ECG files."
        )
    if not feasible:
        limitations.append(
            "Real-data training must stop: independent eligible events/held-out event subjects are insufficient."
        )
    report = {
        "synthetic": synthetic,
        "annotation_only": annotation_only,
        "sources": sorted({getattr(r, "source", "shdb-af") for r in records}),
        "feasible": bool(feasible),
        "records": len(records),
        "subjects": len({r.patient_id for r in records}),
        "annotated_records": sum(r.annotated for r in records),
        "signal_available_records": sum(r["signal_available"] for r in cohort),
        "content_duplicates_complete": missing_signals == 0 and not annotation_only,
        "overlap_time_missing_records": sum(r.start_time is None for r in records),
        "sampling_rates_hz": sorted({r.fs for r in records}),
        "duration_seconds_range": [
            min(r.duration_seconds for r in records),
            max(r.duration_seconds for r in records),
        ],
        "independent_groups": len(set(map(root, parent))),
        "eligible_events": total_events,
        "event_subjects": len(independent_event_groups),
        "eligible_monitoring_hours": monitoring_seconds / 3600,
        "annotation_candidate_events": candidates,
        "sensitivity_eligible_events": sum(
            r["annotation_candidate_events"] for r in cohort if r["exclusion"] is None
        ),
        "annotation_candidate_subjects": len(candidate_subjects),
        "n_as_other_annotation_candidates": n_as_other_candidates
        if not synthetic and any(getattr(r, "source", "shdb-af") == "shdb-af" for r in records)
        else None,
        "n_as_other_candidate_subjects": len(alternate_event_subjects)
        if not synthetic and any(getattr(r, "source", "shdb-af") == "shdb-af" for r in records)
        else None,
        "annotation_rhythm_hours": {key: value / 3600 for key, value in rhythm_seconds.items()},
        "annotation_interval_counts": dict(interval_counts),
        "short_af_intervals": short_af_intervals,
        "n_rhythm_is_other": n_is_other,
        "duplicate_records": len(duplicate_ids),
        "overlapping_records": len(overlap_ids),
        "exclusions": dict(exclusions),
        "split_counts": split_counts,
        "target": config["target"],
        "limitations": limitations,
        "dataset_fingerprint": dataset_fingerprint(data_dir),
        "patient_group_ids": {patient: root(patient) for patient in parent},
        "included_record_ids": [r["record_id"] for r in cohort if r["exclusion"] is None],
    }
    for name, payload in [
        ("records.json", [asdict(r) for r in records]),
        ("field_audit.json", dict(field_counts)),
        ("audit.json", report),
        ("splits.json", splits),
        ("cohort.json", cohort),
        ("config.json", config),
    ]:
        (output_dir / name).write_text(json.dumps(payload, indent=2, allow_nan=False))
    columns = [
        "record_id",
        "patient_id",
        "duration_seconds",
        "annotated",
        "signal_available",
        "exclusion",
        "annotation_candidate_events",
        "eligible_events",
        "eligible_monitoring_hours",
    ]
    with (output_dir / "cohort.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cohort)
    return report
