#!/usr/bin/env python3
"""Reassess predeclared episode durations using immutable frozen forecasts only."""

import argparse
import gzip
import hashlib
import json
from copy import deepcopy
from pathlib import Path

from cardiotwin.data import file_hash, list_records, read_intervals
from cardiotwin.evaluation import evaluate_stream
from cardiotwin.labels import eligible_onsets, label_at
from cardiotwin.modeling import Predictor


def relabel_predictions(rows, records, config):
    """Copy labels only; availability, risk, chronology and audited groups stay fixed."""
    required = {row["record_id"] for row in rows}
    records = {record.record_id: record for record in records if record.record_id in required}
    if not required or set(records) != required:
        raise ValueError("Source record membership is missing frozen prediction record IDs")
    groups = {}
    for row in rows:
        patient, group = row["patient_id"], row["group_id"]
        if patient in groups and groups[patient] != group:
            raise ValueError("Frozen patient belongs to multiple independent groups")
        groups[patient] = group
        record = records[row["record_id"]]
        if patient != record.patient_id or row["duration_seconds"] != record.duration_seconds:
            raise ValueError("Source record patient/duration differs from frozen predictions")
    intervals = {
        name: read_intervals(record, config["audit"].get("n_rhythm_is_other", False))
        for name, record in records.items()
    }
    events = [
        dict(patient_id=record.patient_id, group_id=groups[record.patient_id], record_id=name, onset=onset)
        for name, record in records.items()
        for onset in eligible_onsets(intervals[name], record.duration_seconds, config)
    ]
    labeled = []
    for row in rows:
        record = records[row["record_id"]]
        label, reason = label_at(
            row["timestamp"],
            intervals[record.record_id],
            record.duration_seconds,
            config,
        )
        labeled.append(dict(row, label=label, label_reason=reason))
    return labeled, events


def analyze(artifact, predictions, data, evaluation, output):
    artifact, predictions, data, evaluation, output = map(
        Path,
        (artifact, predictions, data, evaluation, output),
    )
    if output.exists():
        raise ValueError("Endpoint sensitivity output already exists")
    predictor = Predictor.load(artifact, research=True)
    original = json.loads(evaluation.read_text())
    prediction_hash = file_hash(predictions)
    for field, expected in (
        ("model_version", predictor.version),
        ("model_sha256", predictor.manifest["model_sha256"]),
        ("predictions_sha256", prediction_hash),
    ):
        if original.get(field) != expected:
            raise ValueError(f"Original evaluation {field} differs from frozen input")
    if not predictor.manifest.get("frozen_utc") or not original.get("cache_input_sha256"):
        raise ValueError("Require a frozen model and original cache input identity")
    contract = predictor.manifest["cohort_contract"]
    contract_hash = hashlib.sha256(
        json.dumps(
            {key: value for key, value in contract.items() if key != "sha256"},
            sort_keys=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    if contract_hash != contract.get("sha256"):
        raise ValueError("Frozen cohort contract checksum mismatch")
    declaration_path = (
        Path(__file__).resolve().parents[1] / "reports/ml-investigation/endpoint-sensitivity-predeclared.json"
    )
    declaration = json.loads(declaration_path.read_text())
    config = predictor.config
    minimum = declaration["primary_minimum_episode_seconds"]
    if config["target"]["min_episode_seconds"] != minimum:
        raise ValueError("Frozen primary episode endpoint differs from predeclared definition")
    with gzip.open(predictions, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    record_ids = {row["record_id"] for row in rows}
    records = [record for record in list_records(data) if record.record_id in record_ids]
    annotations = {}
    for record in records:
        path = Path(record.path)
        annotation = path.with_suffix(".atr")
        if record.source == "iridia-af":
            annotation = path / f"{record.record_id}_ecg_labels.csv"
        elif path.suffix == ".npz":
            annotation = path.parent / "manifest.json"
        annotations[record.record_id] = file_hash(annotation) if annotation.is_file() else None
    if annotations != original.get("source_annotation_sha256"):
        raise ValueError("Source record annotation checksums differ from original frozen evaluation")
    primary_rows, _ = relabel_predictions(rows, records, config)
    if any(
        (row.get("label"), row.get("label_reason")) != (checked["label"], checked["label_reason"])
        for row, checked in zip(rows, primary_rows, strict=True)
    ):
        raise ValueError("Source annotation labels differ from original frozen predictions")
    del primary_rows
    analyses = {}
    policies = predictor.manifest["cohort_contract"].get("operating_policies", {})
    for endpoint in declaration["analyses"]:
        seconds = endpoint["minimum_episode_seconds"]
        if seconds == minimum:
            continue
        changed = deepcopy(config)
        changed["target"]["min_episode_seconds"] = seconds
        labeled, events = relabel_predictions(rows, records, changed)
        report = evaluate_stream(labeled, events, changed)
        frontier = {"0.25": dict(report)}
        for budget, alert in policies.items():
            if budget == "0.25":
                continue
            policy = deepcopy(changed)
            policy["alert"] = alert
            frontier[budget] = evaluate_stream(labeled, events, policy) if alert is not None else None
        report.update(minimum_episode_seconds=seconds, frontier=frontier)
        analyses[str(seconds)] = report
    result = {
        "rule": declaration["rule"],
        "primary_minimum_episode_seconds": minimum,
        "provenance": {
            "artifact": str(artifact.resolve()),
            "model_version": predictor.version,
            "model_sha256": predictor.manifest["model_sha256"],
            "predictions": str(predictions.resolve()),
            "predictions_sha256": prediction_hash,
            "original_evaluation": str(evaluation.resolve()),
            "original_evaluation_sha256": file_hash(evaluation),
            "cache_input_sha256": original["cache_input_sha256"],
            "cache_rows_sha256": original.get("cache_rows_sha256"),
            "source_annotation_sha256": annotations,
            "data": str(data.resolve()),
            "record_ids": sorted(record_ids),
            "patient_group_ids": {row["patient_id"]: row["group_id"] for row in rows},
            "cohort": original["cohort"],
            "independent_group_note": original.get("independent_group_note"),
            "predeclaration_sha256": file_hash(declaration_path),
        },
        "analyses": analyses,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("artifact", "predictions", "data", "evaluation", "output"):
        parser.add_argument(f"--{option}", required=True, type=Path)
    args = parser.parse_args()
    analyze(**vars(args))


if __name__ == "__main__":
    main()
