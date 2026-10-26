import gzip
import hashlib
import json
import runpy
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import joblib
import pytest

from cardiotwin.config import load_config
from cardiotwin.data import file_hash, list_records
from cardiotwin.labels import RhythmInterval, label_at


@pytest.fixture
def endpoint_inputs(tmp_path):
    data, artifact = tmp_path / "data", tmp_path / "artifact"
    data.mkdir()
    artifact.mkdir()
    config = load_config()
    config["alert"]["persistence_seconds"] = 0
    records, rows = [], []
    for name, length, group, accepted in [
        ("short", 45, "g0", True),
        ("medium", 120, "g1", True),
        ("long", 600, "g1", False),
    ]:
        intervals = [
            RhythmInterval(0, 1800, "other"),
            RhythmInterval(1800, 1800 + length, "AF"),
            RhythmInterval(1800 + length, 3600, "other"),
        ]
        records.append(
            dict(
                record_id=name,
                patient_id=name,
                path=f"{name}.npz",
                fs=200,
                duration_seconds=3600,
                annotated=True,
                clinical={},
                metadata_audit=[],
                intervals=[vars(interval) for interval in intervals],
            )
        )
        for t in range(0, 3600, 30):
            label, reason = label_at(t, intervals, 3600, config)
            rows.append(
                dict(
                    record_id=name,
                    patient_id=name,
                    group_id=group,
                    timestamp=t,
                    duration_seconds=3600,
                    accepted=accepted,
                    risk=(0.9 if 1200 <= t <= 1680 else 0.1) if accepted else None,
                    reason=None if accepted else "quality",
                    label=label,
                    label_reason=reason,
                )
            )
    # A source record without predictions must not add an event to the analysis.
    records.append(dict(records[0], record_id="unconsumed", patient_id="unconsumed"))
    (data / "manifest.json").write_text(json.dumps({"records": records}))
    contract = {"operating_policies": {"0.1": config["alert"], "0.25": config["alert"], "1": None}}
    contract["sha256"] = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    joblib.dump(
        dict(schema=[], config=config, cohort_contract=contract, pipeline=None, calibrator=None),
        artifact / "model.joblib",
    )
    model_hash = file_hash(artifact / "model.joblib")
    (artifact / "manifest.json").write_text(
        json.dumps(
            dict(
                version="frozen-fixture",
                frozen_utc="2026-10-08T12:00:00Z",
                model_sha256=model_hash,
                cohort_contract=contract,
            )
        )
    )
    (artifact / "config.json").write_text(json.dumps(config))
    (artifact / "feature_schema.json").write_text(json.dumps({"features": []}))
    predictions = tmp_path / "predictions.jsonl.gz"
    with gzip.open(predictions, "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(
        json.dumps(
            dict(
                model_version="frozen-fixture",
                model_sha256=model_hash,
                predictions_sha256=file_hash(predictions),
                cache_input_sha256="frozen-input",
                cache_rows_sha256="frozen-rows",
                source_annotation_sha256={
                    name: file_hash(data / "manifest.json") for name in ("short", "medium", "long")
                },
                cohort="primary",
            )
        )
    )
    return artifact, predictions, data, evaluation, tmp_path / "sensitivity.json"


def run_analysis(paths):
    script = Path(__file__).parents[1] / "scripts/evaluate_endpoint_sensitivity.py"
    arguments = [
        str(value)
        for pair in zip(
            ("--artifact", "--predictions", "--data", "--evaluation", "--output"),
            paths,
        )
        for value in pair
    ]
    return subprocess.run([sys.executable, str(script), *arguments], capture_output=True, text=True)


def test_endpoint_duration_changes_events_without_changing_frozen_warnings(endpoint_inputs):
    artifact, predictions, data, evaluation, output = endpoint_inputs
    original_hash = file_hash(predictions)
    result = run_analysis(endpoint_inputs)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    sixty, three_hundred = report["analyses"]["60"], report["analyses"]["300"]
    assert sixty["eligible_events"] == 2
    assert sixty["matched_events"] == 1
    assert three_hundred["eligible_events"] == 1
    assert three_hundred["matched_events"] == 0
    for analysis in (sixty, three_hundred):
        assert analysis["warnings"] == 2
        assert analysis["events_missed_during_abstention"] == 1
        assert analysis["confidence_intervals"]["independent_groups"] == 2
        assert analysis["confidence_intervals"]["samples"] == 300
        assert analysis["frontier"]["1"] is None
    assert sixty["known_excluded_warnings"] == 1
    assert three_hundred["known_excluded_warnings"] == 2
    assert sixty["warning_seconds"] == three_hundred["warning_seconds"]
    assert report["provenance"]["predictions_sha256"] == original_hash
    assert report["provenance"]["cache_input_sha256"] == "frozen-input"
    assert report["provenance"]["record_ids"] == ["long", "medium", "short"]
    assert file_hash(predictions) == original_hash
    assert run_analysis(endpoint_inputs).returncode != 0


@pytest.mark.parametrize("field", ["model_version", "model_sha256", "predictions_sha256"])
def test_endpoint_analysis_rejects_unbound_predictions(endpoint_inputs, field):
    evaluation, output = endpoint_inputs[3:]
    report = json.loads(evaluation.read_text())
    report[field] = "different"
    evaluation.write_text(json.dumps(report))
    result = run_analysis(endpoint_inputs)
    assert result.returncode != 0
    assert field in result.stderr
    assert not output.exists()


def test_endpoint_analysis_rejects_missing_record(endpoint_inputs):
    data, output = endpoint_inputs[2], endpoint_inputs[4]
    manifest = json.loads((data / "manifest.json").read_text())
    manifest["records"] = [record for record in manifest["records"] if record["record_id"] != "long"]
    (data / "manifest.json").write_text(json.dumps(manifest))
    result = run_analysis(endpoint_inputs)
    assert result.returncode != 0
    assert "record" in result.stderr
    assert not output.exists()


def test_relabeling_preserves_each_probability_and_availability(endpoint_inputs):
    script = Path(__file__).parents[1] / "scripts/evaluate_endpoint_sensitivity.py"
    relabel = runpy.run_path(str(script))["relabel_predictions"]
    artifact, predictions, data = endpoint_inputs[:3]
    with gzip.open(predictions, "rt") as handle:
        rows = [json.loads(line) for line in handle]
    unchanged = deepcopy(rows)
    config = json.loads((artifact / "config.json").read_text())
    config["target"]["min_episode_seconds"] = 300
    labeled, events = relabel(rows, list_records(data), config)
    assert [(event["record_id"], event["onset"], event["group_id"]) for event in events] == [
        ("long", 1800, "g1")
    ]
    at_1200 = {
        row["record_id"]: (row["label"], row["label_reason"]) for row in labeled if row["timestamp"] == 1200
    }
    assert at_1200 == {
        "short": (None, "ineligible_event"),
        "medium": (None, "ineligible_event"),
        "long": (1, None),
    }
    for original, changed in zip(rows, labeled, strict=True):
        assert {key: value for key, value in original.items() if key not in {"label", "label_reason"}} == {
            key: value for key, value in changed.items() if key not in {"label", "label_reason"}
        }
    assert rows == unchanged


def test_endpoint_analysis_rejects_changed_source_annotations(endpoint_inputs):
    data, output = endpoint_inputs[2], endpoint_inputs[4]
    path = data / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["records"][0]["intervals"] = [{"start": 0, "end": 3600, "rhythm": "other"}]
    path.write_text(json.dumps(manifest))
    result = run_analysis(endpoint_inputs)
    assert result.returncode != 0
    assert "annotation" in result.stderr
    assert not output.exists()
