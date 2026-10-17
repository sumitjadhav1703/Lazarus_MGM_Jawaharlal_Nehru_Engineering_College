import json
from copy import deepcopy

import pytest

from cardiotwin.evaluation import evaluate_artifact, evaluate_stream, probability_metrics
from cardiotwin.train import _candidate_rank, cohort_contract, validate_cohort_contract


def test_group_macro_brier_is_invariant_to_patient_aliases():
    rows = [
        {"patient_id": "a", "group_id": "linked", "label": 0, "risk": 0.1},
        {"patient_id": "alias", "group_id": "linked", "label": 0, "risk": 0.3},
        {"patient_id": "b", "group_id": "other", "label": 0, "risk": 0.9},
    ]
    original = probability_metrics(rows)
    canonical = probability_metrics([dict(r, patient_id=r["group_id"]) for r in rows])
    assert original["group_macro_brier"] == pytest.approx(0.43)
    assert original["group_macro_brier"] == canonical["group_macro_brier"]
    assert original["probability_groups"] == 2
    assert original["patient_macro_brier"] != canonical["patient_macro_brier"]


def test_contract_binds_membership_groups_and_inclusion_even_if_both_sidecars_changed():
    splits = {"a": "train", "b": "calibration", "c": "validation", "d": "test"}
    groups = {p: p for p in splits}
    contract = cohort_contract(splits, groups, ["r1", "r2"])
    manifest = {"cohort_contract": contract, "cohort_contract_sha256": contract["sha256"]}
    validate_cohort_contract(manifest, splits, groups, ["r2", "r1"])
    for field in ["splits", "groups", "records"]:
        changed_splits, changed_groups, changed_records = dict(splits), dict(groups), ["r1", "r2"]
        if field == "splits":
            changed_splits["a"], changed_splits["d"] = "test", "train"
        elif field == "groups":
            changed_groups["d"] = "new_identity"
        else:
            changed_records = ["r1"]
        with pytest.raises(ValueError, match="cohort contract"):
            validate_cohort_contract(manifest, changed_splits, changed_groups, changed_records)
    damaged = deepcopy(manifest)
    damaged["cohort_contract"]["splits"]["a"] = "test"
    with pytest.raises(ValueError, match="cohort contract"):
        validate_cohort_contract(damaged, splits, groups, ["r1", "r2"])


def test_existing_evaluation_directory_is_rejected_before_artifact_access(tmp_path):
    output = tmp_path / "result"
    output.mkdir()
    previous = output / "evaluation.json"
    previous.write_text('{"historic":true}')
    with pytest.raises(ValueError, match="fresh"):
        evaluate_artifact(tmp_path / "data", tmp_path / "audit", tmp_path / "model", output)
    assert json.loads(previous.read_text()) == {"historic": True}


def test_model_rank_penalizes_unknown_warning_burden_before_brier_and_complexity():
    base = {
        "event_sensitivity": 1.0,
        "false_warnings_per_hour": 0.0,
        "warnings": 3,
        "warning_hours": 0.1,
        "probability": {"group_macro_brier": 0.1},
    }
    simpler = {"model": "logistic_ecg", "feature_count": 6, "selected_policy": dict(base, warnings=5)}
    cleaner = {"model": "xgboost_ecg", "feature_count": 6, "selected_policy": base}
    assert _candidate_rank(cleaner) > _candidate_rank(simpler)
    longer = deepcopy(cleaner)
    longer["selected_policy"]["warning_hours"] = 1.0
    assert _candidate_rank(cleaner) > _candidate_rank(longer)


def test_known_excluded_warnings_and_unknown_outcomes_are_reported_separately():
    config = {
        "target": {"cadence_seconds": 30, "gap_seconds": 120, "horizon_seconds": 600},
        "alert": {
            "threshold": 0.8,
            "release_threshold": 0.6,
            "persistence_seconds": 0,
            "cooldown_seconds": 0,
        },
    }
    rows = [
        {
            "patient_id": "a",
            "record_id": reason,
            "timestamp": 0,
            "duration_seconds": 30,
            "accepted": True,
            "risk": 0.9,
            "label": None,
            "label_reason": reason,
        }
        for reason in ["current_AF", "unknown", "washout", "future"]
    ]
    report = evaluate_stream(rows, [], config, bootstrap_samples=0)
    assert report["false_warnings"] == 0
    assert report["unassessable_warnings"] == 4
    assert report["known_excluded_warnings"] == 2
    assert report["unknown_outcome_warnings"] == 2
    assert report["known_excluded_warning_reasons"] == {"current_AF": 1, "washout": 1}


def test_audit_uses_runtime_warmup_without_dropping_annotation_event_denominator(tmp_path, monkeypatch):
    import numpy as np

    from cardiotwin import audit
    from cardiotwin.data import RecordInfo
    from cardiotwin.labels import RhythmInterval

    path = tmp_path / "signal.npz"
    path.write_bytes(b"hashed fixture only; reading is mocked")
    record = RecordInfo("r", "p", str(path), 100, 100, True, {}, [])
    monkeypatch.setattr(audit, "list_records", lambda _: [record])
    monkeypatch.setattr(
        audit,
        "read_intervals",
        lambda *a, **k: [
            RhythmInterval(0, 22, "other"),
            RhythmInterval(22, 25, "AF"),
            RhythmInterval(25, 100, "other"),
        ],
    )
    monkeypatch.setattr(audit, "read_signal", lambda *a: np.zeros((2000, 2)))
    monkeypatch.setattr(audit, "extract_features", lambda *a: {})
    monkeypatch.setattr(audit, "assess_quality", lambda *a: {"accepted": True})
    config = {
        "seed": 42,
        "target": {
            "history_seconds": 20,
            "gap_seconds": 2,
            "horizon_seconds": 10,
            "cadence_seconds": 2,
            "min_episode_seconds": 3,
            "washout_seconds": 20,
        },
        "baseline": {"minimum_updates": 3},
        "audit": {"min_event_subjects": 4, "min_test_event_subjects": 1},
    }
    report = audit.audit_dataset(tmp_path, tmp_path / "audit", config)
    assert report["annotation_candidate_events"] == 1
    assert report["sensitivity_eligible_events"] == 1
    assert report["eligible_events"] == 0
    assert report["exclusions"]["baseline_warmup"] == 1
    # An external metadata audit cannot open any waveform or claim training feasibility.
    monkeypatch.setattr(audit, "read_signal", lambda *a: pytest.fail("external waveform opened"))
    report = audit.audit_dataset(tmp_path, tmp_path / "external", config, annotation_only=True)
    assert report["annotation_only"] is True
    assert report["content_duplicates_complete"] is False
    assert report["feasible"] is False


def test_evaluation_records_each_test_access_without_overwriting_history(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from cardiotwin import data, train
    from cardiotwin.modeling import Predictor

    splits = {"a": "train", "b": "calibration", "c": "validation", "d": "test"}
    groups = {p: p for p in splits}
    contract = cohort_contract(splits, groups, ["rd"])
    manifest = {
        "frozen": True,
        "synthetic": False,
        "dataset_fingerprint": "data-sha",
        "included_record_ids": ["rd"],
        "patient_group_ids": groups,
        "cohort_contract": contract,
        "cohort_contract_sha256": contract["sha256"],
        "model_sha256": "model-sha",
        "limitations": [],
    }
    config = {
        "seed": 42,
        "target": {
            "history_seconds": 600,
            "cadence_seconds": 30,
            "gap_seconds": 120,
            "horizon_seconds": 600,
            "min_episode_seconds": 30,
            "washout_seconds": 600,
        },
        "alert": {
            "threshold": 0.8,
            "release_threshold": 0.6,
            "persistence_seconds": 0,
            "cooldown_seconds": 0,
            "max_false_warnings_per_hour": 0.25,
            "min_event_sensitivity": 0.5,
        },
        "audit": {},
    }
    predictor = SimpleNamespace(manifest=manifest, config=config, version="model-version")
    monkeypatch.setattr(Predictor, "load", lambda _: predictor)
    records = [data.RecordInfo("r" + p, p, "unused", 100, 30, True, {}, []) for p in splits]
    monkeypatch.setattr(data, "list_records", lambda _: records)
    monkeypatch.setattr(data, "dataset_fingerprint", lambda _: "data-sha")
    monkeypatch.setattr(data, "read_intervals", lambda *a, **k: [])
    monkeypatch.setattr(
        train,
        "iter_record_rows",
        lambda record, config: iter(
            [
                {
                    "patient_id": record.patient_id,
                    "record_id": record.record_id,
                    "timestamp": 0,
                    "duration_seconds": 30,
                    "accepted": False,
                    "reason": "quality",
                    "features": {},
                    "label": 0,
                }
            ]
        ),
    )
    artifact, audit = tmp_path / "artifact", tmp_path / "audit"
    artifact.mkdir()
    audit.mkdir()
    for path in [artifact, audit]:
        (path / "splits.json").write_text(json.dumps(splits))
    (audit / "audit.json").write_text(json.dumps(manifest))
    first = evaluate_artifact(tmp_path, audit, artifact, tmp_path / "result-one")
    second = evaluate_artifact(tmp_path, audit, artifact, tmp_path / "result-two")
    assert first["cohort_integrity"] == "bound"
    assert first["partition"] == "held_out_test"
    assert first["test_consumption"]["previously_observed"] is False
    assert second["test_consumption"]["previously_observed"] is True
    assert second["test_consumption"]["previous_evaluation_ids"] == [
        first["test_consumption"]["evaluation_id"]
    ]
    entries = [json.loads(p.read_text()) for p in (audit / "test-consumption").glob("*.json")]
    assert len(entries) == 4
    assert sorted(e["status"] for e in entries) == ["completed", "completed", "started", "started"]
    changed = dict(splits, a="test", d="train")
    for path in [artifact, audit]:
        (path / "splits.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="cohort contract"):
        evaluate_artifact(tmp_path, audit, artifact, tmp_path / "result-three")
    assert len(list((audit / "test-consumption").glob("*.json"))) == 4
