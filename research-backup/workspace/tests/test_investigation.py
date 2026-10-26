import json
from types import SimpleNamespace

import numpy as np
import pytest

from cardiotwin.config import load_config
from cardiotwin.investigation import (
    _digest,
    _partitions,
    feature_groups,
    make_primary_split,
    paired_group_comparison,
    read_cache,
    rr_motifs,
    select_policy,
)


def _rows(groups=24):
    return [
        {
            "patient_id": f"p{i}",
            "group_id": f"g{i}",
            "record_id": f"r{i}",
            "accepted": True,
            "label": label,
            "features": {"ecg_hr": 70 + label * 10},
        }
        for i in range(groups)
        for label in (0, 1)
    ]


def test_split_keeps_aliases_together_and_is_seeded():
    records = [SimpleNamespace(patient_id=f"p{i}", record_id=f"r{i}") for i in range(25)]
    groups = {r.patient_id: f"g{min(i, 23)}" for i, r in enumerate(records)}
    events = [{"patient_id": f"p{i}"} for i in range(18)]
    split = make_primary_split(records, events, groups)
    assert split == make_primary_split(records, events, groups)
    assert not set(split["test_groups"]) & set(split["development_groups"])
    assert split["sha256"] == _digest({k: v for k, v in split.items() if k != "sha256"})
    assert groups["p23"] == groups["p24"]


def test_nested_roles_are_group_disjoint():
    partitions = _partitions(_rows(), 42)
    assert all(not set(a) & set(b) for i, a in enumerate(partitions) for b in partitions[i + 1 :])
    assert set().union(*map(set, partitions)) == {f"g{i}" for i in range(24)}


def test_xgboost_core_tunes_tree_parameters_despite_linear_search_flag():
    import time

    from cardiotwin.investigation import _tune

    trials = []
    selected = _tune(
        _rows(12),
        "xgboost",
        ["ecg_hr"],
        42,
        3,
        trials.append,
        time.monotonic() + 60,
        preprocessing={"_current_search": True},
    )
    assert len(trials) == 3
    assert len({tuple(sorted(t["parameters"].items())) for t in trials}) == 3
    assert all("max_depth" in t["parameters"] and "C" not in t["parameters"] for t in trials)
    assert "n_estimators" in selected


def test_feature_allowlist_and_availability_control():
    rows = [{"features": {"ecg_hr": 70, "clinical_age": 45, "clinical_age_available": 1, "future_AF": 1}}]
    schemas = feature_groups(rows)
    assert "future_AF" not in set().union(*map(set, schemas.values()))
    assert "clinical_age" not in schemas["availability_control"]
    assert "clinical_age_available" in schemas["availability_control"]


def test_personalization_ablation_changes_only_prior_ecg_features():
    schemas = feature_groups(
        [
            {
                "features": {
                    "ecg_hr": 70,
                    "baseline_ecg_hr_deviation": 2,
                    "clinical_age": 45,
                    "clinical_age_available": 1,
                }
            }
        ]
    )
    assert schemas["personal"] == ["baseline_ecg_hr_deviation", "ecg_hr"]
    assert "clinical_age" in schemas["combined"]


def test_cache_rejects_content_tampering(tmp_path):
    (tmp_path / "rows.jsonl").write_text("{}\n")
    (tmp_path / "index.json").write_text(
        json.dumps(
            {
                "cache_version": "continuous-investigation-v1",
                "feature_version": "causal-rr-v2",
                "rows_sha256": "invalid",
                "config": {},
                "config_sha256": _digest({}),
            }
        )
    )
    with pytest.raises(ValueError, match="checksum"):
        read_cache(tmp_path)


def test_policy_counts_full_warning_burden():
    config = load_config()
    rows = [
        dict(
            patient_id="p",
            group_id="g",
            record_id="r",
            timestamp=float(t),
            duration_seconds=2400,
            accepted=True,
            label=0,
            risk=0.1,
            reason=None,
            label_reason=None,
        )
        for t in range(0, 2401, 30)
    ]
    rows[30]["risk"] = 0.9
    events = [{"patient_id": "p", "record_id": "r", "onset": 1500}]
    policy, searches = select_policy(rows, events, config)
    assert policy is not None
    assert all("warning_hours" in search and "unassessable_warnings" in search for search in searches)
    assert policy["alert"]["max_false_warnings_per_hour"] == 0.25


def test_policy_search_respects_compute_deadline():
    import time

    with pytest.raises(TimeoutError, match="budget"):
        select_policy([], [], load_config(), deadline=time.monotonic() - 1)


def test_pair_bootstrap_uses_group_not_aliases():
    report = {
        "per_patient": [
            dict(patient_id=patient, eligible_events=1, matched_events=1, false_warnings=0, eligible_hours=1)
            for patient in ("a", "alias-a", "b")
        ]
    }
    result = paired_group_comparison(report, report, {"a": "g", "alias-a": "g", "b": "h"})
    assert result["groups"] == 2
    assert result["ci95"] == [0, 0]


def test_selection_checks_supported_challengers_below_uncertain_top_rank():
    from cardiotwin.investigation import select_reports

    groups = {f"p{i}": f"g{i}" for i in range(3)}

    def report(matched):
        return {
            "complete_outer_folds": 3,
            "event_sensitivity": sum(matched) / 30,
            "false_warnings_per_hour": 0,
            "warnings": sum(matched),
            "warning_hours": 0.1,
            "probability": {"group_macro_brier": 0.1, "patient_macro_brier": 0.1},
            "per_patient": [
                dict(
                    patient_id=f"p{i}",
                    eligible_events=10,
                    matched_events=n,
                    false_warnings=0,
                    eligible_hours=1,
                    warnings=n,
                    warning_hours=0.1,
                )
                for i, n in enumerate(matched)
            ],
        }

    combined = {
        "logistic_current_recipe": report([5, 5, 5]),
        "uncertain": report([10, 10, 1]),
        "supported": report([6, 6, 6]),
        "incomplete": dict(report([10, 10, 10]), complete_outer_folds=2),
    }
    selection = select_reports(combined, groups)
    assert selection["selected_candidate"] == "supported"
    assert selection["ranking"][0] == "uncertain"
    assert "incomplete" not in selection["ranking"]

    baseline, worse_burden = report([5, 5, 5]), report([5, 5, 5])
    for item in (baseline, worse_burden):
        for patient in item["per_patient"]:
            patient["eligible_hours"] = 10
    baseline["per_group_brier"] = {group: 0.1 for group in groups.values()}
    worse_burden.update(
        false_warnings_per_hour=0.1,
        warnings=18,
        probability={"group_macro_brier": 0.01, "patient_macro_brier": 0.01},
        per_group_brier={group: 0.01 for group in groups.values()},
    )
    for patient in worse_burden["per_patient"]:
        patient.update(false_warnings=1, warnings=6)
    assert (
        select_reports({"logistic_current_recipe": baseline, "worse_burden": worse_burden}, groups)[
            "selected_candidate"
        ]
        == "logistic_current_recipe"
    )


def test_ranking_does_not_treat_missing_brier_as_perfect():
    from cardiotwin.investigation import _rank

    report = dict(event_sensitivity=0.5, false_warnings_per_hour=0.1, warnings=5, warning_hours=0.1)
    measured = dict(report, probability={"group_macro_brier": 0.1})
    missing = dict(report, probability={"group_macro_brier": None})
    assert _rank(measured) > _rank(missing)


def test_motifs_constant_rhythm_and_missing_gap(monkeypatch):
    monkeypatch.setattr("cardiotwin.investigation._peaks", lambda signal, fs: np.arange(1, 30, 0.8))
    signal = np.ones((3100, 2))
    clean = rr_motifs(signal, 100)
    assert clean["ecg_short_long_fraction"] == 0
    signal[500:2700] = np.nan
    assert all(value is None for value in rr_motifs(signal, 100).values())


def test_compact_cache_preserves_none_groups_and_exact_values(tmp_path):
    from cardiotwin.data import file_hash
    from cardiotwin.investigation import CACHE_VERSION

    rows = [
        dict(
            patient_id="p",
            group_id="g",
            record_id="r",
            timestamp=float(i),
            duration_seconds=120,
            accepted=bool(i),
            label=1 if i else None,
            reason=None if i else "history",
            label_reason=None if i else "history",
            features={"ecg_hr": 70.125 if i else None},
        )
        for i in range(2)
    ]
    (tmp_path / "rows.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    config = load_config()
    index = {
        "cache_version": CACHE_VERSION,
        "feature_version": config["feature_version"],
        "config": config,
        "config_sha256": _digest(config),
        "group_ids": {"p": "g"},
        "allowed_groups": ["g"],
        "events": [],
        "row_count": 2,
        "rows_sha256": file_hash(tmp_path / "rows.jsonl"),
        "feature_fields": ["ecg_hr"],
    }
    index["sha256"] = _digest(index)
    (tmp_path / "index.json").write_text(json.dumps(index))
    cached, _, _ = read_cache(tmp_path)
    assert dict(cached[0])["label"] is None
    assert cached[1]["features"]["ecg_hr"] == 70.125
    assert cached[1]["group_id"] == "g"
    assert (tmp_path / "features.npy").exists()
    cached_again, _, _ = read_cache(tmp_path)
    assert dict(cached_again[1]["features"]) == {"ecg_hr": 70.125}


@pytest.mark.parametrize("fail_artifact", [False, True])
def test_nested_ladder_keeps_wholly_abstaining_group(tmp_path, monkeypatch, fail_artifact):
    import cardiotwin.investigation as investigation
    from cardiotwin.investigation import run_ladder

    write_predictions = investigation._write_predictions

    def write_checked(path, predictions):
        if fail_artifact and path.name == "fold-0-logistic_core-predictions.jsonl.gz":
            raise ValueError("injected artifact failure")
        return write_predictions(path, predictions)

    monkeypatch.setattr(investigation, "_write_predictions", write_checked)
    rng = np.random.default_rng(42)
    rows, events = [], []
    for group in range(25):
        for t in range(0, 2401, 30):
            label = int(1200 <= t <= 1680) if t <= 1800 else None
            rows.append(
                dict(
                    patient_id=f"p{group}",
                    group_id=f"g{group}",
                    record_id=f"r{group}",
                    timestamp=float(t),
                    duration_seconds=2400,
                    accepted=group != 24 and t >= 600,
                    label=label,
                    reason=None if group != 24 else "corrupt",
                    label_reason=None if label is not None else "future",
                    features={"ecg_hr": float(70 + 15 * (label or 0) + rng.normal(0, 0.5))},
                )
            )
        events.append(dict(patient_id=f"p{group}", group_id=f"g{group}", record_id=f"r{group}", onset=1800))
    config = load_config()
    config["training"]["calibration_c"] = 100
    result = run_ladder(
        rows, events, config, tmp_path, trials=1, max_hours=0.02, specs=[("logistic", "core")]
    )
    assert result["status"] == "complete"
    assert result["combined"]["no_warning"]["eligible_events"] == 25
    if fail_artifact:
        assert "logistic_core" not in result["combined"]
        assert result["selected_candidate"] is None
        assert any(item["reason"] == "injected artifact failure" for item in result["failed_experiments"])
        return
    assert len(result["combined"]["logistic_core"]["per_patient"]) == 25
    assert result["combined"]["logistic_core"]["eligible_events"] == 25
    assert result["combined"]["logistic_core"]["matched_events"] < 25
    pooled = result["combined"]["logistic_core"]
    assert pooled["forecast_coverage"] == pytest.approx(
        pooled["available_seconds"] / pooled["monitored_seconds"]
    )
    assert (
        pooled["known_excluded_warnings"] + pooled["unknown_outcome_warnings"]
        == pooled["unassessable_warnings"]
    )
    assert (tmp_path / "experiments.jsonl").exists()


def test_frozen_research_roundtrip_rejects_split_source_and_reuse(tmp_path):
    from cardiotwin.investigation import evaluate_frozen, freeze_finalist
    from cardiotwin.modeling import Predictor

    records = [SimpleNamespace(patient_id=f"p{i}", record_id=f"r{i}") for i in range(24)]
    group_ids = {f"p{i}": f"g{i}" for i in range(24)}
    events = [dict(patient_id=f"p{i}", group_id=f"g{i}", record_id=f"r{i}", onset=1800) for i in range(24)]
    split = make_primary_split(records, events, group_ids)
    rows = [
        dict(
            patient_id=f"p{i}",
            group_id=f"g{i}",
            record_id=f"r{i}",
            timestamp=float(t),
            duration_seconds=2400,
            accepted=t >= 600,
            label=int(1200 <= t <= 1680) if t <= 1800 else None,
            reason=None,
            label_reason=None,
            features={"ecg_hr": 90 if 1200 <= t <= 1680 else 70},
        )
        for i in range(24)
        for t in range(0, 2401, 30)
    ]
    development = [row for row in rows if row["group_id"] in split["development_groups"]]
    test = [row for row in rows if row["group_id"] in split["test_groups"]]
    development_events = [event for event in events if event["group_id"] in split["development_groups"]]
    test_events = [event for event in events if event["group_id"] in split["test_groups"]]
    config = load_config()
    config["training"]["calibration_c"] = 100
    artifact = tmp_path / "frozen"
    manifest = freeze_finalist(
        development, development_events, config, split, "logistic_core", artifact, trials=1
    )
    policies = manifest["cohort_contract"]["operating_policies"]
    assert set(policies) == {"0.1", "0.25", "1"}
    assert (
        policies["0.25"]["threshold"]
        == json.loads((artifact / "config.json").read_text())["alert"]["threshold"]
    )
    loaded = Predictor.load(artifact, research=True)
    assert loaded.schema == ["ecg_hr"]
    assert loaded.probabilities([{"ecg_hr": 90}])[0] > loaded.probabilities([{"ecg_hr": 70}])[0]
    identities = [
        {"record": {"record_id": name}, "input_contract": identity}
        for name, identity in split["record_contracts"].items()
        if group_ids[identity["patient_id"]] in split["test_groups"]
    ]
    index = {
        "records": identities,
        "config": config,
        "input_sha256": "test-fixture-identity",
        "rows_sha256": "fixture",
    }
    original_manifest_text = (artifact / "manifest.json").read_text()
    independently_loaded = json.loads(original_manifest_text)
    independently_loaded["primary_split"]["group_ids"]["p0"] = "tampered-identity"
    independently_loaded["primary_split"]["sha256"] = _digest(
        {key: value for key, value in independently_loaded["primary_split"].items() if key != "sha256"}
    )
    (artifact / "manifest.json").write_text(json.dumps(independently_loaded))
    with pytest.raises(ValueError, match="primary split"):
        evaluate_frozen(artifact, test, test_events, tmp_path / "independent-sidecar-tamper", index=index)
    (artifact / "manifest.json").write_text(original_manifest_text)
    tampered = dict(index, records=[dict(identities[0], input_contract={})] + identities[1:])
    with pytest.raises(ValueError, match="source content"):
        evaluate_frozen(artifact, test, test_events, tmp_path / "bad", index=tampered)
    report = evaluate_frozen(artifact, test, test_events, tmp_path / "evaluation", index=index)
    assert report["eligible_events"] == len(test_events)
    assert set(report["frontier"]) == {"0.1", "0.25", "1"}
    assert report["frontier"]["0.25"]["matched_events"] == report["matched_events"]
    with pytest.raises(ValueError, match="already consumed"):
        evaluate_frozen(artifact, test, test_events, tmp_path / "repeat", index=index)
    manifest["primary_split"]["group_ids"]["p0"] = "unrelated"
    (artifact / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="cohort contract"):
        Predictor.load(artifact)


@pytest.mark.parametrize("development_gate", [None, False, True])
@pytest.mark.parametrize("candidate", ["logistic_core", "blend_core_xgboost"])
def test_strong_final_policy_requires_verified_development_gate(
    tmp_path, monkeypatch, development_gate, candidate
):
    from cardiotwin import investigation
    from cardiotwin.modeling import Predictor

    rows = [dict(row, timestamp=1200.0 + 120 * row["label"]) for row in _rows()]
    split = {
        "development_groups": sorted({row["group_id"] for row in rows}),
        "test_groups": ["unseen"],
    }
    split["sha256"] = _digest(split)
    config = load_config()
    summary = None
    if development_gate is not None:
        summary = {
            "selected_candidate": candidate,
            "seed": 42,
            "extraction_contract": investigation._extraction_contract(config),
            "development_cache_contract": None,
            "fitting_cadence_seconds": 120,
            "tuning_trials": 1,
            "acceptance_gate_passed": development_gate,
        }
        summary["sha256"] = _digest(summary)
    monkeypatch.setattr(investigation, "_tune", lambda *args, **kwargs: {"C": 0.1})
    monkeypatch.setattr(
        investigation,
        "_fit",
        lambda rows, family, schema, parameters, seed, weighting: investigation._fit_model(
            rows, "logistic", "ecg", seed, weighting
        )[0],
    )
    monkeypatch.setattr(investigation, "predict_rows", lambda predictor, rows: [])
    monkeypatch.setattr(investigation, "_frozen_policies", lambda *args: {})
    monkeypatch.setattr(
        investigation,
        "select_policy",
        lambda *args, **kwargs: (config, [{"event_sensitivity": 1.0, "meets_constraint": True}]),
    )
    artifact = tmp_path / candidate
    manifest = investigation.freeze_finalist(
        rows, [], config, split, candidate, artifact, trials=1, selection_summary=summary
    )
    assert manifest["deployment_ready"] is (development_gate is True)
    assert manifest["development_acceptance_gate"] is (development_gate is True)
    assert manifest["final_policy_acceptance_gate"] is True
    assert manifest["acceptance_gate"] is (development_gate is True)
    if development_gate is not True:
        with pytest.raises(ValueError, match="research-only"):
            Predictor.load(artifact)
    assert Predictor.load(artifact, research=True).manifest == manifest
    if summary is not None:
        summary["acceptance_gate_passed"] = not development_gate
        with pytest.raises(ValueError, match="selection summary checksum"):
            investigation.freeze_finalist(
                rows, [], config, split, candidate, tmp_path / "tampered", trials=1, selection_summary=summary
            )


def test_iridia_probe_cannot_become_primary_training(tmp_path):
    from cardiotwin.investigation import require_complete_acquisition

    record = SimpleNamespace(source="iridia-af", path=str(tmp_path / "records" / "record_001"))
    with pytest.raises(ValueError, match="complete acquisition"):
        require_complete_acquisition([record])
    (tmp_path / "source.json").write_text(
        json.dumps({"status": "metadata_only", "archive_md5_verified": False})
    )
    with pytest.raises(ValueError, match="Incomplete"):
        require_complete_acquisition([record])


def test_record_contract_binds_iridia_reviewed_labels(tmp_path):
    from cardiotwin.data import RecordInfo
    from cardiotwin.investigation import _record_contract

    folder = tmp_path / "record_001"
    folder.mkdir()
    signal = folder / "signal.h5"
    signal.write_bytes(b"example physical signal")
    labels = folder / "record_001_ecg_labels.csv"
    labels.write_text("start,end,label\n1,2,AF\n")
    record = RecordInfo(
        "record_001",
        "patient_001",
        str(folder),
        200,
        100,
        True,
        {},
        [],
        source="iridia-af",
        signal_parts=[{"path": str(signal), "samples": 20000, "key": "ecg"}],
    )
    before = _record_contract(record)
    labels.write_text("start,end,label\n1,3,AF\n")
    after = _record_contract(record)
    assert before["signal_sha256"] == after["signal_sha256"]
    assert before["annotation_sha256"] != after["annotation_sha256"]


def test_parallel_record_cache_matches_serial_and_isolates_baselines(tmp_path):
    from cardiotwin.data import RecordInfo, file_hash
    from cardiotwin.investigation import cache_records

    fs, duration = 100, 780
    signal = np.zeros((fs * duration, 2), dtype=np.float32)
    offsets = np.arange(-8, 9)
    for second in np.arange(0.5, duration - 0.1, 0.8):
        center = int(second * fs)
        signal[center + offsets, 0] = np.exp(-0.5 * (offsets / 1.6) ** 2)
        signal[center + offsets, 1] = signal[center + offsets, 0] * 0.8
    records, manifest = [], {"records": []}
    for number in range(2):
        path = tmp_path / f"r{number}.npz"
        np.savez_compressed(path, signal=signal)
        record = RecordInfo(
            f"r{number}", f"p{number}", str(path), fs, duration, True, {}, [], signal_hash=file_hash(path)
        )
        records.append(record)
        manifest["records"].append(
            dict(record_id=record.record_id, intervals=[dict(start=0, end=duration, rhythm="other")])
        )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    groups = {"p0": "g0", "p1": "g1"}
    first = cache_records(
        records, load_config(), groups, tmp_path / "serial", set(groups.values()), workers=1
    )
    second = cache_records(
        records, load_config(), groups, tmp_path / "parallel", set(groups.values()), workers=2
    )
    assert first["rows_sha256"] == second["rows_sha256"]
    assert first["row_count"] == second["row_count"]
    rows, _, index = read_cache(tmp_path / "parallel")
    for patient in groups:
        first_history = next(row for row in rows if row["patient_id"] == patient and row["timestamp"] == 600)
        assert first_history["features"]["baseline_count"] == 0
    index["events"].append(dict(patient_id="p0", record_id="r0", onset=700))
    (tmp_path / "parallel" / "index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="index metadata"):
        read_cache(tmp_path / "parallel")


def test_preinspected_development_group_keeps_all_patient_aliases():
    records = [SimpleNamespace(patient_id=f"p{i}", record_id=f"r{i}") for i in range(25)]
    groups = {r.patient_id: f"g{min(i, 23)}" for i, r in enumerate(records)}
    events = [{"patient_id": f"p{i}"} for i in range(18)]
    split = make_primary_split(records, events, groups, predeclared_development_groups={"g23", "g0"})
    assert {"g23", "g0"} <= set(split["development_groups"])
    assert not {"g23", "g0"} & set(split["test_groups"])
    assert groups["p23"] == groups["p24"] == "g23"
    assert split["predeclared_development_groups"] == ["g0", "g23"]


def test_cache_resume_verifies_completed_records_and_recipe(tmp_path, monkeypatch):
    from cardiotwin import train
    from cardiotwin.data import RecordInfo, file_hash
    from cardiotwin.investigation import cache_records

    records = []
    for number in range(2):
        path = tmp_path / f"r{number}.npz"
        np.savez(path, signal=np.zeros((70000, 2)))
        records.append(
            RecordInfo(
                f"r{number}", f"p{number}", str(path), 100, 700, False, {}, [], signal_hash=file_hash(path)
            )
        )
    (tmp_path / "manifest.json").write_text(json.dumps({"records": []}))
    groups = {"p0": "g0", "p1": "g1"}
    config = load_config()
    original = train.iter_record_rows

    def interrupted(record, settings, **kwargs):
        if record.record_id == "r1":
            raise RuntimeError("deliberate extraction interruption")
        yield from original(record, settings, **kwargs)

    monkeypatch.setattr(train, "iter_record_rows", interrupted)
    output = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="deliberate"):
        cache_records(records, config, groups, output, groups.values(), workers=1)
    assert not (output / "index.json").exists()
    changed = dict(config, seed=123)
    with pytest.raises(ValueError, match="recipe"):
        cache_records(records, changed, groups, output, groups.values(), workers=1, resume=True)
    part = output / "record-0000.jsonl"
    saved = part.read_bytes()
    part.write_bytes(saved + b"{}\n")
    with pytest.raises(ValueError, match="rows checksum"):
        cache_records(records, config, groups, output, groups.values(), workers=1, resume=True)
    part.write_bytes(saved)

    def remaining(record, settings, **kwargs):
        assert record.record_id != "r0", "Completed records must not be extracted again"
        yield from original(record, settings, **kwargs)

    monkeypatch.setattr(train, "iter_record_rows", remaining)
    result = cache_records(records, config, groups, output, groups.values(), workers=1, resume=True)
    assert len(result["records"]) == 2
    assert read_cache(output)[2]["row_count"] == result["row_count"]
    with pytest.raises(ValueError, match="completed"):
        cache_records(records, config, groups, output, groups.values(), workers=1, resume=True)


def test_precomputed_raw_matrix_keeps_fold_fitting_identical():
    from cardiotwin.investigation import _fit
    from cardiotwin.modeling import matrix

    rows = _rows(12)
    rows[0]["features"]["ecg_hr"] = None
    schema = ["ecg_hr"]
    values = matrix([row["features"] for row in rows], schema)
    normal = _fit(rows, "logistic", schema, {"C": 0.1}, 42)
    cached = _fit(rows, "logistic", schema, {"C": 0.1}, 42, precomputed_x=values)
    np.testing.assert_allclose(normal.predict_proba(values), cached.predict_proba(values), atol=0, rtol=0)
    np.testing.assert_equal(normal["impute"].statistics_, cached["impute"].statistics_)
    np.testing.assert_equal(normal["scale"].mean_, cached["scale"].mean_)
    with pytest.raises(ValueError, match="Precomputed matrix"):
        _fit(rows, "logistic", schema, {"C": 0.1}, 42, precomputed_x=values[:-1])


def test_live_sequence_inputs_bind_actual_source_and_membership(tmp_path, capsys):
    import json

    from cardiotwin.data import RecordInfo
    from cardiotwin.investigation import _record_contract, validate_live_inputs

    signal = tmp_path / "signal.npz"
    np.savez_compressed(signal, signal=np.ones((1000, 2)))
    (tmp_path / "manifest.json").write_text('{"records": []}')
    record = RecordInfo("r", "p", str(signal), 100, 10, True, {}, [])
    identity = _record_contract(record)
    index = {
        "records": [{"record": {"record_id": "r"}, "input_contract": identity}],
        "group_ids": {"p": "g"},
        "allowed_groups": ["g"],
    }
    assert validate_live_inputs([record], index) == _digest([identity])
    progress = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert progress[0]["stage"] == "source_hash_validation"
    assert progress[-1]["status"] == "done"
    assert progress[-1]["completed"] == progress[-1]["total"] == 1
    with pytest.raises(ValueError, match="membership"):
        validate_live_inputs([], index)
    np.savez_compressed(signal, signal=np.zeros((1000, 2)))
    with pytest.raises(ValueError, match="Live source"):
        validate_live_inputs([record], index)
    progress = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert progress[-1]["status"] == "failed"
    assert progress[-1]["error_type"] == "ValueError"


def test_policy_quantiles_use_only_accepted_labeled_policy_rows(monkeypatch):
    from cardiotwin import investigation

    seen = []

    def evaluate(rows, events, config, **kwargs):
        seen.append(config["alert"]["threshold"])
        return {
            "false_warnings_per_hour": 0,
            "event_sensitivity": 0.5,
            "warnings": 1,
            "unassessable_warnings": 0,
            "warning_hours": 0.1,
            "eligible_events": 2,
        }

    monkeypatch.setattr(investigation, "evaluate_stream", evaluate)
    rows = [
        dict(accepted=True, label=label, risk=risk) for label, risk in [(0, 0.003), (1, 0.004), (0, 0.005)]
    ]
    rows += [dict(accepted=True, label=None, risk=0.987), dict(accepted=False, label=1, risk=0.9999)]
    _, search = select_policy(rows, [], load_config())
    assert 0.004 in seen
    assert 0.987 not in seen and 0.9999 not in seen
    assert all(item["policy_eligible_windows"] == 3 for item in search)
    count = len(seen)
    select_policy(rows, [], load_config(), budget=0.1, search=search)
    assert len(seen) == count


def test_research_prediction_compression_preserves_values(tmp_path):
    import gzip

    from cardiotwin.investigation import _write_predictions

    rows = [dict(patient_id="p0", timestamp=600, label=0, risk=0.001, features={"ecg_hr": 60})]
    path = tmp_path / "predictions.jsonl.gz"
    assert _write_predictions(path, rows) == str(path)
    with gzip.open(path, "rt") as handle:
        assert json.loads(handle.readline()) == {
            key: value for key, value in rows[0].items() if key != "features"
        }


def test_research_baseline_and_final_tuning_lineage_guards(tmp_path):
    from cardiotwin.investigation import _extraction_contract, freeze_finalist

    config = load_config()
    with pytest.raises(ValueError, match="exact current recipe"):
        freeze_finalist([], [], config, {}, "logistic_core", tmp_path / "wrong", research_baseline=True)
    summary = {
        "selected_candidate": None,
        "seed": 42,
        "extraction_contract": _extraction_contract(config),
        "development_cache_contract": None,
        "fitting_cadence_seconds": 120,
        "tuning_trials": 2,
    }
    summary["sha256"] = _digest(summary)
    with pytest.raises(ValueError, match="tuning budget"):
        freeze_finalist(
            [],
            [],
            config,
            {},
            "logistic_current_recipe",
            tmp_path / "retuned",
            selection_summary=summary,
            research_baseline=True,
            trials=6,
        )
    summary["selected_candidate"] = "logistic_current_recipe"
    summary["sha256"] = _digest({key: value for key, value in summary.items() if key != "sha256"})
    with pytest.raises(ValueError, match="no qualifying winner"):
        freeze_finalist(
            [],
            [],
            config,
            {},
            "logistic_current_recipe",
            tmp_path / "winner-disguised",
            selection_summary=summary,
            research_baseline=True,
            trials=2,
        )


def test_development_audit_keeps_unavailable_events_and_exposure(tmp_path):
    from cardiotwin.data import file_hash
    from cardiotwin.investigation import _compact_cache, audit_development

    config = load_config()
    split = {"development_groups": ["g0", "g1", "g2"], "test_groups": ["heldout"]}
    split["sha256"] = _digest(split)
    rows = []
    for group in range(3):
        for t in (0, 30):
            rows.append(
                dict(
                    patient_id=f"p{group}",
                    record_id=f"r{group}",
                    group_id=f"g{group}",
                    timestamp=float(t + (600 if group == 2 else 0)),
                    duration_seconds=660 if group == 2 else 60,
                    accepted=group != 0,
                    label=None if group == 2 else 1,
                    reason="insufficient_history" if group == 0 else None,
                    label_reason="current_AF" if group == 2 else None,
                    features={"ecg_hr": None if group == 0 else 70.0 + t / 30},
                )
            )
    events = [dict(patient_id=f"p{i}", record_id=f"r{i}", group_id=f"g{i}", onset=600.0) for i in range(3)]
    (tmp_path / "rows.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    compact = _compact_cache(
        tmp_path,
        dict(
            feature_fields=["ecg_hr"],
            row_count=len(rows),
            rows_sha256=file_hash(tmp_path / "rows.jsonl"),
            allowed_groups=split["development_groups"],
            group_ids={f"p{i}": f"g{i}" for i in range(3)},
        ),
    )
    cached = audit_development(compact, events, config, split)
    assert cached["oracle_availability"]["eligible_events"] == 3
    assert cached["oracle_availability"]["event_sensitivity_ceiling"] == pytest.approx(1 / 3)
    assert cached["exposure_hours"]["eligible_including_rejected"] == pytest.approx(120 / 3600)
    assert cached["exposure_hours"]["available_eligible"] == pytest.approx(60 / 3600)
    assert cached["rejection_hours"]["insufficient_history"] == pytest.approx(60 / 3600)
    assert cached["features"]["ecg_hr"]["missing_accepted_eligible_fraction"] == 0
    original_groups = split["development_groups"]
    split["development_groups"] = ["g0", "g1", "heldout"]
    split["sha256"] = _digest({key: value for key, value in split.items() if key != "sha256"})
    with pytest.raises(ValueError, match="held-out"):
        audit_development(compact, events, config, split)
    split["development_groups"] = original_groups


def test_weighted_bootstrap_matches_duplicate_windows_and_aliases():
    from collections import Counter, defaultdict

    from cardiotwin.evaluation import _rates, patient_bootstrap, probability_metrics

    rows = [
        dict(patient_id=patient, group_id=group, label=label, risk=risk)
        for patient, group, label, risk in (
            ("a", "linked", 0, 0.1),
            ("alias", "linked", 1, 0.5),
            ("alias", "linked", 0, 0.5),
            ("b", "independent", 1, 1.0),
            ("b", "independent", 0, 0.5),
            ("c", "unlabeled", None, 0.2),
            ("c", "unlabeled", 0, None),
        )
    ]
    patients = {
        patient: Counter(
            monitored_seconds=60,
            eligible_seconds=30,
            available_seconds=30,
            warnings=1,
            false_warnings=1,
            unassessable_warnings=0,
            eligible_events=1,
            matched_events=int(patient == "alias"),
            warning_seconds=30,
        )
        for patient in ("a", "alias", "b", "c")
    }
    grouped = defaultdict(Counter)
    for patient, counts in patients.items():
        group = next(row["group_id"] for row in rows if row["patient_id"] == patient)
        grouped[group].update(counts)
    ids = list(grouped)
    estimates = defaultdict(list)
    rng = np.random.default_rng(8)
    for _ in range(50):
        selected = rng.choice(ids, len(ids), replace=True)
        total, duplicated = Counter(), []
        for draw, group in enumerate(selected):
            total.update(grouped[group])
            duplicated.extend(
                dict(row, patient_id=f"{draw}:{row['patient_id']}", group_id=str(draw))
                for row in rows
                if row["group_id"] == group
            )
        metrics = {
            **_rates(total),
            **{
                key: value
                for key, value in probability_metrics(duplicated).items()
                if key in {"brier", "pr_auc", "auroc", "expected_calibration_error", "group_macro_brier"}
            },
        }
        for key, value in metrics.items():
            if value is not None:
                estimates[key].append(value)
    actual = patient_bootstrap(patients, rows, seed=8, samples=50)
    for key, values in estimates.items():
        interval = actual["intervals"][key]
        assert (interval["lower"], interval["upper"]) == pytest.approx(
            np.percentile(values, [2.5, 97.5]), abs=1e-12
        )
        assert interval["valid_samples"] == len(values)
    collapsed = [
        dict(row, patient_id="a" if row["patient_id"] == "alias" else row["patient_id"]) for row in rows
    ]
    counts = dict(patients)
    counts["a"] = patients["a"] + counts.pop("alias")
    without_alias = patient_bootstrap(counts, collapsed, seed=8, samples=50)
    assert actual["independent_groups"] == without_alias["independent_groups"] == 3
    assert actual["intervals"] == without_alias["intervals"]
