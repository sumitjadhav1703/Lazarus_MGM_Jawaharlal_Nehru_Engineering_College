import pytest

from cardiotwin.evaluation import evaluate_stream, match_warnings
from cardiotwin.modeling import feature_schema
from cardiotwin.train import (
    _calibrate,
    _fit_model,
    _select_policy,
    iter_record_rows,
    train_dataset,
    validate_splits,
)


def test_match_is_inclusive_one_to_one_and_uses_earliest_eligible_event():
    result = match_warnings([0, 480, 1000, 1100], [120, 600, 1700], 120, 600)
    assert result["matches"] == [
        {"warning": 0.0, "onset": 120.0, "lead_seconds": 120.0},
        {"warning": 480.0, "onset": 600.0, "lead_seconds": 120.0},
        {"warning": 1100.0, "onset": 1700.0, "lead_seconds": 600.0},
    ]
    assert result["false_warnings"] == [1000.0]
    assert result["missed_events"] == []


def test_schema_only_accepts_known_numeric_feature_groups():
    row = {
        "ecg_hr": 60,
        "clinical_age": 40,
        "baseline_ecg_hr": 2,
        "patient_id": 777,
        "timestamp": 1000,
        "label": 1,
        "path": "secret",
    }
    assert feature_schema([row], "combined") == ["clinical_age", "ecg_hr"]
    assert feature_schema([row], "personal") == ["baseline_ecg_hr", "clinical_age", "ecg_hr"]


def test_patient_partitions_and_presence_are_enforced():
    assert (
        validate_splits(
            {"a": "train", "b": "calibration", "c": "validation", "d": "test"}, ["a", "b", "c", "d"]
        )
        is None
    )
    with pytest.raises(ValueError, match="Unknown"):
        validate_splits({"a": "train"}, ["a", "b"])
    with pytest.raises(ValueError, match="partition"):
        validate_splits({"a": "train", "b": "calibration", "c": "validation"}, ["a", "b", "c"])


def test_rejected_time_does_not_hide_missed_events():
    config = {
        "target": {"cadence_seconds": 30, "gap_seconds": 120, "horizon_seconds": 600},
        "alert": {
            "threshold": 0.8,
            "release_threshold": 0.6,
            "persistence_seconds": 60,
            "cooldown_seconds": 600,
        },
    }
    rows = [
        {
            "patient_id": "a",
            "record_id": "r",
            "timestamp": t,
            "duration_seconds": 900,
            "accepted": False,
            "reason": "quality",
            "risk": None,
            "label": None,
            "label_reason": "unknown",
        }
        for t in range(0, 901, 30)
    ]
    report = evaluate_stream(rows, [{"patient_id": "a", "record_id": "r", "onset": 720}], config)
    assert report["eligible_events"] == 1
    assert report["missed_events"] == 1
    assert report["events_missed_during_abstention"] == 1
    assert report["monitored_hours"] == pytest.approx(0.25)
    assert report["rejected_hours"] == pytest.approx(0.25)
    assert report["false_warnings_per_hour"] is None
    assert report["false_warnings_per_all_recording_hour"] == 0
    assert report["exclusion_hours"]["unknown"] == pytest.approx(0.25)


def test_unknown_annotation_warnings_are_unassessable_not_false():
    config = {
        "target": {"cadence_seconds": 30, "gap_seconds": 120, "horizon_seconds": 600},
        "alert": {
            "threshold": 0.8,
            "release_threshold": 0.6,
            "persistence_seconds": 60,
            "cooldown_seconds": 600,
        },
    }
    rows = [
        {
            "patient_id": "a",
            "record_id": "r",
            "timestamp": t,
            "duration_seconds": 180,
            "accepted": True,
            "risk": 0.9,
            "label": None,
            "label_reason": "unknown",
        }
        for t in range(0, 181, 30)
    ]
    report = evaluate_stream(rows, [], config)
    assert report["warnings"] == 1
    assert report["unassessable_warnings"] == 1
    assert report["false_warnings"] == 0
    assert report["warning_precision"] is None
    assert report["warning_seconds"] == 120


def test_calibration_never_refits_training_imputer():
    train = [
        {"patient_id": str(i // 2), "accepted": True, "label": i % 2, "features": {"ecg_hr": 60 + i}}
        for i in range(8)
    ]
    model, schema, details = _fit_model(train, "logistic", "ecg", 42)
    assert model["impute"].statistics_[0] == 63.5
    calibration = [
        {"patient_id": "heldout", "accepted": True, "label": i % 2, "features": {"ecg_hr": 1000 + i}}
        for i in range(4)
    ]
    calibrator = _calibrate(model, schema, calibration, 42)
    assert model["impute"].statistics_[0] == 63.5
    assert calibrator.classes_.tolist() == [0, 1]
    assert details["training_patients"] == 4


def test_synthetic_flag_cannot_bypass_real_feasibility_gate(tmp_path):
    import json

    audit = tmp_path / "audit"
    audit.mkdir()
    (audit / "audit.json").write_text(json.dumps({"synthetic": False, "feasible": False}))
    with pytest.raises(ValueError, match="synthetic"):
        train_dataset(tmp_path, audit, tmp_path / "model", {"alert": {}}, synthetic=True)


def test_record_replay_includes_final_partial_cadence():
    from cardiotwin.config import load_config
    from cardiotwin.data import RecordInfo

    record = RecordInfo("r", "p", "/unused", 100, 95, False, {}, [])
    rows = list(iter_record_rows(record, load_config()))
    assert [row["timestamp"] for row in rows] == [0, 30, 60, 90, 95]


def test_valid_zero_initial_threshold_does_not_divide_by_zero():
    from cardiotwin.config import load_config

    config = load_config()
    config["alert"].update(
        threshold=0, release_threshold=0, max_false_warnings_per_hour=1, min_event_sensitivity=0.5
    )
    best, search = _select_policy([], [], config)
    assert best is None
    assert len(search) == 19


def test_unknown_time_cannot_dilute_assessable_false_warning_rate():
    from cardiotwin.config import load_config

    config = load_config()
    reports = []
    for duration in [180, 780]:
        rows = [
            {
                "patient_id": "p",
                "record_id": "r",
                "timestamp": t,
                "duration_seconds": duration,
                "accepted": t < 180,
                "risk": 0.9 if t < 180 else None,
                "label": 0 if t < 180 else None,
                "label_reason": None if t < 180 else "unknown",
                "reason": None if t < 180 else "quality",
            }
            for t in range(0, duration + 1, 30)
        ]
        reports.append(evaluate_stream(rows, [], config))
    assert reports[0]["false_warnings"] == reports[1]["false_warnings"] == 1
    assert reports[0]["false_warnings_per_hour"] == reports[1]["false_warnings_per_hour"] == 20
    assert reports[1]["false_warnings_per_all_recording_hour"] < 20


def test_washout_warning_cannot_match_later_event():
    from cardiotwin.config import load_config
    from cardiotwin.labels import RhythmInterval, eligible_onsets, label_at

    config = load_config()
    intervals = [
        RhythmInterval(0, 900, "other"),
        RhythmInterval(900, 1000, "AF"),
        RhythmInterval(1000, 1900, "other"),
        RhythmInterval(1900, 2000, "AF"),
        RhythmInterval(2000, 3000, "other"),
    ]
    assert label_at(1320, intervals, 3000, config) == (None, "washout")
    assert 1900 in eligible_onsets(intervals, 3000, config)
    rows = []
    for t in range(0, 3001, 30):
        label, reason = label_at(t, intervals, 3000, config)
        rows.append(
            {
                "patient_id": "p",
                "record_id": "r",
                "timestamp": t,
                "duration_seconds": 3000,
                "accepted": True,
                "risk": 0.9 if t in {1260, 1290, 1320} else 0.1,
                "label": label,
                "label_reason": reason,
            }
        )
    report = evaluate_stream(rows, [{"patient_id": "p", "record_id": "r", "onset": 1900}], config)
    assert report["matched_events"] == 0
    assert report["missed_events"] == 1
    assert report["unassessable_warnings"] == 1
    assert report["unassessable_warning_reasons"] == {"washout": 1}
    assert report["warning_details"][0]["matched"] is False
    assert report["warning_precision"] is None
    assert report["warning_seconds"] == 30


def test_training_rejects_nonempty_output_without_changing_stale_model(tmp_path):
    output = tmp_path / "model"
    output.mkdir()
    stale_model = output / "model.joblib"
    stale_model.write_bytes(b"prior frozen model")
    with pytest.raises(ValueError, match="fresh.*directory"):
        train_dataset(tmp_path, tmp_path / "missing-audit", output, {})
    assert stale_model.read_bytes() == b"prior frozen model"
    assert not (output / "training_report.json").exists()


def test_frozen_package_rejects_mismatched_sidecar_schema(tmp_path):
    import json

    from sklearn.linear_model import LogisticRegression

    from cardiotwin.config import load_config
    from cardiotwin.modeling import Predictor, build_pipeline, matrix
    from cardiotwin.train import _save_predictor

    schema = ["ecg_hr", "ecg_rr_mean"]
    rows = [{"ecg_hr": 60, "ecg_rr_mean": 1}, {"ecg_hr": 90, "ecg_rr_mean": 0.66}]
    pipeline = build_pipeline("logistic", {}, 42)
    pipeline.fit(matrix(rows, schema), [0, 1])
    calibrator = LogisticRegression().fit([[-1], [1]], [0, 1])
    config = load_config()
    predictor = Predictor(pipeline, calibrator, schema, config, {"version": "test"})
    _save_predictor(predictor, tmp_path, config, {"version": "test"}, {})
    assert Predictor.load(tmp_path).schema == schema
    (tmp_path / "feature_schema.json").write_text(json.dumps({"features": list(reversed(schema))}))
    with pytest.raises(ValueError, match="contract"):
        Predictor.load(tmp_path)


def test_research_only_blend_requires_explicit_research_loading(tmp_path):
    import hashlib
    import json

    import joblib

    from cardiotwin.config import load_config
    from cardiotwin.investigation import BlendPredictor
    from cardiotwin.modeling import Predictor

    config, schema = load_config(), ["ecg_hr"]
    member = Predictor(None, None, schema, config, {"version": "member"})
    blend = BlendPredictor(member, member, config, {"version": "embedded"})
    joblib.dump({"predictor": blend, "schema": schema, "config": config}, tmp_path / "model.joblib")
    checksum = hashlib.sha256((tmp_path / "model.joblib").read_bytes()).hexdigest()
    manifest = {"version": "frozen", "model_sha256": checksum, "deployment_ready": False}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "feature_schema.json").write_text(json.dumps({"features": schema}))
    with pytest.raises(ValueError, match="research-only"):
        Predictor.load(tmp_path)
    loaded = Predictor.load(tmp_path, research=True)
    assert isinstance(loaded, BlendPredictor)
    assert loaded.version == "frozen" and loaded.manifest == manifest
    # Dispatch still passes through the shared integrity guards.
    (tmp_path / "feature_schema.json").write_text(json.dumps({"features": []}))
    with pytest.raises(ValueError, match="contract"):
        Predictor.load(tmp_path, research=True)


def test_changed_input_requires_reaudit_before_training(tmp_path):
    import json

    from cardiotwin.config import load_config

    data = tmp_path / "data"
    audit = tmp_path / "audit"
    data.mkdir()
    audit.mkdir()
    (data / "manifest.json").write_text(json.dumps({"synthetic": True, "records": []}))
    (audit / "audit.json").write_text(
        json.dumps({"synthetic": True, "feasible": True, "dataset_fingerprint": "earlier input contents"})
    )
    config = load_config()
    config["alert"].update(max_false_warnings_per_hour=1, min_event_sensitivity=0.5)
    with pytest.raises(ValueError, match="Dataset differs"):
        train_dataset(data, audit, tmp_path / "model", config, synthetic=True)


def test_xgboost_stops_within_grouped_training_and_refits_fixed_tree_count():
    rows = [
        {
            "patient_id": str(i // 4),
            "accepted": True,
            "label": i % 2,
            "features": {"ecg_hr": 60 + 20 * (i % 2) + i // 4},
        }
        for i in range(32)
    ]
    model, schema, details = _fit_model(rows, "xgboost", "ecg", 42)
    chosen = details["chosen"]
    assert len(chosen["fold_tree_counts"]) == chosen["valid_folds"] == 3
    assert all(1 <= count <= 80 for count in chosen["fold_tree_counts"])
    assert model["model"].n_estimators == chosen["parameters"]["n_estimators"]
    assert model["model"].early_stopping_rounds is None
    assert model["impute"].statistics_[0] == 73.5


def test_patient_class_mass_is_equal_and_invariant_to_patient_duplication():
    import numpy as np

    from cardiotwin.train import training_weights

    for repeats in [1, 4]:
        patients = np.asarray(["a"] * (12 * repeats) + ["b"] * 3)
        labels = np.asarray(([0] * 10 + [1] * 2) * repeats + [0] * 3)
        weights = training_weights(labels, patients, "patient_class")
        assert weights.mean() == pytest.approx(1)
        assert weights[patients == "a"].sum() / weights.sum() == pytest.approx(0.5)
        assert weights[patients == "b"].sum() / weights.sum() == pytest.approx(0.5)
        assert weights[(patients == "a") & (labels == 0)].sum() / weights.sum() == pytest.approx(0.25)
        assert weights[(patients == "a") & (labels == 1)].sum() / weights.sum() == pytest.approx(0.25)
        assert weights[labels == 1].sum() / weights.sum() == pytest.approx(0.25)


def test_patient_macro_loss_gives_equal_weight_to_patients():
    import numpy as np

    from cardiotwin.train import patient_macro_log_loss

    labels = np.asarray([0] * 20 + [0])
    risks = np.asarray([0.1] * 20 + [0.9])
    groups = np.asarray(["long"] * 20 + ["short"])
    assert patient_macro_log_loss(labels, risks, groups) == pytest.approx((-np.log(0.9) - np.log(0.1)) / 2)


def test_audited_groups_cannot_cross_development_partitions():
    splits = {"a": "train", "b": "calibration", "c": "validation", "d": "test"}
    groups = {"a": "linked", "b": "linked", "c": "c", "d": "d"}
    with pytest.raises(ValueError, match="group.*partition"):
        validate_splits(splits, list(splits), groups)


def test_grouped_cv_keeps_linked_ids_together_and_archives_macro_loss():
    rows = [
        {
            "patient_id": f"patient-{i // 4}",
            "group_id": f"group-{i // 8}",
            "accepted": True,
            "label": i % 2,
            "features": {"ecg_hr": 60 + 10 * (i % 2) + i // 8},
        }
        for i in range(48)
    ]
    _, _, details = _fit_model(rows, "logistic", "ecg", 42)
    for search in details["search"]:
        assert search["group_macro_log_loss"] is not None
        assert search["window_log_loss"] is not None
        for fold in search["folds"]:
            assert set(fold["fit_groups"]).isdisjoint(fold["hold_groups"])
            assert fold["converged"] is True
    assert details["training_groups"] == 6


def test_grouped_cv_rejects_holdouts_without_any_positive_assessment():
    rows = [
        {
            "patient_id": f"p{patient}",
            "group_id": f"g{patient}",
            "accepted": True,
            "label": int(patient == 0 and i >= 6),
            "features": {"ecg_hr": 60 + 20 * int(patient == 0 and i >= 6) + i * 0.01},
        }
        for patient in range(3)
        for i in range(12)
    ]
    with pytest.raises(ValueError, match="grouped.*fold") as failure:
        _fit_model(rows, "logistic", "ecg", 42)
    for candidate in failure.value.args[1]["search"]:
        coverage = candidate["oof_coverage"]
        assert candidate["valid_folds"] == 2
        assert coverage["positive_windows"] == 0
        assert coverage["has_both_classes"] is False
        assert coverage["window_fraction"] == pytest.approx(2 / 3)
        assert coverage["unassessed_positive_group_ids"] == ["g0"]


def test_fitting_weights_and_macro_loss_use_independent_groups_not_patient_aliases():
    import numpy as np

    from cardiotwin.modeling import matrix
    from cardiotwin.train import patient_macro_log_loss

    rows = [
        {
            "patient_id": f"p{group}-{i // 2}" if group == 0 else f"p{group}",
            "group_id": f"g{group}",
            "accepted": True,
            "label": i % 2,
            "features": {"ecg_hr": 60 + 10 * (i % 2) + group},
        }
        for group in range(6)
        for i in range(8)
    ]
    model, schema, details = _fit_model(rows, "logistic", "ecg", 42)
    masses = details["weighting"]["group_mass"]
    assert len(set(masses.values())) == 1
    assert details["training_patients"] == 9
    assert details["training_groups"] == 6
    assert details["weighting"]["independent_unit"] == "audited_group_id"
    assert details["cv_selection_metric"] == "group_macro_log_loss"
    # Alias relabeling must not change fitting or the scored independent unit.
    canonical = [dict(row, patient_id=row["group_id"]) for row in rows]
    canonical_model, _, canonical_details = _fit_model(canonical, "logistic", "ecg", 42)
    x = matrix([row["features"] for row in rows], schema)
    assert np.allclose(model.predict_proba(x), canonical_model.predict_proba(x))
    assert details["chosen"]["group_macro_log_loss"] == pytest.approx(
        canonical_details["chosen"]["group_macro_log_loss"]
    )
    labels, risks = np.array([0, 0, 0]), np.array([0.1, 0.1, 0.9])
    assert patient_macro_log_loss(labels, risks, ["linked", "linked", "other"]) == pytest.approx(
        (-np.log(0.9) - np.log(0.1)) / 2
    )


def test_validation_ties_choose_model_simplicity_before_candidate_order():
    from cardiotwin.train import _candidate_rank

    policy = {"event_sensitivity": 0.5, "false_warnings_per_hour": 0.1}
    candidates = [
        {"model": "xgboost_ecg", "feature_count": 6, "selected_policy": policy},
        {"model": "elasticnet_personal_engineered", "feature_count": 60, "selected_policy": policy},
        {"model": "logistic_ecg_engineered", "feature_count": 23, "selected_policy": policy},
    ]
    assert max(candidates, key=_candidate_rank)["model"] == "logistic_ecg_engineered"
    candidates.append({"model": "logistic_ecg", "feature_count": 6, "selected_policy": policy})
    assert max(candidates, key=_candidate_rank)["model"] == "logistic_ecg"


def test_no_independent_tuning_groups_fails_instead_of_silently_selecting():
    rows = [
        {
            "patient_id": f"p{i}",
            "group_id": "same",
            "accepted": True,
            "label": i % 2,
            "features": {"ecg_hr": 60 + 10 * (i % 2)},
        }
        for i in range(8)
    ]
    with pytest.raises(ValueError, match="grouped.*fold"):
        _fit_model(rows, "logistic", "ecg", 42)


def test_elasticnet_pipeline_has_scaled_regularized_logistic_challenger():
    from cardiotwin.modeling import build_pipeline

    model = build_pipeline("elasticnet", {"C": 0.1, "l1_ratio": 0.5}, 42)
    assert model["model"].solver == "saga"
    assert model["model"].l1_ratio == 0.5
    assert model["model"].C == 0.1
    assert "scale" in model.named_steps


def test_calibration_scales_only_held_out_scores_with_fixed_regularization():
    import numpy as np

    from cardiotwin.modeling import matrix, raw_score

    train = [
        {
            "patient_id": f"p{i // 2}",
            "accepted": True,
            "label": i % 2,
            "features": {"ecg_hr": 60 + 20 * (i % 2) + i // 2},
        }
        for i in range(12)
    ]
    model, schema, _ = _fit_model(train, "logistic", "ecg", 42)
    before = model["scale"].mean_.copy()
    held = [
        {
            "patient_id": "held",
            "group_id": "held",
            "accepted": True,
            "label": i % 2,
            "features": {"ecg_hr": 100 + i},
        }
        for i in range(4)
    ]
    calibrator = _calibrate(model, schema, held, 42, calibration_c=1)
    assert np.array_equal(model["scale"].mean_, before)
    scores = raw_score(model, matrix([r["features"] for r in held], schema))
    assert calibrator["scale"].mean_[0] == pytest.approx(scores.mean())
    assert calibrator["sigmoid"].C == 1


def test_engineered_allowlist_preserves_core_ablations_and_excludes_identifiers():
    from cardiotwin.features import ENGINEERED_ECG_FIELDS

    row = {key: 1.0 for key in ENGINEERED_ECG_FIELDS}
    row.update(
        ecg_hr=60,
        clinical_age=50,
        baseline_ecg_rr_cv_deviation=0.1,
        baseline_ecg_rr_cv_zscore=0.2,
        ecg_patient_id=999,
        patient_id=888,
        label=1,
        timestamp=777,
    )
    assert feature_schema([row], "combined") == ["clinical_age", "ecg_hr"]
    engineered = feature_schema([row], "combined_engineered")
    assert set(ENGINEERED_ECG_FIELDS) <= set(engineered)
    assert not any("id" in key or key.startswith("baseline_") for key in engineered)
    personal = feature_schema([row], "personal_engineered")
    assert "baseline_ecg_rr_cv_deviation" in personal
    assert "baseline_ecg_rr_cv_zscore" in personal


def test_frozen_predictor_rejects_wrong_feature_version_before_deserialization(tmp_path):
    import json

    (tmp_path / "manifest.json").write_text(json.dumps({"version": "old"}))
    (tmp_path / "config.json").write_text(json.dumps({"feature_version": "unknown-version"}))
    (tmp_path / "model.joblib").write_bytes(b"invalid pickle must not be deserialized")
    from cardiotwin.modeling import Predictor

    with pytest.raises(ValueError, match="feature.*version"):
        Predictor.load(tmp_path)


@pytest.mark.parametrize("mode", ["patient_only", "window_class"])
def test_explicit_weighting_ablations_have_declared_mass(mode):
    import numpy as np

    from cardiotwin.train import training_weights

    labels, patients = np.asarray([0, 0, 0, 1]), np.asarray(["a", "a", "a", "b"])
    weights = training_weights(labels, patients, mode)
    assert weights.mean() == pytest.approx(1)
    if mode == "patient_only":
        assert weights[patients == "a"].sum() == pytest.approx(weights[patients == "b"].sum())
    else:
        assert weights[labels == 0].sum() == pytest.approx(weights[labels == 1].sum())


def test_calibration_diagnostics_report_natural_prevalence_and_independent_groups():
    from cardiotwin.train import calibration_diagnostics

    train = [
        {
            "patient_id": f"p{i // 2}",
            "accepted": True,
            "label": i % 2,
            "features": {"ecg_hr": 60 + 20 * (i % 2) + i // 2},
        }
        for i in range(12)
    ]
    model, schema, _ = _fit_model(train, "logistic", "ecg", 42)
    held = [
        {
            "patient_id": f"held-{i // 2}",
            "group_id": "linked-held",
            "accepted": True,
            "label": int(i == 3),
            "features": {"ecg_hr": 70 + i},
        }
        for i in range(4)
    ]
    calibrator = _calibrate(model, schema, held, 42, calibration_c=1)
    details = calibration_diagnostics(model, schema, calibrator, held)
    assert details["patients"] == 2
    assert details["groups"] == 1
    assert details["positive_prevalence"] == 0.25
    assert details["sample_weighting"] == "none_natural_monitoring_prevalence"
    assert details["calibration_c"] == 1
    assert details["raw_probability"]["labeled_windows"] == 4
    assert details["calibrated_probability"]["labeled_windows"] == 4


def test_failed_challenger_is_reported_without_preventing_valid_baseline():
    from cardiotwin.config import load_config
    from cardiotwin.train import _develop_candidate

    config = load_config()
    config["alert"].update(max_false_warnings_per_hour=1, min_event_sensitivity=0.5)
    train = [
        {
            "patient_id": f"p{i // 2}",
            "group_id": f"p{i // 2}",
            "accepted": True,
            "label": i % 2,
            "features": {"ecg_hr": 60 + 20 * (i % 2) + i // 2},
        }
        for i in range(12)
    ]
    held = [
        {
            "patient_id": "held",
            "group_id": "held",
            "accepted": True,
            "label": i % 2,
            "features": {"ecg_hr": 60 + 20 * (i % 2)},
        }
        for i in range(4)
    ]
    failed, predictor = _develop_candidate(train, held, [], [], config, "logistic", "clinical")
    assert failed["status"] == "failed"
    assert "No audited clinical features" in failed["reason"]
    assert predictor is None
    good, predictor = _develop_candidate(train, held, [], [], config, "logistic", "ecg")
    assert good["status"] == "developed"
    assert predictor is not None
    assert good["selected_policy"] is None
