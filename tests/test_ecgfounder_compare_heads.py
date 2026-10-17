"""Synthetic fixed pointwise heads, controls, natural calibration and common abstentions."""

from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from test_ecgfounder_compare import compare, fixture, pairs, timing_rows


def require(name):
    module = compare()
    assert hasattr(module, name), f"Pending comparison unit2 feature: {name}"
    return module


def test_primary_heads_fit_pointwise_rows_with_fixed_parameters_and_identical_preprocessing(
    tmp_path, monkeypatch
):
    module = require("fit_primary_heads")
    _, _, args = fixture(tmp_path, monkeypatch)
    timing = timing_rows()
    with module.VerifiedTables.open(**args) as data:
        transforms = module.fit_transforms(timing, data, pairs())
        heads = module.fit_primary_heads(timing, data, pairs(), transforms)
        assert list(heads) == ["timing_pointwise", "timing_ecgfounder_delta_pointwise"]
        for name, width in (("timing_pointwise", 46), ("timing_ecgfounder_delta_pointwise", 1070)):
            head = heads[name]
            assert head.transforms is transforms
            assert head.model.coef_.shape == (1, width)
            parameters = head.model.get_params()
            assert (
                parameters["C"],
                parameters["solver"],
                parameters["max_iter"],
                parameters["fit_intercept"],
                parameters["class_weight"],
                parameters["random_state"],
            ) == (0.1, "lbfgs", 3000, False, None, 42)
            features = [timing[i] for i in (1, 1, 2, 3)]
            delta = data.batch([1, 1, 2, 3])[1] if width == 1070 else None
            design = transforms.transform(features, delta)
            reference = LogisticRegression(
                C=0.1, solver="lbfgs", max_iter=3000, fit_intercept=False, class_weight=None, random_state=42
            ).fit(design, [1, 1, 0, 0], sample_weight=[1.0, 1.0, 1.0, 1.0])
            np.testing.assert_allclose(head.model.coef_, reference.coef_, rtol=1e-12, atol=1e-12)


def test_raw_logits_keep_common_abstentions_and_repeated_requested_order(tmp_path, monkeypatch):
    module = require("fit_primary_heads")
    _, _, args = fixture(tmp_path, monkeypatch)
    timing = timing_rows()
    with module.VerifiedTables.open(**args) as data:
        transforms = module.fit_transforms(timing, data, pairs())
        heads = module.fit_primary_heads(timing, data, pairs(), transforms)
        for head in heads.values():
            logits = head.logits(timing, data, [3, 0, 1, 3, 4])
            assert logits.shape == (5,) and np.isnan(logits[[1, 4]]).all()
            assert np.isfinite(logits[[0, 2, 3]]).all() and logits[0] == logits[3]
            with pytest.raises(ValueError):
                head.logits(timing, data, list(range(4097)))


def test_orientation_nulls_swap_labels_as_whole_sorted_people_without_changing_vectors():
    module = require("orientation_labels")
    original = [SimpleNamespace(group_id="h"), SimpleNamespace(group_id="g"), SimpleNamespace(group_id="g")]
    labels, orientation = module.orientation_labels(original, 101)
    assert orientation == {"g": -1.0, "h": 1.0}
    assert labels.tolist() == [1, 0, 0, 0, 1, 1]
    labels2, orientation2 = module.orientation_labels(list(reversed(original)), 101)
    assert orientation2 == orientation and labels2.tolist() == [0, 0, 1, 1, 1, 0]
    with pytest.raises(ValueError):
        module.orientation_labels(original, 42)


def test_nineteen_null_heads_reuse_real_transforms_and_keep_fit_stop_only_scope(tmp_path, monkeypatch):
    module = require("fit_null_heads")
    _, _, args = fixture(tmp_path, monkeypatch)
    timing = timing_rows()
    with module.VerifiedTables.open(**args) as data:
        transforms = module.fit_transforms(timing, data, pairs())
        seeds = []
        for seed, orientation, head in module.fit_null_heads(
            timing, data, pairs(), transforms, stop_groups={"stop"}
        ):
            seeds.append(seed)
            assert head.transforms is transforms and head.model.coef_.shape == (1, 1070)
            assert head.scope == "fit_stop_only" and head.allowed_groups == frozenset({"g", "stop"})
            assert set(orientation) == {"g"}
            assert np.isfinite(head.logits(timing, data, [1, 2, 3])).all()
        assert seeds == list(range(101, 120))


def test_nuisance_control_is52_columns_and_uses_same_fit_row_masses(tmp_path, monkeypatch):
    module = require("fit_nuisance_head")
    _, _, args = fixture(tmp_path, monkeypatch)
    timing = timing_rows()
    with module.VerifiedTables.open(**args) as data:
        transforms = module.fit_transforms(timing, data, pairs())
        head = module.fit_nuisance_head(timing, data, pairs(), transforms, stop_groups={"stop"})
        assert head.model.coef_.shape == (1, 52) and head.transforms is transforms
        np.testing.assert_array_equal(head.nuisance_scaler.mean_, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_equal(head.nuisance_scaler.scale_, [1.0] * 6)
        assert head.scope == "fit_stop_only"
        with pytest.raises(ValueError):
            list(module.score_rows(head, None, [], data, []))


def test_natural_logit_calibrator_matches_existing_calibrate_exactly():
    from cardiotwin.train import _calibrate

    module = require("fit_calibrator")
    logits = np.array([-2.0, -1.0, 1.0, 2.0])
    labels = np.array([0, 0, 1, 1])
    rows = [dict(accepted=True, label=int(y), features={"scalar": float(x)}) for x, y in zip(logits, labels)]
    direct = _calibrate(
        SimpleNamespace(decision_function=lambda x: x[:, 0]), ["scalar"], rows, 42, calibration_c=100.0
    )
    fitted = module.fit_calibrator(logits, labels)
    np.testing.assert_allclose(
        fitted.predict_proba(logits[:, None]), direct.predict_proba(logits[:, None]), rtol=1e-12, atol=1e-12
    )
    summary = module.calibration_summary(fitted)
    assert summary["calibration_c"] == 100.0 and summary["raw_score_slope"] > 0
    assert summary["sample_weighting"] == "none_natural_monitoring_prevalence"


@pytest.mark.parametrize(
    "logits,labels",
    [
        ([0.0, 1.0], [0, 0]),
        ([0.0, np.nan], [0, 1]),
        ([0.0, np.inf], [0, 1]),
        ([0.0, 1.0], [0, -1]),
        ([0.0], [0, 1]),
    ],
)
def test_calibrator_rejects_single_class_unavailable_or_misaligned_inputs(logits, labels):
    module = require("fit_calibrator")
    with pytest.raises(ValueError):
        module.fit_calibrator(np.asarray(logits), np.asarray(labels))


def test_nonpositive_calibration_slope_is_reported_without_silent_polarity_flip():
    module = require("fit_calibrator")
    fitted = module.fit_calibrator(np.array([-2.0, -1.0, 1.0, 2.0]), np.array([1, 1, 0, 0]))
    summary = module.calibration_summary(fitted)
    assert summary["raw_score_slope"] < 0 and not summary["positive_slope"]


def test_primary_score_stream_keeps_all_rows_original_acceptance_and_evaluator_denominators(
    tmp_path, monkeypatch
):
    from cardiotwin.config import load_config
    from cardiotwin.evaluation import evaluate_stream

    module = require("score_rows")
    _, metadata, args = fixture(tmp_path, monkeypatch)
    timing = timing_rows()
    rows = [
        dict(item.values, features=timing[i], label=0, label_reason=None, reason=None, duration_seconds=750.0)
        for i, item in enumerate(metadata)
    ]
    with module.VerifiedTables.open(**args) as data:
        transforms = module.fit_transforms(timing, data, pairs())
        head = module.fit_primary_heads(timing, data, pairs(), transforms)["timing_pointwise"]
        calibrator = module.fit_calibrator(np.array([-2.0, -1.0, 1.0, 2.0]), np.array([0, 0, 1, 1]))
        scored = list(module.score_rows(head, calibrator, rows, data, list(range(5))))
        assert len(scored) == 5 and [r["row_index"] for r in scored] == list(range(5))
        assert [r["original_accepted"] for r in scored] == [False, True, True, True, True]
        assert [r["accepted"] for r in scored] == [False, True, True, True, False]
        assert all(scored[i]["risk"] is None and scored[i]["raw_logit"] is None for i in (0, 4))
        assert all(0 < scored[i]["risk"] < 1 for i in (1, 2, 3))
        assert all(r["label"] == 0 for r in scored)
        assert all("features" not in row for row in scored)
        config = load_config()
        config["alert"].update(
            policy_version="finite-validity-v1",
            max_duration_seconds=600.0,
            rearm_seconds=60.0,
            max_burden_occupancy=0.05,
            max_eligible_negative_burden_occupancy=0.05,
            threshold=1.0,
        )
        report = evaluate_stream(scored, [], config, bootstrap_samples=0)
        assert report["eligible_hours"] == 150 / 3600 and report["monitored_hours"] == 750 / 3600
        assert report["per_patient"][0]["available_seconds"] == 90.0
        assert report["warnings"] == 0


def test_heads_reject_transforms_fitted_on_different_pair_masses(tmp_path, monkeypatch):
    module = require("fit_primary_heads")
    _, _, args = fixture(tmp_path, monkeypatch)
    timing = timing_rows()
    with module.VerifiedTables.open(**args) as data:
        transforms = module.fit_transforms(timing, data, pairs())
        with pytest.raises(ValueError):
            module.fit_primary_heads(timing, data, pairs()[:1], transforms)
        transforms.row_masses[0] += 0.01
        with pytest.raises(ValueError):
            module.fit_primary_heads(timing, data, pairs(), transforms)


def test_control_scores_block_groups_outside_fit_stop_and_overlap(tmp_path, monkeypatch):
    module = require("fit_null_heads")
    _, _, args = fixture(tmp_path, monkeypatch)
    timing = timing_rows()
    with module.VerifiedTables.open(**args) as data:
        transforms = module.fit_transforms(timing, data, pairs())
        null = next(module.fit_null_heads(timing, data, pairs(), transforms, stop_groups={"stop"}))[2]
        nuisance = module.fit_nuisance_head(timing, data, pairs(), transforms, stop_groups={"stop"})
        data.records = (("r", "outer", "p"),)
        for head in (null, nuisance):
            with pytest.raises(ValueError):
                head.logits(timing, data, [0])
        data.records = (("r", "g", "p"),)
        with pytest.raises(ValueError):
            next(module.fit_null_heads(timing, data, pairs(), transforms, stop_groups={"g"}))


@pytest.mark.parametrize(
    "field,value", [("row_index", 2), ("timestamp", 999.0), ("accepted", False), ("patient_id", "wrong")]
)
def test_scoring_rejects_original_identity_or_acceptance_mismatch(tmp_path, monkeypatch, field, value):
    module = require("score_rows")
    _, metadata, args = fixture(tmp_path, monkeypatch)
    timing = timing_rows()
    rows = [dict(item.values, features=timing[i], label=0) for i, item in enumerate(metadata)]
    with module.VerifiedTables.open(**args) as data:
        transforms = module.fit_transforms(timing, data, pairs())
        head = module.fit_primary_heads(timing, data, pairs(), transforms)["timing_pointwise"]
        calibrator = module.fit_calibrator(np.array([-2.0, -1.0, 1.0, 2.0]), np.array([0, 0, 1, 1]))
        rows[1][field] = value
        with pytest.raises(ValueError):
            list(module.score_rows(head, calibrator, rows, data, [1]))


def test_scoring_stream_batches_large_requests_without_dropping_rows(tmp_path, monkeypatch):
    module = require("score_rows")
    _, metadata, args = fixture(tmp_path, monkeypatch)
    timing = timing_rows()
    rows = [dict(item.values, features=timing[i], label=0) for i, item in enumerate(metadata)]
    with module.VerifiedTables.open(**args) as data:
        transforms = module.fit_transforms(timing, data, pairs())
        head = module.fit_primary_heads(timing, data, pairs(), transforms)["timing_pointwise"]
        calibrator = module.fit_calibrator(np.array([-2.0, -1.0, 1.0, 2.0]), np.array([0, 0, 1, 1]))
        scored = list(module.score_rows(head, calibrator, rows, data, iter([1] * 4097)))
        assert len(scored) == 4097 and all(row["accepted"] for row in scored)
        assert all(row["row_index"] == 1 for row in scored)


def test_calibrator_stops_on_convergence_warning(tmp_path, monkeypatch):
    import warnings

    from sklearn.exceptions import ConvergenceWarning

    module = require("fit_calibrator")
    original = LogisticRegression.fit

    def warn_after_fit(self, *args, **kwargs):
        fitted = original(self, *args, **kwargs)
        warnings.warn("synthetic nonconvergence", ConvergenceWarning)
        return fitted

    monkeypatch.setattr(LogisticRegression, "fit", warn_after_fit)
    with pytest.raises(ValueError, match="converge"):
        module.fit_calibrator(np.array([-2.0, -1.0, 1.0, 2.0]), np.array([0, 0, 1, 1]))


def test_null_coefficients_use_swapped_labels_on_unchanged_pointwise_rows(tmp_path, monkeypatch):
    module = require("fit_null_heads")
    _, _, args = fixture(tmp_path, monkeypatch)
    timing = timing_rows()
    with module.VerifiedTables.open(**args) as data:
        transforms = module.fit_transforms(timing, data, pairs())
        seed, orientation, head = next(
            module.fit_null_heads(timing, data, pairs(), transforms, stop_groups={"stop"})
        )
        assert seed == 101 and orientation == {"g": -1.0}
        design = transforms.transform([timing[i] for i in (1, 1, 2, 3)], data.batch([1, 1, 2, 3])[1])
        reference = LogisticRegression(
            C=0.1, solver="lbfgs", max_iter=3000, fit_intercept=False, class_weight=None, random_state=42
        ).fit(design, [0, 0, 1, 1], sample_weight=[1.0, 1.0, 1.0, 1.0])
        np.testing.assert_allclose(head.model.coef_, reference.coef_, rtol=1e-12, atol=1e-12)


def test_nuisance_fit_uses_weighted_unique_rows_and_retains_timing_columns(tmp_path, monkeypatch):
    from test_ecgfounder_compare import reseal

    module = require("fit_nuisance_head")
    _, _, args = fixture(tmp_path, monkeypatch)
    path = args["output"] / "nuisance.npy"
    nuisance = np.load(path)
    nuisance[1] = 1.0
    nuisance[2] = 3.0
    nuisance[3] = 7.0
    np.save(path, nuisance)
    args["receipt_sha256"] = reseal(args["output"], table="nuisance.npy")
    timing = timing_rows()
    with module.VerifiedTables.open(**args) as data:
        transforms = module.fit_transforms(timing, data, pairs())
        head = module.fit_nuisance_head(timing, data, pairs(), transforms, stop_groups={"stop"})
        np.testing.assert_array_equal(head.nuisance_scaler.mean_, [3.0] * 6)
        np.testing.assert_array_equal(head.nuisance_scaler.var_, [6.0] * 6)
        design = np.hstack(
            (
                transforms.transform([timing[i] for i in (1, 1, 2, 3)]),
                np.repeat(np.array([[-2.0], [-2.0], [0.0], [4.0]]) / np.sqrt(6.0), 6, axis=1),
            )
        )
        reference = LogisticRegression(
            C=0.1, solver="lbfgs", max_iter=3000, fit_intercept=False, class_weight=None, random_state=42
        ).fit(design, [1, 1, 0, 0], sample_weight=[1.0, 1.0, 1.0, 1.0])
        np.testing.assert_allclose(head.model.coef_, reference.coef_, rtol=1e-12, atol=1e-12)


def test_score_compaction_retains_unknown_excluded_metadata_and_exact_probability_clipping(
    tmp_path, monkeypatch
):
    import json

    module = require("score_rows")
    _, metadata, args = fixture(tmp_path, monkeypatch)
    timing = timing_rows()
    rows = [
        dict(
            item.values,
            features=timing[i],
            label=-1,
            label_reason="unknown_tail",
            excluded_onsets=[123.0],
            duration_seconds=750.0,
        )
        for i, item in enumerate(metadata)
    ]
    with module.VerifiedTables.open(**args) as data:
        transforms = module.fit_transforms(timing, data, pairs())
        head = module.fit_primary_heads(timing, data, pairs(), transforms)["timing_pointwise"]
        saturating = SimpleNamespace(predict_proba=lambda x: np.tile([0.0, 1.0], (len(x), 1)))
        scored = list(module.score_rows(head, saturating, rows, data, [0, 1, 4]))
        assert scored[1]["risk"] == np.nextafter(1.0, 0.0)
        assert scored[0]["risk"] is None and scored[2]["risk"] is None
        assert all(
            row["label"] == -1
            and row["label_reason"] == "unknown_tail"
            and row["excluded_onsets"] == [123.0]
            and row["duration_seconds"] == 750.0
            for row in scored
        )
        assert "features" not in json.loads(json.dumps(scored, allow_nan=False))[0]
