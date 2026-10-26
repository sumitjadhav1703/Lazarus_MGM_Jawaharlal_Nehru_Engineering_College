import gzip
import json
import time

import numpy as np
import pytest
import test_ecgfounder_study_execution as execution
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cardiotwin import ecgfounder_compare as comparison
from cardiotwin import ecgfounder_study as study
from cardiotwin.precursor import freeze_policy_frontiers

pytestmark = pytest.mark.usefixtures("historical_ecgfounder_clock")


class ReplayModel(execution.SyntheticModel):
    def get_params(self, deep=False):
        return dict(super().get_params(deep), solver="lbfgs", class_weight=None)


class ReplayTransforms(execution.SyntheticTransforms):
    def __init__(self, pairs):
        self.fit_indices, self.row_masses, _ = comparison.pair_row_masses(pairs)

    def manifest(self):
        return dict(
            timing_schema=list(study.LINEAR_SCHEMA),
            timing_columns=46,
            delta_columns=1024,
            candidate_columns=1070,
            delta_order="preclassifier_channel_order",
            fit_indices=self.fit_indices.tolist(),
            row_masses=self.row_masses.tolist(),
            timing_imputer=[0.0] * 46,
            timing_scaler=dict(mean_=[0.0] * 46, var_=[1.0] * 46, scale_=[1.0] * 46),
            delta_scaler=dict(mean_=[0.0] * 1024, var_=[1.0] * 1024, scale_=[1.0] * 1024),
        )


def synthetic_calibrator(logits, labels):
    scale = StandardScaler()
    scale.mean_, scale.var_, scale.scale_ = np.zeros(1), np.ones(1), np.ones(1)
    scale.n_features_in_, scale.n_samples_seen_ = 1, len(labels)
    model = LogisticRegression(C=100.0, max_iter=1000, random_state=42)
    model.coef_, model.intercept_, model.classes_ = (
        np.ones((1, 1)),
        np.asarray([-2.197224577]),
        np.asarray([0, 1]),
    )
    model.n_features_in_ = 1
    return Pipeline([("scale", scale), ("sigmoid", model)])


def replay_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(execution, "SyntheticModel", ReplayModel)
    h = execution.harness(tmp_path / "fixture", monkeypatch, unavailable=[0])
    h.prepared.close()
    h.f.payload["allowed_stages"].append("replay")
    h.f.declaration.write_text(json.dumps(h.f.payload))
    h.prepared = execution.preparation.prepare(h.f)
    h.support = study.audit_common_support(h.prepared, guard=lambda: None)
    monkeypatch.setattr(study, "fit_transforms", lambda features, data, pairs: ReplayTransforms(pairs))
    monkeypatch.setattr(study, "fit_calibrator", synthetic_calibrator)
    monkeypatch.setattr(study, "calibration_summary", comparison.calibration_summary)
    monkeypatch.setattr(study, "freeze_policy_frontiers", freeze_policy_frontiers)

    def nulls(features, data, pairs, transforms, *, stop_groups):
        allowed = frozenset(p.group_id for p in pairs) | frozenset(stop_groups)
        for seed in range(101, 120):
            _, orientation = comparison.orientation_labels(pairs, seed)
            yield (
                seed,
                orientation,
                execution.SyntheticHead(transforms, "candidate", "fit_stop_only", allowed),
            )

    monkeypatch.setattr(study, "fit_null_heads", nulls)
    execution.execute(h, tmp_path / "run")
    return h


def replay(h, tmp_path):
    return study.replay_study(
        h.prepared,
        tmp_path / "run",
        tmp_path / "replay",
        expected_receipt_sha256=execution.preparation.sha(tmp_path / "run/receipt.json"),
        guard=lambda: None,
        storage_check=h.f.storage,
        deadline=time.monotonic() + 120,
    )


def forbid(*args, **kwargs):
    pytest.fail("Independent replay must never fit/select/rebuild pairs")


def test_independent_replay_rescores_all_artifacts_without_any_fit_or_policy_selection(tmp_path, monkeypatch):
    h = replay_fixture(tmp_path, monkeypatch)
    try:
        for name in (
            "fit_transforms",
            "fit_primary_heads",
            "fit_null_heads",
            "fit_nuisance_head",
            "fit_calibrator",
            "freeze_policy_frontiers",
        ):
            monkeypatch.setattr(study, name, forbid)
        for cls in (LogisticRegression, StandardScaler, SimpleImputer):
            monkeypatch.setattr(cls, "fit", forbid)
        import cardiotwin.investigation as investigation
        import cardiotwin.precursor as precursor

        monkeypatch.setattr(investigation, "select_policy", forbid)
        for name in ("make_roles", "build_event_pairs", "pair_support", "validate_pair_receipts"):
            monkeypatch.setattr(precursor, name, forbid)
        result = replay(h, tmp_path)
        assert result["status"] == "complete" and result["packages_checked"] == 66
        assert result["completed_arms"] == 6 and result["outcome_fits"] == 0
        assert result["rows_per_arm"] == len(h.prepared.rows)
        assert result["events_per_arm"] == len(h.prepared.events)
        assert result["decision"] == "reject" and not result["full_run_eligible"]
        assert (tmp_path / "replay/review.json.gz").is_file()
        assert not (tmp_path / "replay/review.json").exists()
    finally:
        h.prepared.close()


@pytest.mark.parametrize("fault", ["missing", "extra", "changed", "bad_receipt", "changed_report"])
def test_untrusted_or_incomplete_artifacts_reject_before_joblib_load(tmp_path, monkeypatch, fault):
    h = replay_fixture(tmp_path, monkeypatch)
    try:
        path = next((tmp_path / "run").glob("*.joblib"))
        if fault == "changed_report":
            (tmp_path / "run/summary.json.gz").write_bytes(gzip.compress(b"{}", mtime=0))
        elif fault == "missing":
            path.unlink()
        elif fault == "extra":
            (tmp_path / "run/unexpected").write_text("unexpected")
        elif fault == "changed":
            path.write_bytes(b"untrusted pickle")
        else:
            receipt = json.loads((tmp_path / "run/receipt.json").read_text())
            receipt["status"] = "running"
            (tmp_path / "run/receipt.json").write_text(json.dumps(receipt))
        monkeypatch.setattr(study.joblib, "load", forbid)
        with pytest.raises(ValueError):
            replay(h, tmp_path)
        assert json.loads((tmp_path / "replay/receipt.json").read_text())["status"] == "failed"
    finally:
        h.prepared.close()


def rebind_run(run):
    manifest = json.loads((run / "artifact-manifest.json").read_text())
    manifest["files_sha256"] = {
        n: execution.preparation.sha(run / n) for n in manifest["files_sha256"] if (run / n).exists()
    }
    (run / "artifact-manifest.json").write_text(json.dumps(manifest))
    receipt = json.loads((run / "receipt.json").read_text())
    receipt["artifact_manifest_sha256"] = execution.preparation.sha(run / "artifact-manifest.json")
    (run / "receipt.json").write_text(json.dumps(receipt))


def test_replay_requires_external_receipt_digest_before_loading_any_binary(tmp_path, monkeypatch):
    h = replay_fixture(tmp_path, monkeypatch)
    try:
        monkeypatch.setattr(study.joblib, "load", forbid)
        with pytest.raises(ValueError, match="Bound selection manifest changed"):
            study.replay_study(
                h.prepared,
                tmp_path / "run",
                tmp_path / "replay",
                expected_receipt_sha256="0" * 64,
                guard=lambda: None,
                storage_check=h.f.storage,
                deadline=time.monotonic() + 120,
            )
    finally:
        h.prepared.close()


@pytest.mark.parametrize(
    "fault",
    [
        "numeric",
        "transform",
        "calibrator",
        "policy",
        "search",
        "prediction",
        "bootstrap",
        "event",
        "checkpoint",
        "missing_numeric",
    ],
)
def test_independent_semantic_replay_rejects_corruption_even_when_rebound(tmp_path, monkeypatch, fault):
    h = replay_fixture(tmp_path, monkeypatch)
    run = tmp_path / "run"
    try:
        model = next(run.glob("*-timing_pointwise-model.joblib"))
        if fault in ("transform", "calibrator"):
            package = study.joblib.load(model)
            if fault == "transform":
                package["head"].transforms.fit_indices[0] += 1
            else:
                package["calibrator"]["sigmoid"].coef_[0, 0] = -1
            study.joblib.dump(package, model)
        elif fault == "missing_numeric":
            (run / (model.name + ".json")).unlink()
            monkeypatch.setattr(study.joblib, "load", forbid)
        elif fault == "prediction":
            path = next(run.glob("*predictions.jsonl.gz"))
            with gzip.open(path, "rt") as handle:
                rows = [json.loads(line) for line in handle]
            rows[0]["label"] = 1 - rows[0]["label"]
            with gzip.open(path, "wt") as handle:
                handle.writelines(json.dumps(v) + "\n" for v in rows)
        else:
            path = {
                "numeric": run / (model.name + ".json"),
                "policy": next(run.glob("*-policy-freeze.json")),
                "search": next(run.glob("*-policy-freeze.json")),
                "bootstrap": run / "summary.json.gz",
                "event": run / "events.json",
                "checkpoint": run / "checkpoint.json",
            }[fault]
            value = study._read_report(path) if path.name.endswith(".json.gz") else json.loads(path.read_text())
            if fault == "numeric":
                value["coefficients"][0][0] = 1
            elif fault == "policy":
                value["policies"][study.PRIMARY_POLICY]["alert"]["threshold"] = 0.123
            elif fault == "search":
                value["searches"][study.PRIMARY_POLICY][0]["warnings"] += 1
            elif fault == "bootstrap":
                value["comparisons"][study.PRIMARY_POLICY]["sensitivity_delta"] = 0.99
            elif fault == "checkpoint":
                value["status"] = "complete"
            else:
                value[0]["onset"] += 1
            data = json.dumps(value).encode()
            path.write_bytes(gzip.compress(data, mtime=0) if path.name.endswith(".json.gz") else data)
        rebind_run(run)
        with pytest.raises((ValueError, FileNotFoundError)):
            replay(h, tmp_path)
    finally:
        h.prepared.close()
