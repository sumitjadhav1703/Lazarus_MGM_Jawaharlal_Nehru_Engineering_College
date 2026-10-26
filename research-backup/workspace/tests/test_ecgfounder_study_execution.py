import copy
import gzip
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
import test_ecgfounder_study as preparation

from cardiotwin import ecgfounder_study as study
from cardiotwin.precursor import PRESETS

pytestmark = pytest.mark.usefixtures("historical_ecgfounder_clock")


class SyntheticTransforms:
    def __init__(self, pairs):
        self.fit_indices = np.asarray(
            sorted({i for p in pairs for i in (p.positive_index, p.negative_index)})
        )

    def manifest(self):
        return dict(
            timing_columns=46,
            delta_columns=1024,
            candidate_columns=1070,
            fit_indices=self.fit_indices.tolist(),
            fixture="synthetic_no_fit",
        )

    def identity_sha256(self):
        return study._canonical_sha(self.manifest())


class SyntheticModel:
    def __init__(self, width):
        self.coef_ = np.zeros((1, width))
        self.intercept_ = np.zeros(1)
        self.classes_ = np.asarray([0, 1])

    def get_params(self, deep=False):
        return dict(C=0.1, random_state=42, fit_intercept=False, max_iter=3000)


class SyntheticHead:
    def __init__(self, transforms, kind="timing", scope="primary", allowed_groups=None):
        self.transforms, self.kind, self.scope = transforms, kind, scope
        self.allowed_groups = allowed_groups
        self.model = SyntheticModel(dict(timing=46, candidate=1070, nuisance=52)[kind])
        if kind == "nuisance":
            self.nuisance_imputer = SimpleNamespace(statistics_=np.zeros(6))
            self.nuisance_scaler = SimpleNamespace(mean_=np.zeros(6), var_=np.ones(6), scale_=np.ones(6))

    def _predict(self, features, available, delta, nuisance):
        return np.where(available, 0.0, np.nan)

    def logits(self, features, data, indices):
        if self.allowed_groups is not None:
            assert all(
                data.records[int(data.rows[i]["record_index"])][1] in self.allowed_groups for i in indices
            )
        return np.where(data.rows[indices]["available"], 0.0, np.nan)


class SyntheticCalibrator:
    def predict_proba(self, logits):
        return np.tile([0.9, 0.1], (len(logits), 1))


def harness(tmp_path, monkeypatch, unavailable=()):
    config = preparation.load_config()
    config["alert"].update(
        policy_version="finite-validity-v1",
        max_duration_seconds=600,
        rearm_seconds=60,
        max_burden_occupancy=0.05,
        max_eligible_negative_burden_occupancy=0.05,
    )
    monkeypatch.setattr(preparation, "load_config", lambda: copy.deepcopy(config))
    f = preparation.fixture(tmp_path, monkeypatch, unavailable=unavailable)
    f.payload["allowed_stages"].append("run")
    f.declaration.write_text(json.dumps(f.payload))
    p = preparation.prepare(f)
    support = study.audit_common_support(p, guard=lambda: None)
    calls = []
    monkeypatch.setattr(
        study, "fit_transforms", lambda features, data, pairs: SyntheticTransforms(pairs), raising=False
    )

    def primary(features, data, pairs, transforms):
        calls.append(("primary", {pair.group_id for pair in pairs}))
        return dict(
            timing_pointwise=SyntheticHead(transforms),
            timing_ecgfounder_delta_pointwise=SyntheticHead(transforms, "candidate"),
        )

    def nulls(features, data, pairs, transforms, *, stop_groups):
        allowed = set(stop_groups) | {pair.group_id for pair in pairs}
        for seed in range(101, 120):
            calls.append(("null", seed))
            yield (
                seed,
                {g: 1.0 for g in allowed - set(stop_groups)},
                SyntheticHead(transforms, "candidate", "fit_stop_only", allowed),
            )

    monkeypatch.setattr(study, "fit_primary_heads", primary, raising=False)
    monkeypatch.setattr(study, "fit_null_heads", nulls, raising=False)
    monkeypatch.setattr(
        study,
        "fit_nuisance_head",
        lambda features, data, pairs, transforms, *, stop_groups: SyntheticHead(
            transforms, "nuisance", "fit_stop_only", set(stop_groups) | {v.group_id for v in pairs}
        ),
        raising=False,
    )

    def calibrate(logits, labels):
        assert len(logits) == len(labels) and set(labels) == {0, 1}
        calls.append(("calibrate", len(labels)))
        return SyntheticCalibrator()

    monkeypatch.setattr(study, "fit_calibrator", calibrate, raising=False)
    monkeypatch.setattr(
        study,
        "calibration_summary",
        lambda cal: dict(raw_score_slope=1.0, raw_score_intercept=0.0, positive_slope=True),
        raising=False,
    )

    def freeze(rows, events, config, *, deadline):
        assert deadline > time.monotonic() and len({r["group_id"] for r in rows}) == 1
        calls.append(("freeze", len(rows)))
        policies = {}
        for far, occupancy in PRESETS:
            value = copy.deepcopy(config)
            value["alert"].update(
                threshold=1.0,
                max_burden_occupancy=occupancy,
                max_eligible_negative_burden_occupancy=occupancy,
            )
            policies[f"far_{far:g}_occupancy_{occupancy:g}"] = value
        legacy = copy.deepcopy(config)
        for key in [
            "policy_version",
            "max_duration_seconds",
            "rearm_seconds",
            "max_burden_occupancy",
            "max_eligible_negative_burden_occupancy",
        ]:
            legacy["alert"].pop(key)
        legacy["alert"]["threshold"] = 1.0
        policies["legacy_bridge"] = legacy
        return policies, {k: [] for k in policies}

    monkeypatch.setattr(study, "freeze_policy_frontiers", freeze, raising=False)
    return SimpleNamespace(f=f, prepared=p, support=support, calls=calls)


def execute(h, output):
    return study.execute_study(
        h.prepared,
        h.support,
        output,
        guard=lambda: None,
        storage_check=h.f.storage,
        deadline=time.monotonic() + 60,
    )


def test_execution_freezes_all_six_before_outer_and_persists_complete_bound_artifacts(tmp_path, monkeypatch):
    h = harness(tmp_path / "fixture", monkeypatch)
    real_score = __import__("cardiotwin.ecgfounder_compare", fromlist=["score_rows"]).score_rows

    def score(head, calibrator, rows, data, indices):
        indices = list(indices)
        if len({rows[i]["group_id"] for i in indices}) >= 2:
            assert sum(v[0] == "freeze" for v in h.calls) == 6
        yield from real_score(head, calibrator, rows, data, indices)

    monkeypatch.setattr(study, "score_rows", score, raising=False)
    try:
        receipt = execute(h, tmp_path / "run")
        assert receipt["status"] == "complete" and receipt["completed_arms"] == 6
        assert receipt["models_completed"] == 66 and receipt["calibrators_completed"] == 6
        assert receipt["decision"] == "reject" and not receipt["full_run_eligible"]
        manifest = json.loads((tmp_path / "run/artifact-manifest.json").read_text())
        assert len([n for n in manifest["files_sha256"] if n.endswith(".joblib")]) == 66
        actual = {str(p.relative_to(tmp_path / "run")) for p in (tmp_path / "run").rglob("*") if p.is_file()}
        assert actual == set(manifest["files_sha256"]) | {"receipt.json", "artifact-manifest.json"}
        for name, digest in manifest["files_sha256"].items():
            assert preparation.sha(tmp_path / "run" / name) == digest
        summary = study._read_report(tmp_path / "run/summary.json.gz")
        assert {n for n in actual if n.endswith(".json.gz")} == {
            "summary.json.gz",
            *(f"fold-{f}-{a}-receipt.json.gz" for f in range(3) for a in study.STUDY_ARMS),
        }
        assert len(summary["folds"]) == 6
        assert all(v["event_denominator"] == 40 for v in summary["comparisons"].values())
        for arm in study.STUDY_ARMS:
            indices = []
            for entry in summary["folds"]:
                if entry["arm"] == arm:
                    with gzip.open(tmp_path / "run" / entry["predictions"], "rt") as handle:
                        predictions = [json.loads(line) for line in handle]
                    indices.extend(r["row_index"] for r in predictions)
            assert sorted(indices) == list(range(len(h.prepared.rows)))
    finally:
        h.prepared.close()


@pytest.mark.parametrize("fault", ["support", "stage", "deadline", "finite_policy"])
def test_execution_prerequisite_failure_starts_no_fit(tmp_path, monkeypatch, fault):
    h = harness(tmp_path / "fixture", monkeypatch)
    if fault == "support":
        h.support["cells"][0]["observable_events"] -= 1
    elif fault == "stage":
        h.prepared.receipt["allowed_stages"] = ["preflight", "support"]
    elif fault == "finite_policy":
        h.prepared.config["alert"].pop("policy_version")
    try:
        with pytest.raises((ValueError, TimeoutError)):
            if fault == "deadline":
                study.execute_study(
                    h.prepared,
                    h.support,
                    tmp_path / "run",
                    guard=lambda: None,
                    storage_check=h.f.storage,
                    deadline=time.monotonic() - 1,
                )
            else:
                execute(h, tmp_path / "run")
        assert not any(v[0] == "primary" for v in h.calls)
    finally:
        h.prepared.close()


def test_execution_original_fit_failure_saves_partial_and_never_scores_outer(tmp_path, monkeypatch):
    h = harness(tmp_path / "fixture", monkeypatch)

    def failed(*a):
        raise RuntimeError("original synthetic fit failure")

    monkeypatch.setattr(study, "fit_primary_heads", failed, raising=False)
    try:
        with pytest.raises(RuntimeError, match="original synthetic fit failure"):
            execute(h, tmp_path / "run")
        receipt = json.loads((tmp_path / "run/receipt.json").read_text())
        assert receipt["status"] == "failed" and receipt["original_error"] == "original synthetic fit failure"
        assert receipt["completed_arms"] == 0 and receipt["models_completed"] == 0
        assert not (tmp_path / "run/artifact-manifest.json").exists()
    finally:
        h.prepared.close()


def test_wrong_copied_source_bytes_stop_before_fitting(tmp_path, monkeypatch):
    h = harness(tmp_path / "fixture", monkeypatch)
    from pathlib import Path

    read = Path.read_bytes
    changed = h.f.root / next(iter(h.prepared.source))
    monkeypatch.setattr(Path, "read_bytes", lambda p: b"wrong copied source" if p == changed else read(p))
    try:
        with pytest.raises(ValueError, match="source"):
            execute(h, tmp_path / "run")
        assert not any(v[0] == "primary" for v in h.calls)
    finally:
        h.prepared.close()


def test_common_abstention_preserves_original_row_and_label(tmp_path, monkeypatch):
    h = harness(tmp_path / "fixture", monkeypatch, unavailable=[0])
    try:
        execute(h, tmp_path / "run")
        summary = study._read_report(tmp_path / "run/summary.json.gz")
        found = []
        for entry in summary["folds"]:
            with gzip.open(tmp_path / "run" / entry["predictions"], "rt") as handle:
                found.extend(json.loads(v) for v in handle if json.loads(v)["row_index"] == 0)
        assert len(found) == 2
        assert all(v["original_accepted"] and not v["accepted"] and v["label"] == 0 for v in found)
        assert all(v["risk"] is None and v["raw_logit"] is None for v in found)
        for arm in study.STUDY_ARMS:
            availability = summary["availability"][arm]
            assert availability["original_accepted_rows"] == len(h.prepared.rows)
            assert availability["common_available_rows"] == len(h.prepared.rows) - 1
            assert availability["newly_unavailable_rows"] == 1
    finally:
        h.prepared.close()


@pytest.mark.parametrize("fault", ["label", "missing", "duplicate", "nonfinite"])
def test_scored_metadata_corruption_or_incomplete_chronology_cannot_complete(tmp_path, monkeypatch, fault):
    h = harness(tmp_path / "fixture", monkeypatch)
    original = study.score_rows

    def corrupt(*args):
        predictions = original(*args)
        first = next(predictions)
        if fault == "label":
            first["label"] = 1 - first["label"]
        elif fault == "nonfinite":
            first["risk"] = float("nan")
        if fault != "missing":
            yield first
        if fault == "duplicate":
            yield first
        yield from predictions

    monkeypatch.setattr(study, "score_rows", corrupt)
    try:
        with pytest.raises(ValueError):
            execute(h, tmp_path / "run")
        receipt = json.loads((tmp_path / "run/receipt.json").read_text())
        assert receipt["status"] == "failed" and receipt["completed_arms"] == 0
    finally:
        h.prepared.close()


def test_manifest_write_error_preserves_original_failure(tmp_path, monkeypatch):
    h = harness(tmp_path / "fixture", monkeypatch)
    write = study._json_artifact

    def fail_manifest(output, name, *args):
        if name == "artifact-manifest.json":
            raise OSError("original synthetic manifest failure")
        return write(output, name, *args)

    monkeypatch.setattr(study, "_json_artifact", fail_manifest)
    try:
        with pytest.raises(OSError, match="original synthetic manifest failure"):
            execute(h, tmp_path / "run")
        receipt = json.loads((tmp_path / "run/receipt.json").read_text())
        assert receipt["status"] == "failed" and receipt["completed_arms"] == 6
        assert receipt["original_error"] == "original synthetic manifest failure"
    finally:
        h.prepared.close()


def test_support_reports_actual_two_table_hash_passes_and_bytes(tmp_path, monkeypatch):
    h = harness(tmp_path / "fixture", monkeypatch)
    verify = study._verify
    measured = []

    def record(bindings, root, guard):
        if all(str(name).endswith(("/delta.npy", "/nuisance.npy", "/rows.npy")) for name in bindings):
            measured.append(sum((root / name).stat().st_size for name in bindings))
        return verify(bindings, root, guard)

    monkeypatch.setattr(study, "_verify", record)
    try:
        result = study.audit_common_support(h.prepared, guard=lambda: None)
        assert len(measured) == 2
        assert result["hash_validation"] == dict(whole_table_sha256_passes=2, bytes=sum(measured))
    finally:
        h.prepared.close()


@pytest.mark.parametrize("fault", ["changed_model", "extra_file", "late_changed_model", "missing_model"])
def test_final_artifact_verification_rejects_drift_or_unexpected_members(tmp_path, monkeypatch, fault):
    h = harness(tmp_path / "fixture", monkeypatch)
    write = study._json_artifact
    report_write = study._report_artifact

    def drift(output, name, *args):
        result = (report_write if name.endswith(".json.gz") else write)(output, name, *args)
        if name == ("artifact-manifest.json" if fault == "late_changed_model" else "summary.json.gz"):
            if fault in ("changed_model", "late_changed_model"):
                next(output.glob("*.joblib")).write_bytes(b"changed fitted artifact")
            elif fault == "missing_model":
                next(output.glob("*.joblib")).unlink()
            else:
                (output / "unexpected.txt").write_text("not produced by declared study")
        return result

    monkeypatch.setattr(study, "_json_artifact", drift)
    monkeypatch.setattr(study, "_report_artifact", drift)
    try:
        with pytest.raises(ValueError, match="artifact|Artifact"):
            execute(h, tmp_path / "run")
        receipt = json.loads((tmp_path / "run/receipt.json").read_text())
        assert receipt["status"] == "failed"
    finally:
        h.prepared.close()


def test_manifest_failure_keeps_checkpoint_incomplete(tmp_path, monkeypatch):
    h = harness(tmp_path / "fixture", monkeypatch)
    write = study._json_artifact

    def fail_manifest(output, name, *args):
        if name == "artifact-manifest.json":
            raise OSError("synthetic finalization failure")
        return write(output, name, *args)

    monkeypatch.setattr(study, "_json_artifact", fail_manifest)
    try:
        with pytest.raises(OSError, match="synthetic finalization failure"):
            execute(h, tmp_path / "run")
        checkpoint = json.loads((tmp_path / "run/checkpoint.json").read_text())
        assert checkpoint["status"] != "complete"
    finally:
        h.prepared.close()


def test_nuisance_package_binds_its_extra_numeric_transform(tmp_path):
    transforms = SyntheticTransforms([])
    head = SyntheticHead(transforms, "nuisance", "fit_stop_only", {"fit", "stop"})
    study._package_artifact(
        tmp_path,
        "nuisance.joblib",
        head,
        None,
        {},
        1_000_000,
        lambda: dict(allocated_bytes=0, logical_bytes=0),
    )
    manifest = json.loads((tmp_path / "nuisance.joblib.json").read_text())
    assert manifest["nuisance_transform"] == dict(
        imputer_statistics=[0.0] * 6,
        scaler_mean=[0.0] * 6,
        scaler_var=[1.0] * 6,
        scaler_scale=[1.0] * 6,
    )


def test_identity_variance_includes_unknown_natural_rows_and_static_person_ties():
    prepared = SimpleNamespace(
        rows=[dict(group_id=g, label=label) for g, label in (("a", -1), ("a", 0), ("b", -1), ("b", 1))],
        tables=SimpleNamespace(rows=np.asarray([(True,)] * 4, dtype=[("available", "?")])),
        features=None,
    )
    head = SimpleNamespace(logits=lambda features, data, indices: np.asarray([2.0, 2.0, 7.0, 7.0])[indices])
    result = study._identity_variance(head, prepared, np.arange(4), lambda: None)
    assert [v["count"] for v in result["people"]] == [2, 2]
    assert result["equal_person_within_variance"] == 0
    assert result["between_person_mean_variance"] == 6.25
    assert result["includes_unknown_labels"]


def test_negative_candidate_calibration_slope_blocks_otherwise_positive_advancement():
    comparisons = {
        name: dict(
            sensitivity_delta=0.1,
            leave_one_person_out=[dict(sensitivity_delta=0.1)],
            bootstrap=dict(event_sensitivity_delta=dict(low=0.01)),
        )
        for name in study.POLICY_NAMES
    }
    entries = [
        dict(
            fold=fold,
            arm=arm,
            reports={study.PRIMARY_POLICY: dict(event_sensitivity=0.1 * position)},
            calibration=dict(raw_score_slope=-1 if fold == 1 and position == 1 else 1),
            primary_constraints=dict(passed=True),
        )
        for fold in range(3)
        for position, arm in enumerate(study.STUDY_ARMS)
    ]
    result = study._advancement(entries, comparisons)
    assert result["decision"] == "reject"
    assert result["failed_advancement_criteria"] == ["nonpositive_candidate_calibration_slope"]
