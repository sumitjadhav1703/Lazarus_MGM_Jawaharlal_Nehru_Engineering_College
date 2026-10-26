import copy

import numpy as np
import pytest

from cardiotwin.alerts import advance_alert
from cardiotwin.config import load_config
from cardiotwin.features import extract_features, feature_row, update_baseline


def ecg(seconds=600, fs=100):
    t = np.arange(seconds * fs) / fs
    phase = np.mod(t, 0.8)
    lead = np.exp(-(((phase - 0.15) / 0.018) ** 2)) + 0.002 * np.sin(2 * np.pi * 17 * t)
    return np.column_stack([lead, 0.8 * lead]), fs


def test_feature_prefix_invariance_and_prior_baseline():
    config = load_config()
    signal, fs = ecg()
    future = np.full((1000, 2), 300.0)
    a = extract_features(signal, fs)
    b = extract_features(np.vstack([signal, future])[: len(signal)], fs)
    assert a == b
    assert 70 < a["ecg_hr"] < 80
    baseline = {}
    row, quality = feature_row(signal, fs, {"age": 62, "AF_Type": 2}, baseline, config)
    assert quality["accepted"]
    assert "clinical_AF_Type" not in row
    assert row["baseline_count"] == 0
    after = update_baseline(baseline, row, config)
    assert baseline == {}
    row2, _ = feature_row(signal, fs, {}, after, config)
    assert row2["baseline_ecg_hr_deviation"] == pytest.approx(0)


def test_unreliable_and_nonfinite_ecg_rejected():
    c = load_config()
    for signal in [np.zeros((10000, 2)), np.full((10000, 2), np.nan)]:
        _, quality = feature_row(signal, 100, {}, {}, c)
        assert not quality["accepted"]
    with pytest.raises(ValueError):
        extract_features(np.ones((10, 2)), 0)


def test_alert_persistence_hysteresis_cooldown_and_quality():
    c = load_config()
    s = {}
    for t in [600, 630]:
        s, w = advance_alert(s, t, 0.9, True, c)
        assert w is None
    s, w = advance_alert(s, 660, 0.9, True, c)
    assert w["timestamp"] == 660
    s, w = advance_alert(s, 690, 0.7, True, c)
    assert s["active"] and w is None
    s, w = advance_alert(s, 720, None, False, c)
    assert not s["active"] and w["end_time"] == 720
    for t in [750, 780, 810]:
        s, w = advance_alert(s, t, 0.9, True, c)
        assert w is None
    assert s["suppressed_count"] > 0
    restored = copy.deepcopy(s)
    assert advance_alert(s, 840, 0.9, True, c) == advance_alert(restored, 840, 0.9, True, c)


def test_alert_gap_breaks_persistence():
    c = load_config()
    s, _ = advance_alert({}, 600, 0.9, True, c)
    s, w = advance_alert(s, 690, 0.9, True, c)
    assert w is None and s["high_since"] == 690


def test_invalid_config_rejected(tmp_path):
    import json

    c = load_config()
    c["target"]["gap_seconds"] = 700
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(c))
    with pytest.raises(ValueError):
        load_config(path)


def test_persisted_replay_is_deterministic_and_future_blind(tmp_path):
    from cardiotwin.data import make_demo
    from cardiotwin.service import TwinService

    c = load_config()
    root = tmp_path / "data"
    make_demo(root, c)
    service = TwinService(root, tmp_path / "state.sqlite", c)
    pid = service.patients()[1]["patient_id"]
    first = service.step(pid, 26)
    assert first["timestamp"] == 780
    assert first["forecast"]["reason"] == "model_not_loaded"
    restarted = TwinService(root, tmp_path / "state.sqlite", c)
    assert restarted.state(pid)["baseline"] == first["baseline"]
    a = service.step(pid)
    b = restarted.step(pid)
    # Each service re-reads persisted cursor, so sequential clients cannot repeat a frame.
    assert b["timestamp"] == a["timestamp"] + 30
    restarted.control(pid, "reset")
    reproduced = restarted.step(pid, 26)
    assert reproduced["features"] == first["features"]
    assert reproduced["baseline"] == first["baseline"]
    assert reproduced["forecast"] == first["forecast"]


def test_changed_future_ecg_and_annotations_cannot_change_earlier_risk(tmp_path):
    import json

    from sklearn.linear_model import LogisticRegression

    from cardiotwin.modeling import Predictor, build_pipeline, matrix
    from cardiotwin.service import TwinService
    from cardiotwin.train import _save_predictor

    config = load_config()
    config["alert"].update(threshold=0.1, release_threshold=0.05)
    signal, fs = ecg(seconds=1500)
    artifacts = tmp_path / "model"
    artifacts.mkdir()
    pipeline = build_pipeline("logistic", {"C": 1}, 42)
    pipeline.fit(matrix([{"ecg_hr": v} for v in [60, 65, 70, 80, 85, 90]], ["ecg_hr"]), [0, 0, 0, 1, 1, 1])
    calibrator = LogisticRegression().fit(np.array([[-2], [-1], [1], [2]]), [0, 0, 1, 1])
    predictor = Predictor(pipeline, calibrator, ["ecg_hr"], config, {"version": "causality-fixture"})
    _save_predictor(predictor, artifacts, config, {"version": "causality-fixture", "synthetic": True}, {})
    states = []
    for variant in range(2):
        root = tmp_path / f"input{variant}"
        root.mkdir()
        samples = signal.copy()
        if variant:
            samples[900 * fs :] = 0
        np.savez(root / "record.npz", signal=samples, fs=fs)
        manifest = {
            "synthetic": True,
            "records": [
                {
                    "record_id": "demo-prefix",
                    "patient_id": "prefix-subject",
                    "path": "record.npz",
                    "fs": fs,
                    "duration_seconds": 1500,
                    "annotated": True,
                    "clinical": {},
                    "metadata_audit": [],
                    "signal_hash": None,
                    "start_time": None,
                    "intervals": [{"start": 0, "end": 1500, "rhythm": "AF" if variant else "other"}],
                }
            ],
        }
        (root / "manifest.json").write_text(json.dumps(manifest))
        service = TwinService(root, tmp_path / f"state{variant}.sqlite", config, artifacts)
        states.append(service.step("prefix-subject", 30))
    assert states[0]["forecast"]["available"]
    assert states[0]["forecast"] == states[1]["forecast"]
    assert states[0]["features"] == states[1]["features"]
    assert states[0]["baseline"] == states[1]["baseline"]
    assert states[0]["warnings"] == states[1]["warnings"]
    assert len(states[0]["warnings"]) == 1


def test_restart_rejects_changed_baseline_rule(tmp_path):
    from cardiotwin.data import make_demo
    from cardiotwin.service import TwinService

    config = load_config()
    make_demo(tmp_path / "data", config)
    TwinService(tmp_path / "data", tmp_path / "state.sqlite", config)
    changed = copy.deepcopy(config)
    changed["baseline"]["max_updates"] = 120
    with pytest.raises(ValueError, match="Inputs/config/model changed"):
        TwinService(tmp_path / "data", tmp_path / "state.sqlite", changed)


def finite_config():
    return load_config("configs/precursor-cycle.json")


def test_expiry_does_not_hide_high_risk():
    import cardiotwin.alerts as alerts

    cfg, state, starts = finite_config(), {}, []
    for t in range(0, 1230, 30):
        state, warning = advance_alert(state, t, .99, True, cfg)
        if warning and warning["end_time"] is None:
            starts.append(warning["timestamp"])
    assert starts == [60]
    assert state["active"] is False
    assert state["awaiting_rearm"] is True
    seconds = alerts.alert_interval_seconds(state, 1200, 1215, cfg)
    assert seconds["warning_seconds"] == 0
    assert seconds["unresolved_elevated_seconds"] == 15
    assert seconds["burden_seconds"] == 15


def test_finite_rearm_requires_continuous_strict_low_and_survives_restart(tmp_path):
    from cardiotwin.store import Store

    cfg, state, start = finite_config(), {}, None
    store = Store(tmp_path / "finite.sqlite")
    for t in range(0, 661, 30):
        state, warning = advance_alert(state, t, .99, True, cfg)
        if warning:
            warning["id"] = "same-warning"
            if warning["end_time"] is None:
                warning.update(evidence=["original"], status="reviewed")
                start = warning.copy()
            with store.transaction() as db:
                store.save_in(db, {"patient_id": "p", "_alert": state}, warning)
    closed = store.warnings("p")[0]
    assert closed["end_time"] == 660
    assert closed["id"] == start["id"]
    assert closed["evidence"] == ["original"] and closed["status"] == "reviewed"
    restored = store.read("p")["_alert"]
    assert advance_alert(restored, 690, .99, True, cfg) == advance_alert(state, 690, .99, True, cfg)
    state = restored
    for t, risk, available in [(690, cfg["alert"]["release_threshold"], True), (720, .1, True),
                               (750, .1, True), (780, None, False), (810, .1, True), (870, .1, True), (900, .1, True)]:
        state, _ = advance_alert(state, t, risk, available, cfg)
        assert state["awaiting_rearm"]
    state, _ = advance_alert(state, 930, .1, True, cfg)
    assert state["awaiting_rearm"] is False
    for t in (960, 990, 1020):
        state, warning = advance_alert(state, t, .99, True, cfg)
    assert warning["timestamp"] == 1020


def test_finite_gap_closes_at_last_supported_boundary_and_resets_rearm():
    cfg, state = finite_config(), {}
    for t in (0, 30, 60):
        state, _ = advance_alert(state, t, .99, True, cfg)
    state, warning = advance_alert(state, 180, .1, True, cfg)
    assert warning["end_time"] == 90
    assert state["awaiting_rearm"] and state["low_since"] == 180
    state, _ = advance_alert(state, 210, .1, True, cfg)
    assert state["awaiting_rearm"]
    state, _ = advance_alert(state, 240, .1, True, cfg)
    assert not state["awaiting_rearm"]


def test_finite_policy_config_and_state_mismatch_fail_closed():
    from cardiotwin.config import validate_config

    for key, invalid in [("policy_version", "typo"), ("max_duration_seconds", 29), ("rearm_seconds", 0),
                         ("max_burden_occupancy", float("nan")), ("max_eligible_negative_burden_occupancy", 1.1)]:
        cfg = finite_config()
        cfg["alert"][key] = invalid
        with pytest.raises(ValueError):
            validate_config(cfg)
    cfg = finite_config()
    del cfg["alert"]["rearm_seconds"]
    with pytest.raises(ValueError):
        validate_config(cfg)
    state, _ = advance_alert({}, 0, .99, True, finite_config())
    with pytest.raises(ValueError, match="version"):
        advance_alert(state, 30, .99, True, load_config())
    with pytest.raises(ValueError, match="version"):
        advance_alert({"last_timestamp": 0}, 30, .99, True, finite_config())
