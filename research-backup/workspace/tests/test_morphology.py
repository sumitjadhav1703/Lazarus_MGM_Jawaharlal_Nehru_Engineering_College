"""Causal, nuisance-resistant shape proxies; no physiological efficacy claim."""

import importlib.util
import json

import numpy as np
import pytest


def api(name):
    assert importlib.util.find_spec("cardiotwin.morphology"), "morphology extractor missing"
    from cardiotwin import morphology

    assert hasattr(morphology, name), f"missing API {name}"
    return getattr(morphology, name)


def fixture(*, bump="none"):
    fs = 200
    t = np.arange(120000) / fs
    phase = (t + .5) % 1 - .5
    qrs = np.exp(-.5 * (phase / .012)**2) - .25 * np.exp(-.5 * ((phase-.04) / .014)**2)
    pre = .10 * np.exp(-.5 * ((phase+.16) / .025)**2)
    wave = qrs + pre
    if bump != "none":
        # Fixed analytic beat's detrended Q-region P95-P5; no production extractor call.
        anchor_scale = 1.141087430259321
        shape = .2 * anchor_scale * np.maximum(0, 1-np.abs(phase+.16)/.04)
        # Halves are assigned by beat anchor; the first recent beat starts before300s.
        active = np.rint(t) >= 300
        if bump == "alternating":
            active &= (np.rint(t).astype(int) % 2) == 0
        wave = wave + shape * active
    return np.column_stack((wave, -.8 * wave)), np.arange(1, 600, dtype=float)


@pytest.mark.parametrize("anchor_mode", ["independent", "shared-channel0-v1"])
def test_known_anchor_fixture_matches_generating_landmarks(anchor_mode):
    signal, landmarks = fixture()
    result = api("extract_morphology")(signal, 200, anchor_mode=anchor_mode)
    assert all(value is not None for value in result["features"].values())
    for channel in result["channels"]:
        anchors = np.asarray(channel["anchor_samples"]) / 200
        errors = np.abs(anchors[:, None] - landmarks).min(axis=1)
        assert np.mean(errors <= .02) >= .95
        assert np.mean(np.abs(landmarks[:, None]-anchors).min(axis=1) <= .02) >= .95


@pytest.mark.parametrize("gain,offset,drift", [(1, 5, 0), (1, -5, 0), (.5, 0, 0),
                                                (2, 0, 0), (-1, 0, 0), (1, 0, .5)])
@pytest.mark.parametrize("anchor_mode", ["independent", "shared-channel0-v1"])
def test_declared_nuisance_transforms_preserve_measurements(gain, offset, drift, anchor_mode):
    signal, _ = fixture()
    extract = api("extract_morphology")
    base = extract(signal, 200, anchor_mode=anchor_mode)
    scale = np.percentile(signal, 95, axis=0)-np.percentile(signal, 5, axis=0)
    changed = gain*signal + offset*scale + np.linspace(0, drift, len(signal))[:, None]*scale
    result = extract(changed, 200, anchor_mode=anchor_mode)
    assert result["available"] == base["available"]
    for name, value in base["features"].items():
        assert abs(value-result["features"][name]) <= .02


@pytest.mark.parametrize("anchor_mode", ["independent", "shared-channel0-v1"])
def test_recent_shape_change_is_preserved(anchor_mode):
    extract = api("extract_morphology")
    original = extract(fixture()[0], 200, anchor_mode=anchor_mode)["features"]
    consistent = extract(fixture(bump="all")[0], 200, anchor_mode=anchor_mode)["features"]
    alternating = extract(fixture(bump="alternating")[0], 200, anchor_mode=anchor_mode)["features"]
    for c in range(2):
        assert consistent[f"morph_c{c}_pre_template_change"]-original[f"morph_c{c}_pre_template_change"] >= .02
        assert alternating[f"morph_c{c}_pre_dispersion_recent"]-original[f"morph_c{c}_pre_dispersion_recent"] >= .02


@pytest.mark.parametrize("anchor_mode", ["independent", "shared-channel0-v1"])
def test_future_perturbation_cannot_change_history(anchor_mode):
    signal, _ = fixture()
    extended = np.vstack((signal, np.zeros((400, 2))))
    extract = api("extract_morphology")
    before = extract(extended, 200, decision_sample=120000, anchor_mode=anchor_mode)
    extended[120000:] = 1e9
    assert extract(extended, 200, decision_sample=120000, anchor_mode=anchor_mode) == before


@pytest.mark.parametrize("anchor_mode", ["independent", "shared-channel0-v1"])
def test_missing_channel_stays_missing_after_zero_fill_with_mask(anchor_mode):
    signal, _ = fixture()
    signal[:, 1] = np.nan
    valid = np.isfinite(signal)
    extract = api("extract_morphology")
    missing = extract(signal, 200, anchor_mode=anchor_mode)
    filled = extract(np.nan_to_num(signal), 200, validity=valid, anchor_mode=anchor_mode)
    assert not missing["available"] and missing["features"] == filled["features"]
    assert all(v is None for k, v in filled["features"].items() if k.startswith("morph_c1"))
    assert all(v is not None for k, v in filled["features"].items() if k.startswith("morph_c0"))


@pytest.mark.parametrize("anchor_mode", ["independent", "shared-channel0-v1"])
def test_constant_shape_has_no_temporal_change(anchor_mode):
    result = api("extract_morphology")(fixture()[0], 200, anchor_mode=anchor_mode)
    for key, value in result["features"].items():
        if "change" in key:
            assert abs(value) < 1e-9


def test_measurement_receipt_is_json_serializable():
    report = api("extract_morphology")(fixture()[0], 200, overlays=True)
    assert json.loads(json.dumps(report, allow_nan=False))["available"] is True


@pytest.mark.parametrize("extreme", [False, True])
def test_unavailable_history_diagnostics_are_strict_json(extreme):
    signal = np.full((120000, 2), np.nan)
    signal[60000] = 1
    if extreme:
        signal[::2], signal[1::2] = -1e308, 1e308
    with np.errstate(over="ignore", invalid="ignore"):
        report = api("extract_morphology")(signal, 200, overlays=True)
    restored = json.loads(json.dumps(report, allow_nan=False))
    assert not restored["available"]
    assert all(value is None for value in restored["features"].values())
    for channel in restored["channels"]:
        assert channel["history_scale" if extreme else "flat_fraction"] is None


def test_flat_or_insufficient_beat_half_is_missing():
    extract = api("extract_morphology")
    assert all(v is None for v in extract(np.zeros((120000, 2)), 200)["features"].values())
    signal, _ = fixture()
    signal[60000:] = 0
    assert all(v is None for v in extract(signal, 200)["features"].values())


@pytest.mark.parametrize("signal,fs,decision", [(np.zeros((119999, 2)), 200, None),
    (np.zeros((120000, 1)), 200, None), (np.zeros((120000, 2)), 250, None),
    (np.zeros((120000, 2)), 200, 120000.5)])
def test_invalid_history_contract_is_rejected(signal, fs, decision):
    with pytest.raises(ValueError):
        api("extract_morphology")(signal, fs, decision_sample=decision)


def test_engineering_selection_is_fit_only_label_blind_and_deterministic():
    rows = [dict(group_id=f"g{g:02}", patient_id=f"g{g:02}", record_id=f"r{g:02}",
                 timestamp=t, accepted=True, label=g % 2) for g in range(16) for t in (630., 660.)]
    roles = [{"fold": n, "fit": [f"g{g:02}" for g in range(15)], "outer": ["g15"]} for n in range(3)]
    select = api("select_audit_rows")
    before = select(rows, roles)
    for r in rows:
        r["label"] = None
    assert before == select(rows, roles)
    assert len(before) == 36
    assert all(item["group_id"] != "g15" for item in before)
    for fold in range(3):
        assert len({r["group_id"] for r in before if r["fold"] == fold}) == 12


def test_engineering_gate_cannot_average_away_failed_fold():
    gate = api("measurement_gate")
    assert gate([12, 12, 9])["status"] == "failed"
    assert gate([11, 11, 11])["status"] == "passed"


def test_measurements_use_saved_identities_exact_past_and_keep_failure(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from cardiotwin import morphology

    selected = [dict(fold=f, record_id="r", timestamp=630.+30*j, row_index=f*12+j)
                for f in range(3) for j in range(12)]
    (tmp_path / "selected.json").write_text(json.dumps(selected))
    calls = []

    def reader(record, start, end):
        assert json.loads((tmp_path / "selected.json").read_text()) == selected
        calls.append((start, end))
        return np.zeros((120000, 2))

    def extract(signal, fs, *, overlays):
        assert fs == 200 and overlays and signal.shape == (120000, 2)
        return dict(available=len(calls) != 1, features={}, channels=[])

    monkeypatch.setattr(morphology, "extract_morphology", extract)
    report = api("measure_histories")(selected, {"r": SimpleNamespace(fs=200)}, reader, tmp_path)
    assert calls == [(r["timestamp"]-600, r["timestamp"]) for r in selected]
    assert report["valid_by_fold"] == [11, 12, 12]
    saved = [json.loads(line) for line in (tmp_path / "measurements.jsonl").read_text().splitlines()]
    assert len(saved) == 36 and saved[0]["row_index"] == 0 and not saved[0]["measurement"]["available"]
    with pytest.raises(FileExistsError):
        api("measure_histories")(selected, {"r": SimpleNamespace(fs=200)}, reader, tmp_path)


def test_measurements_refuse_unsaved_or_changed_selection(tmp_path):
    selected = [dict(fold=0, record_id="r", timestamp=630.)]
    with pytest.raises((FileNotFoundError, ValueError)):
        api("measure_histories")(selected, {}, None, tmp_path)
    (tmp_path / "selected.json").write_text("[]")
    with pytest.raises(ValueError, match="selection"):
        api("measure_histories")(selected, {}, None, tmp_path)


def test_runner_uses_original_canonical_roles_hash_before_raw_reads(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from cardiotwin.data import file_hash
    from cardiotwin.investigation import _digest

    module_spec = importlib.util.spec_from_file_location("morphology_audit", "scripts/run_morphology_audit.py")
    runner = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(runner)
    audit = tmp_path / "audit"
    audit.mkdir()
    roles = [dict(fold=f, fit=["g"]) for f in range(3)]
    (audit / "roles.json").write_text(json.dumps(dict(cache_index_sha256="index", roles=roles), indent=2))
    (audit / "pairs.jsonl").write_text("[]\n")
    (audit / "receipt.json").write_text(json.dumps(dict(status="passed", metadata_status="passed",
        pair_support_status="passed", cache_index_sha256="index", roles_sha256=_digest(roles),
        pairs_sha256=file_hash(audit / "pairs.jsonl"))))
    validation = tmp_path / "validation.json"
    binding = {"index_sha256": "index"}
    validation.write_text(json.dumps(dict(status="passed", binding=binding)))
    spec = tmp_path / "spec.md"
    spec.write_text("fixture")
    record = dict(record_id="r", patient_id="p", path="/not-read", fs=200, duration_seconds=1000,
                  annotated=True, clinical={}, metadata_audit=[])
    monkeypatch.setattr(runner, "compact_cache_binding", lambda *a, **kw: binding)
    monkeypatch.setattr(runner, "read_cache", lambda *a: ([], [], {"records": [dict(record=record, input_contract={})]}))
    monkeypatch.setattr(runner, "select_audit_rows", lambda *a: [dict(record_id="r", fold=f) for f in range(3)])

    def stop_at_raw(record):
        assert (tmp_path / "out" / "selected.json").is_file()
        assert (tmp_path / "out" / "predeclaration.json").is_file()
        raise RuntimeError("reached raw preflight")

    monkeypatch.setattr(runner, "_record_contract", stop_at_raw)
    args = SimpleNamespace(output=tmp_path / "out", audit=audit, source_validation=validation,
                           cache="not-read", spec=spec, max_seconds=1200)
    with pytest.raises(RuntimeError, match="reached raw preflight"):
        runner.run(args)
    receipt = json.loads((tmp_path / "out" / "receipt.json").read_text())
    assert receipt["error"] == "reached raw preflight" and receipt["fits"] == 0
    with pytest.raises(ValueError, match="output already exists"):
        runner.run(args)


def shared_fixture(*, distractors="none", noise=False):
    t = np.arange(120000)/200

    def phase(center):
        return (t-center+.5) % 1-.5
    signal = np.column_stack((np.exp(-.5*(phase(0)/.015)**2),
                              -.5*np.exp(-.5*(phase(.02)/.015)**2)))
    broad = .6*np.exp(-.5*(phase(.3)/.08)**2)+.4*np.exp(-.5*(phase(.55)/.08)**2)
    if distractors != "none":
        signal[:, 1] += broad
    if distractors == "both":
        signal[:, 0] += broad
    if noise:
        signal += np.random.default_rng(42).normal(0, .01, signal.shape)
    return signal


@pytest.mark.parametrize("distractors", ["none", "secondary", "both"])
def test_shared_reference_known_landmarks_with_fixed_distractors(distractors):
    extract = api("extract_morphology")
    report = extract(shared_fixture(distractors=distractors), 200, anchor_mode="shared-channel0-v1", overlays=True)
    assert report["available"]
    anchors = np.asarray(report["channels"][0]["anchor_samples"])/200
    expected = np.arange(1, 600)
    assert np.mean(np.abs(anchors[:, None]-expected).min(axis=1) <= .02) >= .95
    assert np.mean(np.abs(expected[:, None]-anchors).min(axis=1) <= .02) >= .95
    assert report["channels"][1]["anchor_samples"] == report["channels"][0]["anchor_samples"]
    for channel in report["channels"]:
        assert len(channel["overlays"]) >= 6
        assert any(o.get("selection") == "lowest_concentration" for o in channel["overlays"])
        for half in channel["quality_halves"]:
            assert half["concentrated_fraction"] >= .9
            assert half["scaled_fraction"] >= .95


def test_shared_reference_noise_records_failed_original_invariance_limit():
    extract = api("extract_morphology")
    base = extract(shared_fixture(distractors="both"), 200, anchor_mode="shared-channel0-v1")
    report = extract(shared_fixture(distractors="both", noise=True), 200, anchor_mode="shared-channel0-v1")
    assert base["available"] and report["available"]
    anchors = np.asarray(report["channels"][0]["anchor_samples"])/200
    expected = np.arange(1, 600)
    assert np.mean(np.abs(anchors[:, None]-expected).min(axis=1) <= .02) >= .95
    assert np.mean(np.abs(expected[:, None]-anchors).min(axis=1) <= .02) >= .95
    failed = [name for name, value in base["features"].items() if abs(report["features"][name]-value) > .02]
    assert failed == ["morph_c1_pre_dispersion_recent", "morph_c1_anchor_dispersion_recent"]


@pytest.mark.parametrize("kind", ["flat", "masked", "noise"])
def test_shared_reference_rejects_bad_secondary_without_switching(kind):
    extract = api("extract_morphology")
    signal = shared_fixture()
    original = extract(signal, 200, anchor_mode="shared-channel0-v1")
    signal[:, 1] = {"flat": 0., "masked": np.nan, "noise": 0.}[kind]
    if kind == "noise":
        signal[:, 1] = np.random.default_rng(43).normal(0, .1, len(signal))
    report = extract(signal, 200, anchor_mode="shared-channel0-v1", overlays=True)
    assert not report["available"] and report["channels"][0]["available"]
    assert report["channels"][0]["anchor_samples"] == original["channels"][0]["anchor_samples"]
    assert all(value is None for name, value in report["features"].items() if name.startswith("morph_c1"))
    json.dumps(report, allow_nan=False)


def test_shared_reference_missing_primary_never_falls_back_and_masks_survive():
    signal = shared_fixture()
    signal[:, 0] = np.nan
    extract = api("extract_morphology")
    missing = extract(signal, 200, anchor_mode="shared-channel0-v1")
    filled = extract(np.nan_to_num(signal), 200, validity=np.isfinite(signal), anchor_mode="shared-channel0-v1")
    assert not missing["available"] and missing["features"] == filled["features"]
    assert all(v is None for v in missing["features"].values())
