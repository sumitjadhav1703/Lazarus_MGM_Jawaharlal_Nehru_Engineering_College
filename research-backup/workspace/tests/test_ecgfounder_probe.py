import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cardiotwin import ecgfounder_probe as probe


class BlindedRow(Mapping):
    def __init__(self, **values):
        self.values = values

    def __getitem__(self, key):
        assert key in {"group_id", "patient_id", "record_id", "timestamp", "accepted"}
        return self.values[key]

    def __iter__(self):
        raise AssertionError("Outcome-blind selection must not enumerate row fields")

    def __len__(self):
        raise AssertionError("Outcome-blind selection must not inspect full rows")


def selection_fixture():
    records, rows = {}, []
    for i in range(15):
        group = f"g{i:02}"
        for suffix in ("a", "b"):
            record_id = group + suffix
            records[record_id] = dict(record_id=record_id, patient_id=group, group_id=group,
                                      duration_seconds=1800.005, fs=200,
                                      signal_parts=[dict(samples=240001), dict(samples=120000)])
            for timestamp in (630., 870., 930., 1200.):
                rows.append(BlindedRow(group_id=group, patient_id=group, record_id=record_id,
                                       timestamp=timestamp, accepted=True))
    roles = [dict(fold=f, fit=[f"g{i:02}" for i in range(15)]) for f in range(3)]
    return rows, records, roles


def test_selector_is_outcome_blind_hash_ranked_and_keeps_all_slots():
    rows, records, roles = selection_fixture()
    requests = probe.select_probe_requests(rows, records, roles)
    assert len(requests) == 48
    for fold in range(3):
        normal = [r for r in requests if r["fold"] == fold and r["slot"] == "normal"]
        expected = sorted(roles[fold]["fit"],
                          key=lambda g: (hashlib.sha256(f"42|{fold}|{g}".encode()).hexdigest(), g))[:12]
        assert [r["group_id"] for r in normal] == expected
        assert all(r["record_id"] == r["group_id"] + "a" and r["timestamp"] == 930 for r in normal)
        assert all(r["role"] == "fit" and r["status"] == "selected" for r in normal)
    daily = [r for r in requests if r["slot"] == "daily"]
    assert all(r["timestamp"] == 1230.005 and r["decision_sample"] == 246001 for r in daily)
    terminal = [r for r in requests if r["slot"] == "terminal"]
    assert all(r["decision_sample"] == 360001 and not r["synthetic_terminal_required"] for r in terminal)


def test_selector_uses_earlier_midpoint_ties_and_first_qualifying_record():
    rows, records, roles = selection_fixture()
    for record in records.values():
        record["duration_seconds"] = 1800.
    requests = probe.select_probe_requests(rows, records, roles)
    normal = [r for r in requests if r["slot"] == "normal"]
    assert all(r["timestamp"] == 870 for r in normal)
    assert all(r["synthetic_terminal_required"] for r in requests if r["slot"] == "terminal")


def test_selector_fails_instead_of_substituting_outside_fit():
    rows, records, roles = selection_fixture()
    roles[1]["fit"] = roles[1]["fit"][:11]
    with pytest.raises(ValueError, match="12|twelve"):
        probe.select_probe_requests(rows, records, roles)


def test_selector_keeps_structurally_missing_daily_slot_without_replacement():
    rows, records, roles = selection_fixture()
    for record in records.values():
        record["signal_parts"] = [dict(samples=360000)]
    requests = probe.select_probe_requests(rows, records, roles)
    missing = [r for r in requests if r["status"] == "structural_not_applicable"]
    assert len(requests) == 48 and len(missing) == 3
    assert all(r["slot"] == "daily" and r["reason"] for r in missing)


def test_binding_checks_source_changes_and_path_escape_before_processing(tmp_path):
    (tmp_path / "bound.txt").write_text("frozen")
    bound = {"bound.txt": hashlib.sha256(b"frozen").hexdigest()}
    probe.verify_files(bound, root=tmp_path)
    (tmp_path / "bound.txt").write_text("changed")
    with pytest.raises(ValueError, match="binding"):
        probe.verify_files(bound, root=tmp_path)
    with pytest.raises(ValueError, match="path"):
        probe.verify_files({"../outside": "bad"}, root=tmp_path)


def small_model():
    torch = pytest.importorskip("torch")

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(1024, dtype=torch.float32)[None] / 1024)

        def forward(self, values):
            vector = values[:, 0, :1000].mean(-1, keepdim=True) * self.weight
            return vector[:, :1], vector

    return Encoder().eval().requires_grad_(False)


def history(start, end):
    t = np.arange(start, end) / 200
    return np.sin(2 * np.pi * 1.3 * t) + 0.02 * np.cos(2 * np.pi * 29 * t)


def test_bounded_ring_matches_direct_and_reuses_only_exact_same_record_chunks():
    from cardiotwin.ecgfounder import encode_chunks, pool_delta, preprocess_history

    model = small_model()
    ring = probe.HistoryEncoder(model)
    first = ring.encode(history(0, 120000), timestamp=600, record_id="a")
    second = ring.encode(history(6000, 126000), timestamp=630, first_sample=6000, record_id="a")
    direct = pool_delta(encode_chunks(model, preprocess_history(history(6000, 126000), timestamp=630,
                                                               first_sample=6000)))
    assert first["available"] and second["available"]
    assert first["new_chunks"] == 60 and second["new_chunks"] == 3
    assert len(ring.cache) <= 60
    np.testing.assert_allclose(second["delta"], direct, atol=1e-5, rtol=1e-4)
    terminal = ring.encode(history(6001, 126001), timestamp=630.005, first_sample=6001, record_id="a")
    assert terminal["new_chunks"] == 60
    changed_record = ring.encode(history(6001, 126001) * 2, timestamp=630.005,
                                 first_sample=6001, record_id="b")
    assert changed_record["new_chunks"] == 60 and len(ring.cache) == 60


def test_ring_preserves_unavailability_reasons_without_zero_filling():
    ring = probe.HistoryEncoder(small_model())
    values = history(0, 120000)
    values[5999] = np.nan
    result = ring.encode(values, timestamp=600, record_id="a")
    assert not result["available"] and result["delta"] is None
    assert result["rejections"] == {"nonfinite_native": 1}
    assert result["valid_chunks"] == 59


def test_nuisance_features_are_native_all_chunk_means_and_recent_minus_early():
    ring = probe.HistoryEncoder(small_model())
    values = history(0, 120000).reshape(60, 2000)
    values[30:] *= 2
    result = ring.encode(values.ravel(), timestamp=600, record_id="a")
    chunks = np.column_stack([values.mean(1), values.std(1), np.abs(np.diff(values, axis=1)).mean(1)])
    expected = np.r_[chunks.mean(0), chunks[30:].mean(0) - chunks[:30].mean(0)]
    np.testing.assert_allclose(result["nuisance"], expected, atol=1e-7, rtol=1e-6)


def test_cost_gate_requires_every_overhead_component_and_actual_t4_measurement():
    records = [dict(unique_chunks=60, elapsed_seconds=0.5) for _ in range(4)]
    # cost_gate hard-codes a 2026-10-17 12:30 UTC deadline; pin a clock before it so the test is not time-dependent.
    early = datetime(2026, 10, 1, tzinfo=timezone.utc)
    overhead = {name: 1. for name in probe.OVERHEAD_COMPONENTS}
    result = probe.cost_gate(records, overhead, device_name="Tesla T4", now_utc=early)
    assert result["full_run_eligible"]
    assert result["planning_chunks_per_second"] == 120
    assert result["estimated_seconds"] == 1.5 * (1895675 / 120 + len(overhead))
    assert not probe.cost_gate(records, {}, device_name="Tesla T4", now_utc=early)["full_run_eligible"]
    assert not probe.cost_gate(records, overhead, device_name="cpu", now_utc=early)["full_run_eligible"]
    assert not probe.cost_gate(records, {name: 0. for name in probe.OVERHEAD_COMPONENTS},
                               device_name="Tesla T4", now_utc=early)["full_run_eligible"]
    assert not probe.cost_gate(records, overhead, device_name="Tesla T4",
                               now_utc=datetime(2026, 10, 17, 12, 30, tzinfo=timezone.utc))["full_run_eligible"]
    assert not probe.cost_gate([dict(unique_chunks=60, elapsed_seconds=100.)], overhead,
                               device_name="Tesla T4", now_utc=early)["full_run_eligible"]
    with pytest.raises(ValueError):
        probe.cost_gate([dict(unique_chunks=60, elapsed_seconds=float("nan"))], overhead,
                        device_name="Tesla T4", now_utc=early)


def test_resources_enforce_time_rss_device_memory_and_diagnostic_bytes():
    probe.check_resources(elapsed=1, rss=1, reserved=1, output_bytes=1)
    for key, value in (("elapsed", 1801), ("rss", 10000000001), ("reserved", 12000000001),
                       ("output_bytes", 250000001)):
        args = dict(elapsed=1, rss=1, reserved=1, output_bytes=1)
        args[key] = value
        with pytest.raises(RuntimeError, match="resource"):
            probe.check_resources(**args)


def test_manifest_identity_detects_changed_requests(tmp_path):
    manifest = dict(requests=[dict(fold=0, slot="normal", timestamp=630)], inputs={"source": "hash"})
    path = tmp_path / "selected.json"
    path.write_text(json.dumps(manifest))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert probe.read_bound_manifest(path, digest) == manifest
    path.write_text(json.dumps({**manifest, "requests": []}))
    with pytest.raises(ValueError, match="selection"):
        probe.read_bound_manifest(path, digest)


def test_probe_cli_help_needs_no_data_or_weights():
    script = Path(__file__).resolve().parents[1] / "scripts/run_ecgfounder_probe.py"
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": str(script.parents[1] / "src")})
    assert result.returncode == 0, result.stderr
    assert "--selection-sha256" in result.stdout and "--mode" in result.stdout


def test_selection_metadata_reader_never_opens_feature_or_outcome_arrays(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/run_ecgfounder_probe.py"
    spec = importlib.util.spec_from_file_location("ecgfounder_probe_runner", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Features file intentionally absent; only the four allowlisted metadata fields exist.
    values = np.array([(0, 630., 1)], dtype=[("record", "i4"), ("timestamp", "f8"), ("accepted", "i1")])
    np.save(tmp_path / "metadata.npy", values)
    identity = [dict(record_id="r", patient_id="p", group_id="g")]
    rows = module.SelectionRows(tmp_path, identity)
    assert rows[0] == dict(record_id="r", patient_id="p", group_id="g", timestamp=630., accepted=True)
    assert len(rows) == 1


def runner_module():
    script = Path(__file__).resolve().parents[1] / "scripts/run_ecgfounder_probe.py"
    spec = importlib.util.spec_from_file_location("synthetic_ecgfounder_runner", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_runner_setup(tmp_path, monkeypatch, *, fallback=False, mask_fault=None, mask_slot="initial",
                           clear_invalid=False, raise_on=None):
    module = runner_module()
    rows, records, roles = selection_fixture()
    if fallback:
        for record in records.values():
            record["duration_seconds"] = 1800.
            record["signal_parts"] = [dict(samples=360000)]
    manifest = dict(requests=probe.select_probe_requests(rows, records, roles),
                    inputs=dict(source_sha256={}, files_sha256={}, compact_binding={}))
    selection = tmp_path / "selected.json"
    selection.write_text(json.dumps(manifest))
    index = dict(records=[dict(record=dict(record_id=rid, patient_id=r["patient_id"], path="synthetic",
                                           fs=200., duration_seconds=r["duration_seconds"], annotated=False,
                                           clinical={}, metadata_audit=[], source="iridia-af",
                                           signal_parts=r["signal_parts"]), input_contract={})
                          for rid, r in records.items()])
    monkeypatch.setattr(module, "selection_inputs", lambda args: (manifest, index))
    monkeypatch.setattr(module, "load_frozen_encoder", lambda *a, **k: (small_model(), {"weights_only": True}))
    monkeypatch.setattr(module, "_record_contract", lambda record: {})
    monkeypatch.setattr(module, "compact_cache_binding", lambda *a, **k: {})
    monkeypatch.setattr(module, "storage_measurement", lambda: dict(logical_bytes=1, allocated_bytes=1))
    physical = []

    @contextmanager
    def selected_reader():
        def read(record, start, end):
            a, b = round(start * 200), round(end * 200)
            physical.append((record.record_id, a, b))
            if len(physical) == raise_on:
                raise OSError("synthetic source read failed")
            values = history(a, b)
            positions = np.arange(a, b)
            expected = positions < 6000
            if not fallback:
                expected |= (positions >= 240001) & (positions < 246001)
            if mask_fault and b == {"initial": 120000, "daily": 246001}[mask_slot]:
                if mask_fault == "one_nan":
                    values[np.flatnonzero(expected)[0]] = np.nan
                elif mask_fault == "one_finite":
                    values[expected] = np.nan
                    values[np.flatnonzero(expected)[-1]] = 0.123
                elif mask_fault == "wrong_nan":
                    values[np.flatnonzero(~expected)[0]] = np.nan
            else:
                values[expected] = np.nan
            if clear_invalid:
                values[positions == 6000] = np.nan
            return np.column_stack([values, values])
        yield read

    monkeypatch.setattr(module, "cached_signal_reader", selected_reader)
    args = SimpleNamespace(selection=selection,
                           selection_sha256=hashlib.sha256(selection.read_bytes()).hexdigest(),
                           cycle=tmp_path, cache=tmp_path, audit=tmp_path, device="cpu", additional_budget=None)
    return module, args, manifest, physical


def fixture_main(module, args, output):
    return module.main(["--cycle", str(args.cycle), "--cache", str(args.cache), "--audit", str(args.audit),
                        "--output", str(output), "--mode", "probe", "--selection", str(args.selection),
                        "--selection-sha256", args.selection_sha256, "--device", "cpu"])


def test_probe_runner_deduplicates_raw_histories_preserves_slots_and_never_promotes_cpu(tmp_path, monkeypatch):
    module, args, manifest, physical = synthetic_runner_setup(tmp_path, monkeypatch)
    output = tmp_path / "probe"
    output.mkdir()
    result = module.execute_probe(args, output, module.time.monotonic())
    assert result["status"] == "passed" and result["normal_available_by_fold"] == [12, 12, 12]
    assert result["request_slots"] == 48 and not result["full_run_eligible"] and result["fits"] == 0
    assert set(result["stage_seconds"]) == {"read_seconds", "resample_seconds", "filter_seconds",
                                           "normalize_seconds", "preprocess_seconds", "infer_seconds",
                                           "pool_seconds", "write_seconds"}
    assert all(value >= 0 for value in result["stage_seconds"].values())
    assert len(physical) == len(set(physical)) == result["raw_history_requests"]
    measurements = [json.loads(line) for line in (output / "measurements.jsonl").read_text().splitlines()]
    assert sum(len(item["request_slots"]) for item in measurements) == 48
    assert any(item["numerical_checks"].get("batch1_batch8") for item in measurements)


@pytest.mark.parametrize("slot", ["initial", "daily"])
@pytest.mark.parametrize("fault", ["one_nan", "one_finite", "wrong_nan"])
def test_boundary_gate_rejects_incomplete_or_misplaced_calibration_masks(tmp_path, monkeypatch, slot, fault):
    module, args, _, _ = synthetic_runner_setup(tmp_path, monkeypatch, mask_fault=fault, mask_slot=slot)
    output = tmp_path / "failed-mask"
    with pytest.raises(ValueError, match="calibration mask"):
        fixture_main(module, args, output)
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "failed" and not receipt["full_run_eligible"]
    assert "calibration mask" in (output / "original-error.txt").read_text()
    assert (output / "failed-selected-native.npy").is_file()


def test_defective_calibration_mask_is_rejected_before_encoder_forward(tmp_path, monkeypatch):
    module, args, _, _ = synthetic_runner_setup(tmp_path, monkeypatch, mask_fault="one_finite")
    model = small_model()
    forwards = []
    model.register_forward_hook(lambda *args: forwards.append(True))
    monkeypatch.setattr(module, "load_frozen_encoder", lambda *args, **kwargs: (model, {"weights_only": True}))
    with pytest.raises(ValueError, match="calibration mask"):
        fixture_main(module, args, tmp_path / "before-forward")
    assert forwards == []


def test_exact_acquisition_mask_uses_all_cumulative_file_intersections():
    parts = [dict(samples=240001), dict(samples=120000), dict(samples=180000)]
    initial = probe.acquisition_calibration_mask(parts, first_sample=0, length=120000)
    expected = np.zeros(120000, dtype=bool)
    expected[:6000] = True
    np.testing.assert_array_equal(initial, expected)
    assert not probe.acquisition_calibration_mask(parts, first_sample=6000, length=120000).any()
    daily = probe.acquisition_calibration_mask(parts, first_sample=126001, length=120000)
    expected[:] = False
    expected[-6000:] = True
    np.testing.assert_array_equal(daily, expected)
    partial = probe.acquisition_calibration_mask(parts, first_sample=243001, length=120000)
    expected[:] = False
    expected[:3000] = True
    expected[117000:] = True
    np.testing.assert_array_equal(partial, expected)


def test_t630_clearance_allows_separately_invalid_physiology_or_missingness(tmp_path, monkeypatch):
    module, args, _, _ = synthetic_runner_setup(tmp_path, monkeypatch, clear_invalid=True)
    output = tmp_path / "clear-other-rejection"
    assert fixture_main(module, args, output) == 0
    items = [json.loads(line) for line in (output / "measurements.jsonl").read_text().splitlines()]
    clear = next(item for item in items if any(r["slot"] == "clear" for r in item["request_slots"]))
    assert not clear["available"] and clear["rejections"]["nonfinite_native"] >= 1
    assert clear["numerical_checks"]["acquisition_mask_samples"] == 0
    assert clear["numerical_checks"]["t630_acquisition_clear"]


def test_required_synthetic_fallbacks_are_executed_receipted_without_raw_replacements(tmp_path, monkeypatch):
    module, args, manifest, physical = synthetic_runner_setup(tmp_path, monkeypatch, fallback=True)
    output = tmp_path / "fallbacks"
    assert fixture_main(module, args, output) == 0
    receipt = json.loads((output / "receipt.json").read_text())
    checks = receipt["synthetic_checks"]
    assert checks["required"] == ["daily", "terminal"]
    assert all(checks["checks"][name]["status"] == "passed" for name in checks["required"])
    assert checks["checks"]["terminal"]["decision_sample"] == 126001
    assert checks["checks"]["daily"]["calibration_samples"] == 6000
    assert receipt["synthetic_check_seconds"] >= 0
    assert receipt["synthetic_checks_sha256"] == hashlib.sha256((output / "synthetic-checks.json").read_bytes()).hexdigest()
    selected = [r for r in manifest["requests"] if r["status"] == "selected"]
    assert len(physical) == len({(r["record_id"], r["decision_sample"]) for r in selected})
    assert receipt["request_slots"] == 48 and receipt["structural_slots"] == 3
    assert len(checks["required_slots"]) == 6


@pytest.mark.parametrize("failure", ["daily", "terminal", "missing_terminal"])
def test_required_synthetic_failure_prevents_pass_before_any_raw_request(tmp_path, monkeypatch, failure):
    module, args, _, physical = synthetic_runner_setup(tmp_path, monkeypatch, fallback=True)
    kind = "terminal" if failure == "missing_terminal" else failure

    def fail(model):
        if failure == "missing_terminal":
            return None
        raise ValueError("synthetic fallback deliberately failed")

    monkeypatch.setattr(module, f"synthetic_{kind}_check", fail, raising=False)
    output = tmp_path / "failed-fallback"
    with pytest.raises(ValueError, match="synthetic fallback|missing synthetic"):
        fixture_main(module, args, output)
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "failed" and not receipt["full_run_eligible"]
    assert receipt["raw_history_requests"] == 0 and receipt["successful_raw_history_reads"] == 0
    assert receipt["synthetic_checks"]["checks"][kind]["status"] == "failed"
    assert physical == []


@pytest.mark.parametrize("raise_on", [1, 3])
def test_reader_exception_preserves_attempt_success_completed_and_failing_request(tmp_path, monkeypatch, raise_on):
    module, args, _, physical = synthetic_runner_setup(tmp_path, monkeypatch, raise_on=raise_on)
    output = tmp_path / "reader-error"
    with pytest.raises(OSError, match="synthetic source read failed"):
        fixture_main(module, args, output)
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["raw_history_requests"] == len(physical) == raise_on
    assert receipt["successful_raw_history_reads"] == raise_on - 1
    assert receipt["completed_unique_histories"] == raise_on - 1
    assert receipt["current_request"]["record_id"] == physical[-1][0]
    assert receipt["current_request"]["decision_sample"] == physical[-1][2]
    assert "synthetic source read failed" in (output / "original-error.txt").read_text()


def test_selection_inputs_reject_alternate_unbound_audit_roles(tmp_path, monkeypatch):
    module = runner_module()
    monkeypatch.chdir(tmp_path)
    cycle, cache, original, alternate = [tmp_path / name for name in ("cycle", "cache", "original", "alternate")]
    for path in (cycle, cache, original, alternate):
        path.mkdir()
    (original / "roles.json").write_text(json.dumps({"roles": "original"}))
    (alternate / "roles.json").write_text(json.dumps({"cache_index_sha256": "index", "roles": []}))
    declaration = dict(files_sha256={"original/roles.json": hashlib.sha256((original / "roles.json").read_bytes()).hexdigest()},
                       compact_binding={"index_sha256": "index"})
    (cycle / "declaration-binding.json").write_text(json.dumps(declaration))
    (cache / "index.json").write_text(json.dumps({"records": []}))
    (cache / "compact.json").write_text(json.dumps({"records": []}))
    monkeypatch.setattr(module, "compact_cache_binding", lambda *a, **k: declaration["compact_binding"])
    with pytest.raises(ValueError, match="role"):
        module.selection_inputs(SimpleNamespace(cycle=cycle, cache=cache, audit=alternate))


def test_runner_preserves_original_error_and_partial_raw_count(tmp_path, monkeypatch):
    module = runner_module()

    def fail(args, output, started):
        (output / "execution-state.json").write_text(json.dumps({"raw_history_requests": 3,
                                                                "completed_unique_histories": 2}))
        raise ValueError("original fixed failure")

    monkeypatch.setattr(module, "execute_probe", fail)
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="original fixed failure"):
        module.main(["--cycle", str(tmp_path), "--cache", str(tmp_path), "--audit", str(tmp_path),
                     "--output", str(output), "--mode", "probe", "--selection", "fixed.json",
                     "--selection-sha256", "fixedhash"])
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["raw_history_requests"] == 3 and receipt["completed_unique_histories"] == 2
    assert receipt["status"] == "failed" and not receipt["retry"] and not receipt["full_run_eligible"]
    assert "original fixed failure" in (output / "original-error.txt").read_text()


@pytest.mark.parametrize("nested", [False, True])
def test_runtime_relocation_keeps_portable_raw_contract_without_rewriting_index(tmp_path, nested):
    from cardiotwin.data import RecordInfo
    from cardiotwin.investigation import _record_contract

    original = tmp_path / "original" / "record_001"
    original.mkdir(parents=True)
    for name in ("record_001_ecg_00.h5", "record_001_ecg_labels.csv"):
        (original / name).write_bytes(b"fixed source bytes")
    record = dict(record_id="record_001", patient_id="p", path=str(original), fs=200.,
                  duration_seconds=1200., annotated=True, clinical={}, metadata_audit=[], source="iridia-af",
                  signal_parts=[dict(path=str(original / "record_001_ecg_00.h5"), samples=240000, key="ecg")])
    before = json.dumps(record, sort_keys=True)
    destination = tmp_path / "staged"
    destination.mkdir()
    actual = destination / "record_001" if nested else destination
    actual.mkdir(exist_ok=True)
    for file in original.iterdir():
        shutil.copyfile(file, actual / file.name)
    relocated = probe.relocate_record(record, destination)
    assert json.dumps(record, sort_keys=True) == before
    assert Path(relocated["path"]) == actual.resolve()
    assert _record_contract(RecordInfo(**record)) == _record_contract(RecordInfo(**relocated))


@pytest.mark.parametrize("kind", ["record_id", "filename", "traversal", "symlink", "missing"])
def test_runtime_relocation_rejects_unsafe_or_missing_source_paths_before_read(tmp_path, kind):
    root = tmp_path / "staged"
    root.mkdir()
    target = root / "record_001_ecg_00.h5"
    target.write_bytes(b"bound")
    record = dict(record_id="record_001", path="/old/record_001", source="iridia-af",
                  signal_parts=[dict(path="/old/record_001/record_001_ecg_00.h5", samples=240000, key="ecg")])
    if kind == "record_id":
        record["record_id"] = "../record_001"
    elif kind == "filename":
        record["signal_parts"][0]["path"] = "/old/record_001/record_002_ecg_00.h5"
    elif kind == "traversal":
        record["signal_parts"][0]["path"] = "/old/record_001/../record_001/record_001_ecg_00.h5"
    elif kind == "symlink":
        target.unlink()
        outside = tmp_path / "outside"
        outside.write_bytes(b"unbound")
        target.symlink_to(outside)
    elif kind == "missing":
        target.unlink()
    with pytest.raises(ValueError, match="relocation|source"):
        probe.relocate_record(record, root)


def supported_fixture():
    records, rows = {}, []
    for i in range(16):
        group = f'g{i:02}'
        for suffix in ('a', 'b'):
            rid = group + suffix
            records[rid] = dict(record_id=rid, patient_id=group, group_id=group, fs=200,
                                duration_seconds=126001 / 200, signal_parts=[dict(samples=126001)])
            for sample in [*range(0, 126001, 6000), 126001]:
                rows.append(BlindedRow(group_id=group, patient_id=group, record_id=rid,
                                       timestamp=sample / 200, accepted=sample >= 120000))
    roles = [dict(fold=f, fit=[f'g{i:02}' for i in range(12)], stop=['g12'],
                  calibration=['g13'], policy=['g14'], outer=['g15']) for f in range(3)]
    old = probe.select_probe_requests(rows, records, roles)
    return rows, records, roles, old


def test_repaired_canonical_hash_has_exact_utf8_array_serialization():
    assert probe.panel_rank('group', 0, 'é') == '5def3ff1a7eedb84a8c891c9b4b1d1d3dc864f6bae02cdf257188f1bafcffd58'
    assert probe.panel_rank('row', 2, 'g00', 'rβ', 126001) == '30de6bef0666e19228f706441bc7ec1bafa19e502de2aa2611e4780efa932208'


def test_repaired_selection_all_records_terminal_and_original_boundaries():
    rows, records, roles, old = supported_fixture()
    selected, support = probe.select_metadata_supported_requests(rows, records, roles, old)
    normal = [r for r in selected if r['slot'] == 'normal']
    assert len(selected) == 48 and len(normal) == 36
    assert any(r['record_id'].endswith('b') for r in normal)
    assert any(r['decision_sample'] == 126001 for r in normal)
    assert [r for r in selected if r['slot'] != 'normal'] == [r for r in old if r['slot'] != 'normal']
    assert support['unique_histories'] <= 48
    for fold in range(3):
        chosen = [r for r in normal if r['fold'] == fold]
        assert len({r['group_id'] for r in chosen}) == 12
        assert [r['group_id'] for r in chosen] == sorted(roles[fold]['fit'],
            key=lambda g: (probe.panel_rank('group', fold, g), g.encode()))
        assert support['folds'][fold]['qualifying_groups'] == 12
        for r in chosen:
            pool = [(probe.panel_rank('row', fold, r['group_id'], rid, end), rid.encode(), end, index)
                    for index, row in enumerate(rows) if row['group_id'] == r['group_id']
                    for rid, end in [(row['record_id'], round(row['timestamp'] * 200))]
                    if end in (126000, 126001)]
            best = min(pool)
            assert (r['record_id'].encode(), r['decision_sample'], r['row_index']) == best[1:]
            assert support['folds'][fold]['eligible_rows_by_group'][r['group_id']] == 4


def test_repaired_known_mask_half_open_endpoints_actual_offsets_no_dense_masks(monkeypatch):
    rows, records, roles, old = supported_fixture()
    # Second short acquisition starts at sample 126000: terminal intersects its mask,
    # while history ending exactly at its start is clear.
    for record in records.values():
        record['signal_parts'] = [dict(samples=126000), dict(samples=1)]
    monkeypatch.setattr(probe, 'acquisition_calibration_mask', lambda *a, **k: pytest.fail('dense mask scan'))
    selected, support = probe.select_metadata_supported_requests(rows, records, roles, old)
    assert all(r['decision_sample'] == 126000 for r in selected if r['slot'] == 'normal')
    assert all(v == 2 for f in support['folds'] for v in f['eligible_rows_by_group'].values())


@pytest.mark.parametrize('fault', ['identity', 'accepted_warmup', 'acceptance', 'coordinate', 'cadence',
                                    'parts', 'duration', 'fold', 'overlap', 'few', 'boundary'])
def test_repaired_invalid_metadata_fails_without_replacement(fault):
    rows, records, roles, old = supported_fixture()
    if fault == 'identity':
        rows[0].values['group_id'] = 'unknown'
    elif fault == 'accepted_warmup':
        rows[0].values['accepted'] = True
    elif fault == 'acceptance':
        rows[0].values['accepted'] = 2
    elif fault == 'coordinate':
        rows[1].values['timestamp'] += .001
    elif fault == 'cadence':
        del rows[1]
    elif fault == 'parts':
        records['g00a']['signal_parts'][0]['samples'] = 0
    elif fault == 'duration':
        records['g00a']['duration_seconds'] += .005
    elif fault == 'fold':
        roles[1]['fold'] = 0
    elif fault == 'overlap':
        roles[0]['stop'].append('g00')
    elif fault == 'few':
        roles[0]['fit'].remove('g00')
        roles[0]['stop'].append('g00')
    else:
        old[-2]['decision_sample'] -= 1
    with pytest.raises(ValueError):
        probe.select_metadata_supported_requests(rows, records, roles, old)


def panel_runner_fixture(tmp_path, monkeypatch):
    module = runner_module()
    monkeypatch.chdir(tmp_path)
    cycle = Path('reports/ml-investigation/next-cycle/ecgfounder-cycle-20261013')
    cache, audit = Path('data/cache'), Path('reports/ml-investigation/next-cycle/audit')
    for directory in (cycle / 'repair-panel-20261014-b', cycle / 'selection-20261014', cache, audit):
        directory.mkdir(parents=True)
    rows, records, roles, old_requests = supported_fixture()
    index = dict(records=[dict(record=r) for r in records.values()])
    compact = dict(records=[{k: r[k] for k in ('record_id', 'patient_id', 'group_id')} for r in records.values()])
    (cache / 'index.json').write_text(json.dumps(index))
    (cache / 'compact.json').write_text(json.dumps(compact))
    (audit / 'roles.json').write_text(json.dumps(dict(cache_index_sha256='index-hash', roles=roles)))
    lookup = {rid: i for i, rid in enumerate(records)}
    values = [(lookup[r['record_id']], r['timestamp'], r['accepted']) for r in rows]
    np.save(cache / 'metadata.npy', np.array(values, dtype=[('record', 'i4'), ('timestamp', 'f8'), ('accepted', 'i1')]))
    original_files = {str(p): module.file_hash(p) for p in (cache / 'index.json', audit / 'roles.json')}
    binding = dict(index_sha256='index-hash', compact_sha256=module.file_hash(cache / 'compact.json'))
    declaration = cycle / 'declaration-binding.json'
    declaration.write_text(json.dumps(dict(files_sha256=original_files, compact_binding=binding)))
    monkeypatch.setattr(module, 'compact_cache_binding', lambda *a, **k: binding)
    monkeypatch.setattr(module, 'source_manifest', lambda: dict(files={}))
    sources = ['src/cardiotwin/' + name + '.py' for name in
               ('ecgfounder', 'ecgfounder_probe', 'data', 'precursor', 'investigation', 'progress', 'gpu_telemetry')]
    sources += ['scripts/run_ecgfounder_probe.py', 'scripts/kaggle_ecgfounder_probe.py',
                'reports/ml-investigation/monitor-storage.py']
    for name in sources:
        path = Path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('synthetic source\n')
    old_selection = cycle / 'selection-20261014/selected.json'
    old_selection.write_text(json.dumps(dict(requests=old_requests, inputs=dict(files_sha256=original_files,
        declaration_sha256=module.file_hash(declaration), compact_binding=binding,
        source_sha256={'historical-source-must-not-be-compared.py': 'old-source-hash'}))))
    attempt = cycle / 'repair-panel-20261014-b/spec.md'
    design = cycle / 'repair-design-20261014.md'
    addendum = attempt.parent / 'dense-dispatch-timing-addendum.md'
    for path in (attempt, design, addendum):
        path.write_text('fixed synthetic ' + path.name)
    for constant, path in [('PANEL_SPEC_SHA', attempt), ('PANEL_DESIGN_SHA', design),
                           ('PANEL_TIMING_SHA', addendum), ('OLD_SELECTION_SHA', old_selection)]:
        monkeypatch.setattr(module, constant, module.file_hash(path))
    names = [attempt, design, addendum, old_selection, declaration, *(Path(p) for p in sources),
             *(Path(p) for p in original_files)]
    attempt_declaration = attempt.parent / 'declaration-binding.json'
    attempt_declaration.write_text(json.dumps(dict(files_sha256={str(p): module.file_hash(p) for p in names},
        protected_files_sha256={'README.md': 'opaque-parent-attestation'})))
    args = SimpleNamespace(cycle=cycle, cache=cache, audit=audit, attempt=attempt)
    return module, args, attempt_declaration, old_selection, design


def test_panel_runner_binds_attempt_before_after_and_old_sources_are_historical(tmp_path, monkeypatch):
    module, args, declaration, old, _ = panel_runner_fixture(tmp_path, monkeypatch)
    manifest, _ = module.selection_inputs(args)
    assert manifest['attempt']['identity'] == probe.PANEL_IDENTITY
    assert manifest['attempt']['historical_source_sha256'] == {'historical-source-must-not-be-compared.py': 'old-source-hash'}
    assert manifest['inputs']['files_sha256'][str(declaration)] == module.file_hash(declaration)
    assert manifest['attempt']['old_selection_sha256'] == module.file_hash(old)
    assert manifest['support']['unique_histories'] <= 48
    assert 'README.md' not in manifest['inputs']['files_sha256']
    # Omitting --attempt must still produce the original midpoint manifest/requests.
    args.attempt = None
    default, _ = module.selection_inputs(args)
    assert 'attempt' not in default and 'support' not in default
    assert default['requests'] == json.loads(old.read_text())['requests']


@pytest.mark.parametrize('change', ['during', 'old_inputs', 'missing_timing', 'wrong_path', 'acceptance'])
def test_panel_runner_refuses_bound_input_changes_and_invalid_flags(tmp_path, monkeypatch, change):
    module, args, declaration, old, design = panel_runner_fixture(tmp_path, monkeypatch)
    if change == 'during':
        selector = module.select_metadata_supported_requests

        def mutate(*values):
            result = selector(*values)
            design.write_text('changed while selecting')
            return result

        monkeypatch.setattr(module, 'select_metadata_supported_requests', mutate)
    elif change == 'old_inputs':
        payload = json.loads(old.read_text())
        payload['inputs']['compact_binding'] = {}
        old.write_text(json.dumps(payload))
        monkeypatch.setattr(module, 'OLD_SELECTION_SHA', module.file_hash(old))
        payload = json.loads(declaration.read_text())
        payload['files_sha256'][str(old)] = module.file_hash(old)
        declaration.write_text(json.dumps(payload))
    elif change == 'missing_timing':
        payload = json.loads(declaration.read_text())
        del payload['files_sha256'][str(args.attempt.parent / 'dense-dispatch-timing-addendum.md')]
        declaration.write_text(json.dumps(payload))
    elif change == 'wrong_path':
        args.attempt = design
    else:
        data = np.load(args.cache / 'metadata.npy')
        data['accepted'][0] = 2
        np.save(args.cache / 'metadata.npy', data)
    with pytest.raises(ValueError):
        module.selection_inputs(args)


def test_fixed_dense_dispatch_replays_twenty_threes_and_preserves_all_vectors(monkeypatch):
    module = runner_module()
    calls = []
    prepared = np.arange(60, dtype=np.float32)[:, None] * np.ones((1, 5000), np.float32)
    reference = np.broadcast_to(np.arange(60, dtype=np.float32)[:, None], (60, 1024)).copy()
    monkeypatch.setattr(module, 'preprocess_history', lambda *a, **k: prepared)

    def encode(model, chunks):
        calls.append(len(chunks))
        return np.broadcast_to(chunks[:, :1], (len(chunks), 1024)).copy()

    monkeypatch.setattr(module, 'encode_chunks', encode)
    result = module.dense_dispatch_check(None, np.zeros(120000), dict(timestamp=630., decision_sample=126000),
                                         reference, module.pool_delta(reference), device='cpu')
    assert calls == [3] * 20
    assert result['chunks'] == 60 and result['forward_batches'] == 20 and result['status'] == 'passed'
    assert result['timing_kind'] == 'in_memory_reprocessing'
    assert all(result[name] >= 0 for name in ('preprocess_seconds', 'infer_seconds', 'pool_seconds',
                                             'total_seconds', 'comparison_seconds'))
    reference[0, 0] += 1
    with pytest.raises(AssertionError):
        module.dense_dispatch_check(None, np.zeros(120000), dict(timestamp=630., decision_sample=126000),
                                    reference, module.pool_delta(reference), device='cpu')


def test_dense_dispatch_summary_requires_all_available_unique_normals_and_slower_rate():
    module = runner_module()
    selected = [dict(record_id='a', decision_sample=126000, slot='normal', status='selected') for _ in range(3)]
    selected += [dict(record_id='b', decision_sample=126000, slot='normal', status='selected'),
                 dict(record_id='c', decision_sample=126000, slot='normal', status='selected'),
                 dict(record_id='a', decision_sample=130000, slot='terminal', status='selected')]
    results = {('a', 126000): dict(available=True, numerical_checks=dict(dense_dispatch=dict(
        status='passed', chunks=60, total_seconds=2., preprocess_seconds=.5, infer_seconds=1., pool_seconds=.5))),
        ('b', 126000): dict(available=True, numerical_checks=dict(dense_dispatch=dict(
        status='passed', chunks=60, total_seconds=6., preprocess_seconds=1., infer_seconds=4., pool_seconds=1.))),
        ('c', 126000): dict(available=False)}
    summary = module.dispatch_summary(results, selected)
    assert summary['unique_normal_histories'] == 2 and summary['chunks'] == 120
    assert summary['aggregate_chunks_per_second'] == 15
    assert summary['planning_chunks_per_second'] == pytest.approx(12)
    assert len(summary['record_timings']) == 2
    del results['a', 126000]['numerical_checks']['dense_dispatch']
    with pytest.raises(ValueError, match='dispatch'):
        module.dispatch_summary(results, selected)


def test_panel_failure_receipts_best_effort_hashes_do_not_mask_original(tmp_path, monkeypatch):
    module, args, declaration, _, design = panel_runner_fixture(tmp_path, monkeypatch)
    output = tmp_path / 'failed'

    def fail_select(values):
        design.unlink()
        raise OSError('original synthetic selection failure')

    monkeypatch.setattr(module, 'selection_inputs', fail_select)
    with pytest.raises(OSError, match='original synthetic'):
        module.main(['--cycle', str(args.cycle), '--cache', str(args.cache), '--audit', str(args.audit),
                     '--attempt', str(args.attempt), '--output', str(output), '--mode', 'select'])
    receipt = json.loads((output / 'receipt.json').read_text())
    after = receipt['binding_hashes_after']
    assert after['files_sha256'][str(declaration)] == module.file_hash(declaration)
    assert str(design) in after['hash_errors']
    assert receipt['error'] == 'original synthetic selection failure'


@pytest.mark.parametrize('fault', ['shared_patient', 'multiple_patients'])
def test_repaired_group_patient_identity_must_be_bijective(fault):
    rows, records, roles, old = supported_fixture()
    for record in records.values():
        if fault == 'shared_patient':
            record['patient_id'] = 'same-person'
        elif record['record_id'] == 'g00b':
            record['patient_id'] = 'second-person'
    for row in rows:
        row.values['patient_id'] = records[row['record_id']]['patient_id']
    for request in old:
        if request['status'] == 'selected':
            request['patient_id'] = records[request['record_id']]['patient_id']
    with pytest.raises(ValueError, match='patient|group'):
        probe.select_metadata_supported_requests(rows, records, roles, old)


def test_panel_absolute_remote_paths_produce_identical_portable_manifest(tmp_path, monkeypatch):
    module, args, _, _, _ = panel_runner_fixture(tmp_path, monkeypatch)
    relative, _ = module.selection_inputs(args)
    for name in ('cycle', 'cache', 'audit', 'attempt'):
        setattr(args, name, getattr(args, name).resolve())
    absolute, _ = module.selection_inputs(args)
    assert relative == absolute
    assert all(not Path(name).is_absolute() for name in absolute['inputs']['files_sha256'])


def test_panel_execution_dispatch_is_unique_available_normals_only(tmp_path, monkeypatch):
    module, args, manifest, physical = synthetic_runner_setup(tmp_path, monkeypatch)
    manifest['attempt'] = dict(identity=probe.PANEL_IDENTITY)
    args.attempt = tmp_path / 'spec.md'
    args.selection.write_text(json.dumps(manifest))
    args.selection_sha256 = module.file_hash(args.selection)
    seen = []

    def checks(model, native, request, measured, *, dense_dispatch=False, device='cpu'):
        result = dict(direct_ring=True, batch1_batch8=True, future_suffix=True)
        if dense_dispatch:
            seen.append((request['record_id'], request['decision_sample']))
            result['dense_dispatch'] = dict(status='passed', chunks=60, total_seconds=1.,
                preprocess_seconds=.2, infer_seconds=.7, pool_seconds=.1)
        return result

    monkeypatch.setattr(module, 'numerical_checks', checks)
    output = tmp_path / 'probe'
    output.mkdir()
    result = module.execute_probe(args, output, module.time.monotonic())
    expected = {(r['record_id'], r['decision_sample']) for r in manifest['requests'] if r['slot'] == 'normal'}
    assert set(seen) == expected and len(seen) == len(expected)
    assert result['dense_dispatch']['unique_normal_histories'] == len(expected)
    assert len(physical) == len(set(physical)) == result['raw_history_requests']
    assert result['cost']['planning_chunks_per_second'] <= result['dense_dispatch']['planning_chunks_per_second']
    assert not result['full_run_eligible']
    assert json.loads((output / 'dense-dispatch.json').read_text())['status'] == 'passed'


def test_dense_dispatch_failure_preserves_partial_status_without_extra_reads(tmp_path, monkeypatch):
    module, args, manifest, physical = synthetic_runner_setup(tmp_path, monkeypatch)
    manifest['attempt'] = dict(identity=probe.PANEL_IDENTITY)
    args.selection.write_text(json.dumps(manifest))
    args.selection_sha256 = module.file_hash(args.selection)

    def failed(*a, **k):
        if k.get('dense_dispatch'):
            raise ValueError('synthetic dense replay discrepancy')
        return dict(direct_ring=True)

    monkeypatch.setattr(module, 'numerical_checks', failed)
    output = tmp_path / 'probe'
    with pytest.raises(ValueError, match='synthetic dense'):
        module.main(['--cycle', str(args.cycle), '--cache', str(args.cache), '--audit', str(args.audit),
            '--output', str(output), '--mode', 'probe', '--selection', str(args.selection),
            '--selection-sha256', args.selection_sha256, '--attempt', str(tmp_path / 'spec.md')])
    receipt = json.loads((output / 'receipt.json').read_text())
    assert receipt['dense_dispatch']['status'] == 'failed'
    assert receipt['dense_dispatch']['current_request']
    assert receipt['error'] == 'synthetic dense replay discrepancy'
    assert len(physical) == len(set(physical)) == receipt['raw_history_requests']


def test_panel_runner_rejects_any_additional_budget_before_selection(tmp_path, monkeypatch):
    module, args, _, _, _ = panel_runner_fixture(tmp_path, monkeypatch)
    args.additional_budget = tmp_path / 'budget.json'
    args.additional_budget.write_text(json.dumps({name: 1 for name in probe.OVERHEAD_COMPONENTS}))
    monkeypatch.setattr(module, 'select_metadata_supported_requests', lambda *a: pytest.fail('selector called'))
    with pytest.raises(ValueError, match='no additional budget'):
        module.selection_inputs(args)


def test_dense_dispatch_synchronizes_declared_cuda_around_forward_timing(monkeypatch):
    module = runner_module()
    torch = pytest.importorskip('torch')
    events = []
    reference = np.zeros((60, 1024), dtype=np.float32)
    monkeypatch.setattr(module, 'preprocess_history', lambda *a, **k: np.zeros((60, 5000), np.float32))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device: events.append(('sync', device)))

    def encode(model, chunks):
        events.append(('forward', len(chunks)))
        return np.zeros((len(chunks), 1024), np.float32)

    monkeypatch.setattr(module, 'encode_chunks', encode)
    result = module.dense_dispatch_check(None, np.zeros(120000), dict(timestamp=630., decision_sample=126000),
                                         reference, module.pool_delta(reference), device='cuda:0')
    assert events == [('sync', 'cuda:0'), *[('forward', 3)] * 20, ('sync', 'cuda:0')]
    assert result['device'] == 'cuda:0'


def test_repaired_qualifying_groups_are_ranked_before_twelve_group_cutoff():
    rows, records, roles, old = supported_fixture()
    for role in roles:
        role.update(fit=[f'g{i:02}' for i in range(14)], stop=[], calibration=[])
    selected, support = probe.select_metadata_supported_requests(rows, records, roles, old)
    for fold in range(3):
        expected = sorted(roles[fold]['fit'], key=lambda g: (probe.panel_rank('group', fold, g), g.encode()))[:12]
        assert [r['group_id'] for r in selected if r['fold'] == fold and r['slot'] == 'normal'] == expected
        assert support['folds'][fold]['qualifying_groups'] == 14
        assert len(support['folds'][fold]['eligible_rows_by_group']) == 14
