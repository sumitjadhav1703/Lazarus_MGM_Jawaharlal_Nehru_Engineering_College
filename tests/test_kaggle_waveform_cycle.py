import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

needs_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="requires torch (install with: uv sync --extra neural)",
)
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("waveform_cycle", SCRIPTS / "kaggle_waveform_cycle.py")
cycle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cycle)


def test_runtime_gate_rejects_expensive_work_and_accounts_for_scoring():
    cheap = cycle.cost_gate(0.001, 0.002, 0.001, 0.001, 50000, 1000000, 41400)
    assert cheap["cost_gate_passed"]
    assert cheap["maximum_batched_histories"] == 1320000
    expensive = cycle.cost_gate(0.1, 0.1, 0.1, 0.1, 50000, 1000000, 41400)
    assert not expensive["cost_gate_passed"]
    assert expensive["estimated_seconds"] > cheap["estimated_seconds"]


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_runtime_gate_rejects_invalid_timings(value):
    with pytest.raises(ValueError):
        cycle.cost_gate(value, 0.1, 0.1, 0.1, 1, 1, 100)


def test_failed_hardware_gate_saves_receipt_and_never_starts_fit(tmp_path, monkeypatch):
    import contextlib
    import json

    output = tmp_path / "fresh-output"
    argv = [
        "cycle",
        "--data",
        str(tmp_path),
        "--cache",
        str(tmp_path),
        "--split",
        str(tmp_path / "split.json"),
        "--output",
        str(output),
        "--session-seconds-remaining",
        "1000",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(cycle, "dataset_bytes", lambda _: 100)
    monkeypatch.setattr(cycle, "GpuTelemetry", contextlib.nullcontext)
    monkeypatch.setattr(cycle, "probe", lambda *args: {"cost_gate_passed": False})
    monkeypatch.setattr(
        cycle, "run_child", lambda *args, **kw: pytest.fail("Fit launched after rejected gate")
    )
    with pytest.raises(RuntimeError, match="cost/memory"):
        cycle.main()
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "blocked_by_runtime_gate"
    assert "traceback" in receipt and not (output / "model").exists()
    with pytest.raises(FileExistsError):
        cycle.main()


def test_probe_exception_preserves_original_error(tmp_path, monkeypatch):
    import contextlib
    import json

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cycle",
            "--data",
            str(tmp_path),
            "--cache",
            str(tmp_path),
            "--split",
            str(tmp_path / "split.json"),
            "--output",
            str(tmp_path / "out"),
            "--session-seconds-remaining",
            "1000",
        ],
    )
    monkeypatch.setattr(cycle, "dataset_bytes", lambda _: 100)
    monkeypatch.setattr(cycle, "GpuTelemetry", contextlib.nullcontext)

    def failed_probe(*args):
        raise OSError("original HDF5 error")

    monkeypatch.setattr(cycle, "probe", failed_probe)
    with pytest.raises(OSError, match="original HDF5 error"):
        cycle.main()
    receipt = json.loads((tmp_path / "out/receipt.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["original_error"] == "original HDF5 error"


@pytest.mark.parametrize(
    "folds,seed,expected", [(3, 42, "complete"), (2, 42, "incomplete"), (3, 1, "incomplete")]
)
@needs_torch
def test_completion_requires_three_folds_and_declared_seed(tmp_path, monkeypatch, folds, seed, expected):
    import contextlib
    import json

    output = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cycle",
            "--data",
            str(tmp_path),
            "--cache",
            str(tmp_path),
            "--split",
            str(tmp_path / "split.json"),
            "--output",
            str(output),
            "--session-seconds-remaining",
            "1000",
        ],
    )
    monkeypatch.setattr(cycle, "dataset_bytes", lambda _: 100)
    monkeypatch.setattr(cycle, "GpuTelemetry", contextlib.nullcontext)
    monkeypatch.setattr(cycle, "probe", lambda *args: {"cost_gate_passed": True})

    def fake_child(command, log, cap, storage_roots):
        assert "waveform16" in command and "--sequence-cache" not in command
        assert 0 < cap < 700
        model = output / "model"
        model.mkdir()
        summary = dict(
            status="complete",
            failures=[],
            specs=[["waveform", 16]],
            seed=seed,
            screening_epochs=4,
            screening_max_training_windows=10000,
            early_stop_weighting="patient_only",
            combined={"waveform_cnn_16": {"complete_outer_folds": folds}},
        )
        (model / "summary.json").write_text(json.dumps(summary))
        return {"status": "finished"}

    monkeypatch.setattr(cycle, "run_child", fake_child)
    if expected == "complete":
        cycle.main()
    else:
        with pytest.raises(RuntimeError, match="Incomplete"):
            cycle.main()
    assert json.loads((output / "receipt.json").read_text())["status"] == expected


def test_measured_probe_uses_separate_access_patterns_and_bounded_screen(monkeypatch):
    timings = (0.0619445341, 0.0575658146, 0.0117829149, 0.0005457239)
    four = cycle.cost_gate(*timings, 66981, 895889, 40926)
    assert four["cost_gate_passed"]
    assert 34000 < four["estimated_seconds"] < 35000
    assert four["random_training_histories"] == 120000
    assert four["chronological_evaluation_histories"] == 1163813
    monkeypatch.setattr(cycle, "EPOCHS", 8)
    eight = cycle.cost_gate(*timings, 66981, 895889, 40926)
    assert not eight["cost_gate_passed"]
