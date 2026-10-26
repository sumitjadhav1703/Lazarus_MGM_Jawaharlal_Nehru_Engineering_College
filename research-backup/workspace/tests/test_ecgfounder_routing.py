"""Composed fixed-phase routing; generated receipts only, no workload or jobs."""
import importlib.util
import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest


def module(monkeypatch):
    path = Path(__file__).parents[1] / "scripts/kaggle_ecgfounder_engineering.py"
    spec = importlib.util.spec_from_file_location("routing_engineering", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(m.time, "monotonic", lambda: 50.0)
    monkeypatch.setattr(m, "datetime", SimpleNamespace(now=lambda tz: m.DEADLINE - timedelta(days=1)))
    return m


def timeout(m, tmp_path, phase, *, valid=True):
    receipt = dict(status="failed", phase=phase, session_started=10.0,
                   engineering_started=20.0, phase_started=25.0,
                   timeout_scope="phase_only", cleanup_complete=True, drain_complete=True,
                   termination_requested_by_supervisor=True, exit_code=-15,
                   post_failure_safety_validated=True, original_error="Local phase ceiling exhausted")
    path = tmp_path / (phase + ".json")
    if valid:
        path.write_text(json.dumps(receipt))
    return m.PhaseTimeout(receipt, path)


def run(m, tmp_path, *, fail=None, kind="timeout", bad_validation=False):
    receipt = dict(session_started=10.0, engineering_started=20.0,
                   source_identity="generated-source", phases={}, full_run_eligible=False)
    calls, saves, validations = [], [], []

    def phase(name):
        calls.append(name)
        if name == fail:
            if kind == "timeout":
                raise timeout(m, tmp_path, name)
            if kind == "missing_receipt":
                raise timeout(m, tmp_path, name, valid=False)
            raise RuntimeError(kind)
        path = tmp_path / (name + ".json")
        terminal = dict(status="complete", phase=name, session_started=10.0,
                        engineering_started=20.0, elapsed_seconds=1.0, receipt_path=str(path),
                        cleanup_complete=True, drain_complete=True)
        path.write_text(json.dumps(terminal))
        return terminal

    def validate():
        validations.append(True)
        if bad_validation and fail in calls:
            raise RuntimeError("Fresh source/integrity check failed")

    def save():
        saves.append(json.loads(json.dumps(receipt)))

    return receipt, calls, saves, validations, lambda: m.run_phases(
        receipt, run_phase=phase, revalidate=validate, save=save,
    )


def test_four_phases_complete_in_fixed_order_with_states(monkeypatch, tmp_path):
    m = module(monkeypatch)
    receipt, calls, saves, _, execute = run(m, tmp_path)
    execute()
    assert calls == ["staging_identity", "synthetic_cpu", "synthetic_table", "dense_native"]
    assert receipt["status"] == "complete_engineering_measurements_only"
    assert all(v["status"] == "complete" for v in receipt["phases"].values())
    assert any(v["phases"]["dense_native"]["status"] == "pending" for v in saves)
    assert not receipt["full_run_eligible"]


def test_cpu_local_timeout_preserves_failure_then_runs_independent_table_dense(monkeypatch, tmp_path):
    m = module(monkeypatch)
    receipt, calls, saves, validations, execute = run(m, tmp_path, fail="synthetic_cpu")
    execute()
    assert calls == ["staging_identity", "synthetic_cpu", "synthetic_table", "dense_native"]
    assert receipt["status"] == "partial_engineering_measurements"
    assert receipt["phases"]["synthetic_cpu"]["status"] == "failed"
    assert any(s["phases"]["synthetic_cpu"]["status"] == "failed" and
               s["phases"]["synthetic_table"]["status"] == "pending" for s in saves)
    assert len(validations) >= 5  # Fresh validation also follows the known local failure.
    assert receipt["session_started"] == 10.0 and receipt["engineering_started"] == 20.0


@pytest.mark.parametrize("phase,kind", [
    ("staging_identity", "timeout"), ("synthetic_cpu", "global deadline"),
    ("synthetic_cpu", "storage"), ("synthetic_cpu", "RSS"),
    ("synthetic_cpu", "integrity"), ("synthetic_cpu", "child error"),
    ("synthetic_cpu", "missing_receipt"), ("synthetic_cpu", "cleanup"),
    ("synthetic_cpu", "drain"),
])
def test_composed_fatal_faults_record_then_block_future_work(monkeypatch, tmp_path, phase, kind):
    m = module(monkeypatch)
    receipt, calls, saves, _, execute = run(m, tmp_path, fail=phase, kind=kind)
    with pytest.raises((RuntimeError, TimeoutError, FileNotFoundError)):
        execute()
    assert calls[-1] == phase
    assert receipt["status"] == "failed"
    assert receipt["phases"][phase]["status"] == "failed"
    assert receipt["phases"]["dense_native"]["status"] == "blocked"
    assert saves[-1]["status"] == "failed"


def test_post_timeout_source_validation_failure_blocks_continuation(monkeypatch, tmp_path):
    m = module(monkeypatch)
    receipt, calls, _, _, execute = run(m, tmp_path, fail="synthetic_cpu", bad_validation=True)
    with pytest.raises(RuntimeError, match="source/integrity"):
        execute()
    assert calls == ["staging_identity", "synthetic_cpu"]
    assert receipt["phases"]["synthetic_cpu"]["original_error"] == "Local phase ceiling exhausted"
    assert receipt["phases"]["synthetic_table"]["status"] == "blocked"


def test_exhausted_cumulative_engineering_clock_never_starts_next_node(monkeypatch, tmp_path):
    m = module(monkeypatch)
    receipt, calls, _, _, execute = run(m, tmp_path)
    receipt["engineering_started"] = 10.0
    monkeypatch.setattr(m.time, "monotonic", lambda: 3611.0)
    with pytest.raises(TimeoutError):
        execute()
    assert calls == [] and receipt["status"] == "failed"


@pytest.mark.parametrize("fault", [None, "cleanup", "drain", "drain_race", "storage", "global", "child_exit"])
def test_actual_supervisor_timeout_is_recoverable_only_after_safe_cleanup(monkeypatch, tmp_path, fault):
    m = module(monkeypatch)
    clock = [0.0]
    monkeypatch.setattr(m.time, "monotonic", lambda: clock[0])
    process = SimpleNamespace(pid=123, returncode=None, poll=lambda: None, stdout=None)

    def start(*args, **kwargs):
        clock[0] = 601.0
        return process

    def stop(child):
        child.returncode = 1 if fault == "child_exit" else -15
        if fault == "cleanup":
            raise RuntimeError("generated cleanup failure")
        if fault == "global":
            clock[0] = 3601.0

    def usage(roots):
        if process.returncode is not None and fault == "storage":
            raise RuntimeError("generated post-failure storage violation")
        return dict(apparent=0, allocated=0)

    class Thread:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def join(self, **kwargs):
            if fault == "drain":
                raise RuntimeError("generated drain failure")

        def is_alive(self):
            return fault == "drain_race"

    cycle = SimpleNamespace(dataset_usage=usage, _stop_group=stop)
    helper = SimpleNamespace(write_json=lambda p, v: Path(p).write_text(json.dumps(v)), sanitized=str)
    monkeypatch.setattr(m, "load_script", lambda root, name: cycle if name == "kaggle_cycle" else helper)
    monkeypatch.setattr(m.subprocess, "Popen", start)
    monkeypatch.setattr(m.threading, "Thread", Thread)
    with pytest.raises((m.PhaseTimeout, RuntimeError, TimeoutError)) as raised:
        m.supervise(["generated"], project=tmp_path, output=tmp_path / "guard",
                    roots=[tmp_path], session_started=0.0, engineering_started=0.0, phase="dense_native")
    terminal = json.loads((tmp_path / "guard/receipt.json").read_text())
    assert terminal["status"] == "failed"
    assert terminal["exit_code"] == (1 if fault == "child_exit" else -15)
    assert "ceiling exhausted" in terminal["original_error"]
    assert isinstance(raised.value, m.PhaseTimeout) is (fault is None)
    if fault is None:
        assert raised.value.receipt == terminal
        assert terminal["post_failure_safety_validated"]
    else:
        assert terminal.get("timeout_scope") is None


def test_ordinary_timeout_type_never_authorizes_independent_work(monkeypatch, tmp_path):
    m = module(monkeypatch)
    receipt, calls, _, _, execute = run(m, tmp_path)
    original = m.run_phases

    def failed(name):
        raise TimeoutError("Unknown child timeout, no supervisor authority")

    with pytest.raises(TimeoutError):
        original(receipt, run_phase=failed, revalidate=lambda: None, save=lambda: None)
    assert receipt["phases"]["synthetic_table"]["status"] == "blocked"
    assert calls == []


@pytest.mark.parametrize("child_error", [False, True])
def test_composed_real_supervisor_routes_only_its_safe_phase_timeout(monkeypatch, tmp_path, child_error):
    m = module(monkeypatch)
    clock = [0.0]
    monkeypatch.setattr(m.time, "monotonic", lambda: clock[0])
    calls = []

    def start(command, **kwargs):
        phase = command[0]
        calls.append(phase)
        clock[0] += 1801 if phase == "synthetic_cpu" else 1
        process = SimpleNamespace(pid=123, returncode=None if phase == "synthetic_cpu" else 0)
        process.poll = lambda: process.returncode
        return process

    def stop(process):
        if process.returncode is None:
            process.returncode = 1 if child_error else -15

    class Thread:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def join(self, **kwargs):
            pass

        def is_alive(self):
            return False

    cycle = SimpleNamespace(dataset_usage=lambda roots: dict(apparent=0, allocated=0), _stop_group=stop)
    helper = SimpleNamespace(write_json=lambda p, v: Path(p).write_text(json.dumps(v)), sanitized=str)
    monkeypatch.setattr(m, "load_script", lambda root, name: cycle if name == "kaggle_cycle" else helper)
    monkeypatch.setattr(m.subprocess, "Popen", start)
    monkeypatch.setattr(m.threading, "Thread", Thread)
    receipt = dict(session_started=0.0, engineering_started=0.0, source_identity="generated", phases={})

    def execute():
        m.run_phases(receipt, run_phase=lambda phase: m.supervise(
            [phase], project=tmp_path, output=tmp_path / phase, roots=[tmp_path],
            session_started=0.0, engineering_started=0.0, phase=phase,
        ), revalidate=lambda: None, save=lambda: (tmp_path / "outer.json").write_text(json.dumps(receipt)))

    if child_error:
        with pytest.raises(RuntimeError):
            execute()
        assert calls == ["staging_identity", "synthetic_cpu"]
        assert receipt["phases"]["dense_native"]["status"] == "blocked"
    else:
        execute()
        assert calls == ["staging_identity", "synthetic_cpu", "synthetic_table", "dense_native"]
        assert receipt["status"] == "partial_engineering_measurements"
        assert receipt["phases"]["synthetic_cpu"]["status"] == "failed"
        assert receipt["phases"]["dense_native"]["status"] == "complete"
    assert receipt["session_started"] == receipt["engineering_started"] == 0.0
