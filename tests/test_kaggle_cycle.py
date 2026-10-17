import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "kaggle_cycle", Path(__file__).resolve().parents[1] / "scripts/kaggle_cycle.py"
)
cycle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cycle)


def test_allocated_measure_is_opt_in_and_deduplicates_links(tmp_path):
    item = tmp_path / "tiny.bin"
    item.write_bytes(b"x")
    (tmp_path / "nested").mkdir()
    os.link(item, tmp_path / "nested" / "linked.bin")
    assert cycle.dataset_bytes([tmp_path, tmp_path / "nested"]) == 1
    assert cycle.dataset_bytes([tmp_path, tmp_path / "nested"], allocated=True) == item.stat().st_blocks * 512


def test_allocated_watchdog_catches_low_apparent_high_allocated(tmp_path, monkeypatch):
    monkeypatch.setattr(cycle, "dataset_bytes", lambda roots, *, allocated=False: cycle.DATASET_BYTES + 1 if allocated else 1)
    result = cycle.run_child([sys.executable, "-c", "import time;time.sleep(5)"], tmp_path / "allocated.log", 2,
                             storage_roots=[tmp_path], check_allocated=True)
    assert result["status"] == "failed"
    assert "storage cap" in result["original_error"]


def summary(mode, complete=True):
    return dict(
        seed=42,
        specs=[["rr", 16]],
        screening_epochs=8,
        screening_max_training_windows=10_000,
        early_stop_weighting=mode,
        status="complete" if complete else "budget_exhausted",
        failures=[],
        combined={"rr_tcn_16": {"complete_outer_folds": 3}} if complete else {},
    )


@pytest.mark.parametrize("exit_code", [0, 7])
def test_child_keeps_stdout_stderr_and_exit_code(tmp_path, capsys, exit_code):
    log = tmp_path / "child.log"
    result = cycle.run_child(
        [
            sys.executable,
            "-c",
            f"import sys; print('phase'); print('original failure', file=sys.stderr); sys.exit({exit_code})",
        ],
        log,
        5,
    )
    assert result["exit_code"] == exit_code
    assert result["status"] == ("finished" if exit_code == 0 else "failed")
    assert "phase" in log.read_text() and "original failure" in log.read_text()
    assert "original failure" in capsys.readouterr().out
    if exit_code:
        assert "original failure" in result["child_output_tail"]
    assert result["output_reader_finished"]


@pytest.mark.skipif(os.name != "posix", reason="Kaggle process groups require POSIX")
def test_timeout_kills_process_group_and_unblocks_descendant_output(tmp_path):
    script = (
        "import subprocess,sys,time,signal; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        "subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        "print('started',flush=True); time.sleep(30)"
    )
    started = time.monotonic()
    result = cycle.run_child([sys.executable, "-u", "-c", script], tmp_path / "timeout.log", 0.2)
    assert result["status"] == "timed_out"
    assert result["exit_code"] != 0
    assert result["output_reader_finished"]
    assert time.monotonic() - started < 5
    assert "started" in (tmp_path / "timeout.log").read_text()


def test_child_stops_when_dataset_cap_exceeded(tmp_path, monkeypatch):
    monkeypatch.setattr(cycle, "dataset_bytes", lambda _: cycle.DATASET_BYTES + 1)
    result = cycle.run_child(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        tmp_path / "storage.log",
        5,
        storage_roots=[tmp_path],
    )
    assert result["status"] == "failed"
    assert "storage cap exceeded" in result["original_error"]


@pytest.mark.parametrize("complete", [False, True])
def test_summary_requires_complete_recipe_bound_three_folds(tmp_path, complete):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary("patient_only", complete)))
    assert cycle.classify_arm(path, "patient_only")[0] == ("complete" if complete else "partial")
    assert cycle.classify_arm(path, "patient_class")[0] == "failed"
    assert cycle.classify_arm(tmp_path / "missing", "patient_only")[0] == "partial"


@pytest.fixture
def inputs(tmp_path):
    data, cache = tmp_path / "data", tmp_path / "cache"
    data.mkdir()
    cache.mkdir()
    split = tmp_path / "split.json"
    split.write_text('{"seed":42}')
    return dict(data=data, cache=cache, split=split, output=tmp_path / "cycle", session_seconds_remaining=400)


@pytest.mark.parametrize("complete", [False, True])
def test_cycle_uses_shared_cache_matched_recipe_and_retains_partial(tmp_path, monkeypatch, inputs, complete):
    calls = []

    def child(command, log_path, timeout_seconds, storage_roots):
        calls.append((command, timeout_seconds))
        output = Path(command[command.index("--output") + 1])
        output.mkdir()
        mode = command[command.index("--early-stop-weighting") + 1]
        (output / "summary.json").write_text(json.dumps(summary(mode, complete)))
        return dict(status="finished", exit_code=0, original_error=None, wall_seconds=0.001)

    monkeypatch.setattr(cycle, "run_child", child)
    result = cycle.run_cycle(**inputs, arm_seconds=50)
    assert result["status"] == ("complete" if complete else "incomplete")
    assert len(calls) == (2 if complete else 1)
    for command, cap in calls:
        assert cap <= 50
        assert command[command.index("--models") + 1] == "rr16"
        assert command[command.index("--sequence-cache") + 1] == str(inputs["output"] / "rr-sequences.sqlite")
    stored = json.loads((inputs["output"] / "receipt.json").read_text())
    assert stored["status"] == result["status"]
    assert "supervisor only" in stored["telemetry_scope"]
    with pytest.raises(FileExistsError):
        cycle.run_cycle(**inputs)


@pytest.mark.parametrize("remaining,arm", [(300, 1), (41401, 1), (float("nan"), 1), (400, 0), (400, 20161)])
def test_invalid_caps_fail_before_output_creation(inputs, remaining, arm):
    inputs["session_seconds_remaining"] = remaining
    with pytest.raises(ValueError):
        cycle.run_cycle(**inputs, arm_seconds=arm)
    assert not inputs["output"].exists()


def test_dataset_count_does_not_double_count_nested_roots(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "data").write_bytes(b"123")
    assert cycle.dataset_bytes([tmp_path, nested]) == 3


def test_dataset_count_deduplicates_hardlinks_and_symlinks(tmp_path):
    original = tmp_path / "data"
    original.write_bytes(b"12345")
    os.link(original, tmp_path / "hardlink")
    (tmp_path / "symlink").symlink_to(original)
    (tmp_path / "broken").symlink_to(tmp_path / "missing")
    assert cycle.dataset_bytes([tmp_path]) == 5


def test_cycle_requires_rr_cache_and_output_headroom_before_launch(monkeypatch, inputs):
    monkeypatch.setattr(cycle, "dataset_bytes", lambda _: cycle.DATASET_BYTES - cycle.RR_CACHE_BYTES)
    with pytest.raises(ValueError, match="headroom"):
        cycle.run_cycle(**inputs)
    assert not inputs["output"].exists()


def test_cycle_retains_launch_error_receipt(monkeypatch, inputs):
    def fail(*args, **kwargs):
        raise OSError("original spawn failure")

    monkeypatch.setattr(cycle, "run_child", fail)
    with pytest.raises(OSError, match="original spawn failure"):
        cycle.run_cycle(**inputs)
    receipt = json.loads((inputs["output"] / "receipt.json").read_text())
    assert receipt["status"] == receipt["arms"][0]["status"] == "failed"
    assert receipt["original_error"] == receipt["arms"][0]["original_error"] == "original spawn failure"


@pytest.mark.skipif(os.name != "posix", reason="Kaggle process groups require POSIX")
def test_sigterm_unwinds_cycle_and_terminates_separate_child_group(tmp_path, inputs):
    ready = tmp_path / "child.pid"
    runner = tmp_path / "runner.py"
    runner.write_text(f"import os,time\nopen({str(ready)!r}, 'w').write(str(os.getpid()))\ntime.sleep(30)\n")
    code = (
        "import importlib.util,signal; "
        f"spec=importlib.util.spec_from_file_location('cycle',{cycle.__file__!r}); "
        "cycle=importlib.util.module_from_spec(spec); spec.loader.exec_module(cycle); "
        f"cycle.RUNNER={str(runner)!r}; "
        "signal.signal(signal.SIGTERM,cycle._termination_handler); "
        f"cycle.run_cycle({str(inputs['data'])!r},{str(inputs['cache'])!r},"
        f"{str(inputs['split'])!r},{str(inputs['output'])!r},1000)"
    )
    supervisor = subprocess.Popen(
        [sys.executable, "-c", code], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and supervisor.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), "Synthetic training child did not start"
        child_pid = int(ready.read_text())
        supervisor.send_signal(signal.SIGTERM)
        _, stderr = supervisor.communicate(timeout=5)
        assert supervisor.returncode != 0
        assert b"Supervisor received SIGTERM" in stderr
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        receipt = json.loads((inputs["output"] / "receipt.json").read_text())
        assert receipt["status"] == receipt["arms"][0]["status"] == "failed"
        assert receipt["original_error"] == "Supervisor received SIGTERM"
    finally:
        if supervisor.poll() is None:
            supervisor.kill()
            supervisor.wait(timeout=5)
