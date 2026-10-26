import json
import threading
import time

import pytest


def test_progress_rate_limits_updates_and_retains_final_counts(monkeypatch, capsys):
    from cardiotwin import progress

    now = [10.0]
    monkeypatch.setattr(progress.time, "monotonic", lambda: now[0])
    stage = progress.Progress("fixture", total=4, interval_seconds=30, fold=2)
    stage.update(1)
    now[0] = 40
    stage.update(2, batches=1)
    stage.update(3)
    stage.finish()
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event["status"] for event in events] == ["started", "running", "done"]
    assert events[-1]["completed"] == 3
    assert events[-1]["total"] == 4
    assert events[-1]["fold"] == 2
    assert events[-1]["elapsed_seconds"] == 30
    assert events[-1]["batches"] == 1
    assert all(event["timestamp_utc"].endswith("+00:00") for event in events)


def test_progress_forced_update_and_failure_preserve_original_error(capsys):
    from cardiotwin.progress import Progress

    with pytest.raises(ValueError, match="private payload"):
        with Progress("fixture") as stage:
            stage.update(2, force=True)
            raise ValueError("private payload")
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event["status"] for event in events] == ["started", "running", "failed"]
    assert events[-1]["completed"] == 2
    assert events[-1]["error_type"] == "ValueError"
    assert "private payload" not in json.dumps(events)


def test_progress_flushes_every_emitted_json_line(monkeypatch):
    from cardiotwin.progress import Progress

    calls = []
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: calls.append((args, kwargs)))
    stage = Progress("fixture")
    stage.finish(status="failed", error_type="TimeoutError")
    assert len(calls) == 2
    assert all(kwargs.get("flush") is True for _, kwargs in calls)
    assert json.loads(calls[-1][0][0])["status"] == "failed"


def test_explicit_finish_inside_context_does_not_duplicate(capsys):
    from cardiotwin.progress import Progress

    with Progress("fixture", total=1) as stage:
        stage.update(1)
        stage.finish()
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event["status"] for event in events] == ["started", "done"]


@pytest.mark.parametrize("error", [BrokenPipeError("closed pipe"), ValueError("closed stream")])
def test_output_failure_never_replaces_workload_exception(monkeypatch, error):
    from cardiotwin.progress import Progress

    def broken_sink(*args, **kwargs):
        raise error

    monkeypatch.setattr("builtins.print", broken_sink)
    with Progress("successful_work"):
        pass
    with pytest.raises(RuntimeError, match="original training error"):
        with Progress("failed_work"):
            raise RuntimeError("original training error")


def test_concurrent_events_remain_independent_json_lines(monkeypatch):
    from cardiotwin.progress import Progress

    output = []

    def slow_sink(payload, **kwargs):
        output.append(payload)
        time.sleep(0.002)
        output.append("\n")

    monkeypatch.setattr("builtins.print", slow_sink)
    barrier = threading.Barrier(4)

    def emit(number):
        barrier.wait()
        with Progress("concurrent", writer=number):
            pass

    threads = [threading.Thread(target=emit, args=(number,)) for number in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    events = [json.loads(line) for line in "".join(output).splitlines()]
    assert len(events) == 8
    assert sorted(event["writer"] for event in events) == [0, 0, 1, 1, 2, 2, 3, 3]
