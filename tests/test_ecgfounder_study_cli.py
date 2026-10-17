import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_cli():
    path = Path(__file__).resolve().parents[1] / "scripts/run_ecgfounder_study.py"
    spec = importlib.util.spec_from_file_location("study_cli_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_requires_original_bootstrap_clock_and_replay_receipt(tmp_path):
    cli = load_cli()
    args = [
        "--root",
        str(tmp_path),
        "--declaration",
        "decl.json",
        "--declaration-sha256",
        "a" * 64,
        "--output",
        str(tmp_path / "out"),
        "--mode",
        "run",
    ]
    with pytest.raises(SystemExit):
        cli.main(args)
    with pytest.raises(SystemExit):
        cli.main([*args[:-1], "replay", "--session-start-monotonic", str(time.monotonic())])


@pytest.mark.parametrize("started", [float("nan"), float("inf"), -1, "future", "exhausted"])
def test_bootstrap_clock_fails_closed(started):
    cli = load_cli()
    if started == "future":
        started = time.monotonic() + 10
    elif started == "exhausted":
        started = time.monotonic() - 41101
    with pytest.raises((ValueError, TimeoutError)):
        cli.work_deadline(started)


def test_live_processgroup_rss_parser_sums_descendants_and_rejects_missing():
    cli = load_cli()
    assert cli.parse_group_rss("10 10 2\n11 10 3\n12 12 8\n", 10) == 5120
    with pytest.raises(RuntimeError):
        cli.parse_group_rss("12 12 8\n", 10)


def test_independent_watchdog_terminates_uncooperative_group_and_preserves_error(tmp_path, monkeypatch):
    cli = load_cli()
    monkeypatch.setattr(cli, "group_rss", lambda process, remaining: 1)
    started = time.monotonic()
    result = cli.supervise(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        tmp_path / "out",
        root=Path(__file__).resolve().parents[1],
        session_started=started,
        deadline=started + 0.2,
        storage_check=lambda: dict(allocated_bytes=0, logical_bytes=0),
    )
    assert result["status"] == "failed" and result["error_type"] == "TimeoutError"
    assert result["process_group_terminated"] and result["child_exit_code"] != 0
    assert time.monotonic() - started < 5
    assert json.loads((tmp_path / "out/supervisor.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("fault", ["rss", "storage", "source"])
def test_watchdog_resource_or_unbound_source_failure_cannot_launch_useful_work(tmp_path, monkeypatch, fault):
    cli = load_cli()
    if fault == "source":
        declaration = tmp_path / "declaration.json"
        declaration.write_text(json.dumps(dict(source_sha256={})))
        with pytest.raises(ValueError, match="source|Source"):
            cli.source_authority(tmp_path, declaration, cli.file_hash(declaration), lambda: None)
        return
    monkeypatch.setattr(cli, "group_rss", lambda process, remaining: 10_000_000_001 if fault == "rss" else 1)
    result = cli.supervise(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        tmp_path / "out",
        root=Path(__file__).resolve().parents[1],
        session_started=time.monotonic(),
        deadline=time.monotonic() + 10,
        storage_check=lambda: dict(
            allocated_bytes=25_000_000_000 if fault == "storage" else 0, logical_bytes=0
        ),
    )
    assert result["status"] == "failed"
    if fault == "storage":
        assert result["child_pid"] is None
    else:
        assert result["process_group_terminated"]


@pytest.mark.parametrize("mode", ["preflight", "support", "run", "replay"])
def test_worker_dispatches_stage_under_original_clock_and_closes_preparation(tmp_path, monkeypatch, mode):
    cli = load_cli()
    args = SimpleNamespace(
        root=str(tmp_path),
        declaration="bound.json",
        declaration_sha256="a" * 64,
        output=str(tmp_path),
        mode=mode,
        worker_parent_pid=__import__("os").getppid(),
        session_start_monotonic=time.monotonic() - 100,
        run_dir="run",
        run_receipt_sha256="b" * 64,
    )
    calls = []
    prepared = SimpleNamespace(receipt=dict(status="prepared"), close=lambda: calls.append("closed"))
    monkeypatch.setattr(cli, "source_authority", lambda *a: calls.append("source"))

    def prepare(*a, **kwargs):
        kwargs["guard"]()
        calls.append("prepared")
        return prepared

    monkeypatch.setattr(cli, "prepare_study", prepare)
    monkeypatch.setattr(cli, "audit_common_support", lambda *a, **kw: dict(status="passed"))

    def stage(*a, **kwargs):
        kwargs["guard"]()
        assert kwargs["deadline"] <= args.session_start_monotonic + 41100
        calls.append(mode)
        if mode == "replay":
            assert kwargs["expected_receipt_sha256"] == args.run_receipt_sha256
        return dict(status="complete")

    monkeypatch.setattr(cli, "execute_study", stage)
    monkeypatch.setattr(cli, "replay_study", stage)
    assert cli.worker(args) == 0
    assert calls[:2] == ["source", "prepared"] and calls[-1] == "closed"


def test_worker_preserves_original_error_when_diagnostic_write_also_fails(tmp_path, monkeypatch):
    cli = load_cli()
    args = SimpleNamespace(
        worker_parent_pid=__import__("os").getppid(),
        output=str(tmp_path),
        root=str(tmp_path),
        session_start_monotonic=time.monotonic(),
        declaration="d",
        declaration_sha256="a" * 64,
    )

    def original(*args):
        raise ValueError("original source failure")

    monkeypatch.setattr(cli, "source_authority", original)
    monkeypatch.setattr(cli, "_write", lambda *args: (_ for _ in ()).throw(OSError("diagnostic failure")))
    with pytest.raises(ValueError, match="original source failure"):
        cli.worker(args)
    assert "original source failure" in (tmp_path / "original-error.txt").read_text()


def test_watchdog_failure_receipt_write_cannot_mask_timeout_or_skip_group_cleanup(tmp_path, monkeypatch):
    cli = load_cli()
    monkeypatch.setattr(cli, "group_rss", lambda process, remaining: 1)
    write = cli._write

    def diagnostics(path, value):
        if value["status"] == "failed":
            raise OSError("diagnostic failure")
        return write(path, value)

    monkeypatch.setattr(cli, "_write", diagnostics)
    started = time.monotonic()
    result = cli.supervise(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        tmp_path / "out",
        root=Path(__file__).resolve().parents[1],
        session_started=started,
        deadline=started + 0.2,
        storage_check=lambda: dict(allocated_bytes=0, logical_bytes=0),
    )
    assert result["status"] == "failed" and result["error_type"] == "TimeoutError"
    assert result["process_group_terminated"] and result["child_exit_code"] != 0
