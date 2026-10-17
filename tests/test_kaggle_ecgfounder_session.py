import copy
import hashlib
import importlib.util
import json
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

needs_addendum = pytest.mark.skipif(
    not (
        Path(__file__).resolve().parents[1] / "reports/ml-investigation/next-cycle/ecgfounder-cycle-20261013/calendar-addendum-20261017.json"
    ).exists(),
    reason="requires private research receipts not shipped in the public repo",
)


def load():
    name = Path(__file__).resolve().parents[1] / "scripts/kaggle_ecgfounder_session.py"
    spec = importlib.util.spec_from_file_location("session_bootstrap_test", name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bootstrap_has_one_original_clock_and_no_submission_api():
    m = load()
    assert m.BOOTSTRAP_OPERATIONS["supervised_worker"] == 1
    assert not hasattr(m, "push")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return sha(path)


def fixture(tmp_path, monkeypatch):
    m = load()
    root = tmp_path / "prepared"
    root.mkdir()
    source = {}
    for name in m.REQUIRED:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(m.__file__).read_bytes() if name == m.ENTRY else b"generated source")
        source[name] = sha(target)
    (root / "pyproject.toml").write_text('[project]\ndependencies=["fake-dependency==1"]\n')
    source["pyproject.toml"] = sha(root / "pyproject.toml")
    actual = Path(__file__).resolve().parents[1]
    calendar = root / m.CALENDAR
    calendar.parent.mkdir(parents=True, exist_ok=True)
    calendar.write_bytes((actual / m.CALENDAR).read_bytes())
    files = {
        m.CALENDAR: sha(calendar),
        "data/cache/features.npy": "b" * 64,
        m.CYCLE + "/weights/1_lead_ECGFounder.pth": "c" * 64,
    }
    files["cost.json"] = write(
        root / "cost.json",
        dict(bootstrap_operations=m.BOOTSTRAP_OPERATIONS, session_operation_scope=m.SESSION_OPERATION_SCOPE),
    )
    files["review.json"] = write(
        root / "review.json",
        dict(
            status="passed",
            source_sha256=source,
            cost_sha256=files["cost.json"],
            bootstrap_operations=m.BOOTSTRAP_OPERATIONS,
            session_operation_scope=m.SESSION_OPERATION_SCOPE,
        ),
    )
    extraction = dict(
        source_sha256=source, files_sha256=files.copy(), cost_receipt="cost.json", cost_review="review.json"
    )
    files["extraction.json"] = write(root / "extraction.json", extraction)
    files["template.json"] = write(
        root / "template.json", dict(source_sha256=source, files_sha256=files.copy())
    )
    value = dict(
        version="ecgfounder-session-v1",
        source_sha256=source,
        files_sha256=files,
        extraction_declaration="extraction.json",
        study_template="template.json",
        output="results",
        staging={"data/cache/features.npy": "cache/features.npy"},
        input_data_root="datasets/owner/private-generated",
    )
    digest = write(root / "session.json", value)
    real_load = m.load_script
    helper = real_load(actual, "kaggle_ecgfounder_probe")
    monkeypatch.setattr(m, "load_script", lambda root, name: helper)
    payload = m.package(
        root,
        root / "session.json",
        digest,
        root / "package",
        "sumit1703/cardiotwin-ecgfounder-session-20261017-generated",
    )
    return SimpleNamespace(m=m, root=root, payload=payload, value=value, digest=digest)


def remote_fixture(tmp_path, monkeypatch, fault=None):
    h = fixture(tmp_path, monkeypatch)
    working, inputs = tmp_path / "working", tmp_path / "inputs"
    working.mkdir()
    inputs.mkdir()
    monkeypatch.setattr(h.m, "WORKING", working)
    monkeypatch.setattr(h.m, "INPUTS", inputs)
    log = []
    monkeypatch.setattr(h.m, "arm_clock", lambda start, deadline: log.append(("clock", start, deadline)))
    monkeypatch.setattr(h.m.signal, "setitimer", lambda *args: log.append(("timer", *args)))

    def supervise(command, **kwargs):
        log.append(("supervise", command, kwargs))
        output, project = kwargs["output"], kwargs["project"]
        output.mkdir()
        result = dict(
            status="complete",
            exit_code=0,
            session_started=kwargs["session_started"],
            diagnostic_storage={"apparent": 100, "allocated": 100},
        )
        if fault == "supervisor":
            result["exit_code"] = 1
        if fault == "diagnostic":
            result["diagnostic_storage"]["apparent"] = 249500000
        write(output / "receipt.json", result)
        write(project / "results/bundle/artifact-manifest.json", {"files_sha256": {}})
        write(
            project / "results/session-receipt.json",
            dict(
                status="work_complete_pending_supervisor",
                independent_replay_complete=fault != "replay",
                session_started=kwargs["session_started"],
                session_declaration_sha256=h.digest if fault != "declaration" else "e" * 64,
                bundle_manifest_sha256=sha(project / "results/bundle/artifact-manifest.json"),
            ),
        )
        if fault == "source_stamp":
            (project / "execution-source-manifest.json").write_text("{}")
        if fault == "bundle_manifest":
            (project / "results/bundle/artifact-manifest.json").write_text('{"changed":true}')
        return result

    deadline = datetime.fromisoformat(
        json.loads((h.root / h.m.CALENDAR).read_text())["new_work_calendar_deadline_utc"]
    )
    monkeypatch.setattr(
        h.m, "load_script", lambda *args: SimpleNamespace(DEADLINE=deadline, supervise=supervise)
    )
    return h, log


@needs_addendum
def test_package_contains_only_source_and_bound_small_metadata(tmp_path, monkeypatch):
    h = fixture(tmp_path, monkeypatch)
    assert not any(name.startswith("data/") or "weights" in Path(name).parts for name in h.payload["files"])
    metadata = json.loads((h.root / "package/kernel-metadata.json").read_text())
    assert metadata["is_private"] and metadata["enable_gpu"]
    assert metadata["id"] == h.payload["identity_name"]
    assert h.payload["manifest"][h.m.ENTRY] == sha(h.root / h.m.ENTRY)
    assert "BOOTSTRAP_STARTED=time.monotonic()" in (h.root / "package/train.py").read_text()
    with pytest.raises(FileExistsError):
        h.m.package(h.root, h.root / "session.json", h.digest, h.root / "package", h.payload["identity_name"])


@needs_addendum
def test_outer_supervisor_covers_dependency_exec_and_pending_bundle_then_seals(tmp_path, monkeypatch):
    h, log = remote_fixture(tmp_path, monkeypatch)
    original = time.monotonic() - 100
    result = h.m.remote(h.payload, session_started=original)
    assert result["status"] == "complete" and result["session_started"] == original
    launch = next(item for item in log if item[0] == "supervise")
    command, kwargs = launch[1:]
    assert kwargs["session_started"] == original and kwargs["engineering_started"] is None
    assert kwargs["phase"] == "staging_identity"
    assert command[1].endswith(h.m.ENTRY)
    assert command[command.index("--session-start-monotonic") + 1] == str(original)
    assert kwargs["table_root"] == kwargs["project"] / "results/extraction/tables"
    assert kwargs["env"]["TMPDIR"] == str(kwargs["project"] / "dependencies/.pip-tmp")
    seal = json.loads((kwargs["project"] / "results/bundle/completion-seal.json").read_text())
    assert seal["worker_receipt_sha256"] == sha(kwargs["project"] / "results/session-receipt.json")
    assert seal["supervisor_receipt_sha256"] == sha(kwargs["output"] / "receipt.json")
    assert (
        json.loads((kwargs["project"] / "results/session-receipt.json").read_text())["status"]
        == "work_complete_pending_supervisor"
    )


@pytest.mark.parametrize(
    "fault", ["supervisor", "replay", "source_stamp", "diagnostic", "declaration", "bundle_manifest"]
)
@needs_addendum
def test_failed_completion_never_seals_and_preserves_original_error(tmp_path, monkeypatch, fault):
    h, log = remote_fixture(tmp_path, monkeypatch, fault)
    with pytest.raises((ValueError, RuntimeError)):
        h.m.remote(h.payload, session_started=time.monotonic() - 1)
    project = h.m.WORKING / h.payload["identity_name"].split("/")[1]
    assert not (project / "results/bundle/completion-seal.json").exists()
    receipt = json.loads((project / "session-bootstrap/bootstrap-receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["original_error"]


@needs_addendum
def test_corrupt_source_capsule_rejected_before_overlay_or_launch(tmp_path, monkeypatch):
    h, log = remote_fixture(tmp_path, monkeypatch)
    changed = copy.deepcopy(h.payload)
    changed["files"][h.m.ENTRY] = "YmFk"
    with pytest.raises(ValueError, match="hash differ"):
        h.m.remote(changed, session_started=time.monotonic() - 1)
    assert not (h.m.WORKING / h.payload["identity_name"].split("/")[1]).exists()
    assert not any(item[0] == "supervise" for item in log)


@needs_addendum
def test_worker_pip_stays_in_group_then_execs_direct_caller_with_original_clock(tmp_path, monkeypatch):
    h = fixture(tmp_path, monkeypatch)
    project = tmp_path / "worker"
    h.m.overlay(project, h.payload)
    (project / "session-bootstrap").mkdir()
    monkeypatch.setattr(h.m, "__file__", str(project / h.m.ENTRY))
    monkeypatch.setenv("TMPDIR", str(project / "dependencies/.pip-tmp"))
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    calls = []

    def pip(command, **kwargs):
        calls.append(("pip", command, kwargs))

    def execve(executable, command, env):
        calls.append(("exec", command, env))
        raise RuntimeError("generated exec intercepted")

    monkeypatch.setattr(h.m.subprocess, "run", pip)
    monkeypatch.setattr(h.m.os, "execve", execve)
    original = time.monotonic() - 100
    args = SimpleNamespace(
        root=str(project),
        inputs=str(tmp_path / "inputs"),
        declaration="session.json",
        declaration_sha256=h.digest,
        session_start_monotonic=original,
        worker_parent_pid=h.m.os.getppid(),
    )
    with pytest.raises(RuntimeError, match="exec intercepted"):
        h.m.worker(args)
    assert calls[0][0] == "pip" and "start_new_session" not in calls[0][2]
    assert calls[0][2]["check"] is True
    command, env = calls[1][1:]
    assert command[1] == str(project / h.m.CALLER)
    assert command[command.index("--worker-parent-pid") + 1] == str(args.worker_parent_pid)
    assert command[command.index("--session-start-monotonic") + 1] == str(original)
    assert command[command.index("--data-root") + 1] == str(Path(args.inputs) / h.value["input_data_root"])
    assert env["TMPDIR"] == str(project / "scratch")


def test_hard_alarm_rejects_exhausted_original_clock_before_launch(monkeypatch):
    m = load()
    monkeypatch.setattr(time, "monotonic", lambda: 100_000.0)  # fresh CI runners have short uptime
    with pytest.raises(TimeoutError, match="reserve exhausted"):
        m.arm_clock(time.monotonic() - 41101, datetime.fromisoformat("2026-10-18T03:30:00+00:00"))


@needs_addendum
def test_generated_train_bootstraps_without_running_worker_argparse(tmp_path, monkeypatch):
    import builtins

    h = fixture(tmp_path, monkeypatch)
    captured = []

    def execute_wrapper(code, namespace):
        assert namespace["__name__"] != "__main__"
        builtins.exec(code, namespace)
        namespace["remote"] = lambda payload, **kwargs: captured.append((payload, kwargs))

    namespace = {"__name__": "__main__", "exec": execute_wrapper}
    builtins.exec((h.root / "package/train.py").read_text(), namespace)
    assert len(captured) == 1
    assert captured[0][0] == h.payload
    assert captured[0][1]["session_started"] == namespace["BOOTSTRAP_STARTED"]
