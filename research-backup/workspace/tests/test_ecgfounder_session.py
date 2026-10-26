import gzip
import hashlib
import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import HistoricalECGFounderDatetime

pytestmark = pytest.mark.usefixtures("historical_ecgfounder_clock")


needs_addendum = pytest.mark.skipif(
    not (
        Path(__file__).resolve().parents[1] / "reports/ml-investigation/next-cycle/ecgfounder-cycle-20261013/calendar-addendum-20261021.json"
    ).exists(),
    reason="requires private research receipts not shipped in the public repo",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, sort_keys=True).encode()
    path.write_bytes(gzip.compress(data, mtime=0) if path.name.endswith(".json.gz") else data)
    return sha(path)


def load():
    path = Path(__file__).resolve().parents[1] / "scripts/run_ecgfounder_session.py"
    spec = importlib.util.spec_from_file_location("session_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "datetime"):
        module.datetime = HistoricalECGFounderDatetime
    real_load_script = module.load_script

    def load_script(root, name):
        loaded = real_load_script(root, name)
        if hasattr(loaded, "datetime"):
            loaded.datetime = HistoricalECGFounderDatetime
        return loaded

    module.load_script = load_script
    return module


def fixture(tmp_path, monkeypatch, fault=None):
    m = load()
    root, inputs = tmp_path / "project", tmp_path / "inputs"
    root.mkdir()
    inputs.mkdir()
    log = []
    source = {}
    for name in m.REQUIRED_SOURCE:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(Path(m.__file__).read_bytes() if name == m.ENTRY else b"synthetic source")
        source[name] = sha(path)
    static = {}
    for name, value in {
        "cost.json": {
            "scope": "generated",
            "session_operations": {} if fault == "inventory" else m.SESSION_OPERATIONS,
        },
        "cost-review.json": {"status": "passed", "source_sha256": source},
        "raw.json": {"files_sha256": {"record_ecg.h5": sha_bytes(b"fake raw")}},
    }.items():
        static[name] = write(root / name, value)
    (inputs / "record_ecg.h5").write_bytes(b"fake raw")
    static["cost-review.json"] = write(
        root / "cost-review.json",
        {
            "status": "passed",
            "source_sha256": source,
            "cost_sha256": static["cost.json"],
            "session_operations": m.SESSION_OPERATIONS,
        },
    )
    stage = {}
    for name in ("index.json", "compact.json", "metadata.npy", "features.npy"):
        target, origin = "data/cache/" + name, "cache/" + name
        path = inputs / origin
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"generated cache " + name.encode())
        static[target] = sha(path)
        stage[target] = origin
    static[m.CHECKPOINT_PATH] = m.CHECKPOINT_SHA256
    extraction = dict(
        version="ecgfounder-full-extraction-v1",
        files_sha256=dict(static),
        source_sha256=source,
        cache="data/cache",
        roles="roles.json",
        raw_manifest="raw.json",
        compact_binding={"fixture": True},
        cost_receipt="cost.json",
        cost_review="cost-review.json",
        metadata_identity_sha256="a" * 64,
        table_byte_cap=2619117060,
    )
    static["extraction.json"] = write(root / "extraction.json", extraction)
    dynamic = [
        "results/extraction/" + n
        for n in ("receipt.json", "preflight.json", "coverage.json", "tables/receipt.json")
    ]
    template = dict(
        version="ecgfounder-study-v1",
        files_sha256={**static, **dict.fromkeys(dynamic)},
        source_sha256=source,
        cache="data/cache",
        metadata_root="data/annotations",
        roles="roles.json",
        raw_manifest="raw.json",
        compact_binding=extraction["compact_binding"],
        cost_receipt="cost.json",
        cost_review="cost-review.json",
        allowed_stages=["preflight", "support", "run", "replay"],
        extraction_declaration="extraction.json",
        runner_receipt=dynamic[0],
        runner_preflight=dynamic[1],
        coverage_receipt=dynamic[2],
        tables="results/extraction/tables",
    )
    static["template.json"] = write(root / "template.json", template)
    value = dict(
        version="ecgfounder-session-v1",
        files_sha256=static,
        source_sha256=source,
        extraction_declaration="extraction.json",
        study_template="template.json",
        study_declaration="results/study-declaration.json",
        output="results",
        staging=stage,
    )
    declaration = root / "session.json"
    digest = write(declaration, value)

    def cost(*args):
        log.append("cost")
        if fault == "cost":
            raise ValueError("original cost gate")
        return dict(estimated_complete_seconds=1e9 if fault == "remaining_cost" else 0.1)

    def staged(input_root, project, mapping, pins, *, guard, **kwargs):
        log.append("stage")
        for target, origin in mapping.items():
            guard()
            destination = project / target
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((input_root / origin).read_bytes())
            assert sha(destination) == pins[target]
        return dict(copy_bytes=1)

    def checkpoint(*args):
        log.append("checkpoint")
        return dict(verified=True)

    def preflight(*args, **kwargs):
        log.append(("preflight", kwargs["guard"].session_started))
        kwargs["guard"]()
        return dict(root=root, source=source, raw_hashes={"record_ecg.h5": sha(inputs / "record_ecg.h5")})

    def extract(prepared, output, **kwargs):
        log.append(("extract", kwargs["session_started"]))
        cells = [
            dict(fold=f, role=r, **m.coverage_cell(100, 100, 100)) for f in range(3) for r in m.ROLE_NAMES
        ]
        if fault == "coverage_membership":
            cells[-1] = cells[0]
        if fault == "coverage_fraction":
            cells[0] = dict(fold=0, role=m.ROLE_NAMES[0], **m.coverage_cell(100, 100, 94))
        coverage = dict(
            cells=cells, passed=fault != "coverage", rows=m.EXPECTED_ROWS, identity_sha256="a" * 64
        )
        result = dict(
            status="complete",
            coverage_passed=fault != "coverage",
            tables_complete=True,
            rows_completed=m.EXPECTED_ROWS,
        )
        write(output / "preflight.json", {"status": "preflight_passed_not_executed"})
        write(output / "tables/receipt.json", {"status": "complete", "complete": True})
        write(output / "coverage.json", coverage)
        write(output / "receipt.json", result)
        if fault == "template":
            template["allowed_stages"] = ["run"]
            write(root / "template.json", template)
        return result

    def prepare(path, digest, **kwargs):
        log.append("labels")
        concrete = json.loads(path.read_text())
        assert all(concrete["files_sha256"][n] == sha(root / n) for n in dynamic)
        assert {k: v for k, v in concrete.items() if k != "files_sha256"} == {
            k: v for k, v in template.items() if k != "files_sha256"
        }
        return SimpleNamespace(close=lambda: log.append("close"))

    def support(*args, **kwargs):
        log.append("support")
        return dict(status="failed" if fault == "support" else "passed")

    def execute(prepared, support, output, **kwargs):
        log.append("fit")
        result = dict(status="complete", decision="reject", completed_arms=6)
        write(output / "summary.json.gz", {"fixture": True})
        result["artifact_manifest_sha256"] = write(
            output / "artifact-manifest.json",
            {"files_sha256": {"summary.json.gz": sha(output / "summary.json.gz")}},
        )
        write(output / "receipt.json", result)
        return result

    def replay(prepared, run, output, **kwargs):
        log.append("replay")
        assert kwargs["expected_receipt_sha256"] == sha(run / "receipt.json")
        result = dict(status="complete", independent_replay_complete=fault != "replay", outcome_fits=0)
        write(output / "receipt.json", result)
        write(output / "review.json.gz", {"fixture": True})
        return result

    def raw_post(*args):
        log.append("raw_post")
        if fault == "postcheck":
            raise ValueError("original final raw postcheck")
        return dict(files_sha256={"record_ecg.h5": sha(inputs / "record_ecg.h5")}, hash_errors={})

    def verify_run(run, receipt_sha, guard):
        assert sha(run / "receipt.json") == receipt_sha
        original = json.loads((run / "receipt.json").read_text())
        assert sha(run / "artifact-manifest.json") == original["artifact_manifest_sha256"]
        manifest = json.loads((run / "artifact-manifest.json").read_text())
        for name, pin in manifest["files_sha256"].items():
            assert sha(run / name) == pin
        return original, manifest

    api = SimpleNamespace(
        source_manifest=lambda: dict(files=source),
        cost=cost,
        stage_files=staged,
        checkpoint=checkpoint,
        preflight=preflight,
        extract=extract,
        prepare=prepare,
        support=support,
        execute=execute,
        replay=replay,
        raw_post=raw_post,
        verify_run=verify_run,
        same=lambda a, b: a == b,
    )
    monkeypatch.setattr(m, "load_apis", lambda root: api)
    monkeypatch.setattr(m, "verify_checkpoint", lambda *args: None)
    return SimpleNamespace(
        m=m,
        root=root,
        inputs=inputs,
        declaration=declaration,
        digest=digest,
        value=value,
        template=template,
        log=log,
    )


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def run(h):
    return h.m.run_session(
        h.declaration,
        h.digest,
        root=h.root,
        input_root=h.inputs,
        data_root=h.inputs,
        session_started=time.monotonic() - 10,
        storage_check=lambda: dict(allocated_bytes=0, logical_bytes=0),
    )


def supervisor_for(h, **changes):
    receipt = json.loads((h.root / "results/session-receipt.json").read_text())
    supervisor = h.root / "supervisor.json"
    value = dict(
        status="complete",
        exit_code=0,
        session_started=receipt["session_started"],
        phase="staging_identity",
        engineering_started=None,
    )
    value.update(changes)
    write(supervisor, value)
    return supervisor


def test_session_runs_fixed_order_replays_then_publishes_no_table(tmp_path, monkeypatch):
    h = fixture(tmp_path, monkeypatch)
    result = run(h)
    assert result["status"] == "work_complete_pending_supervisor" and result["independent_replay_complete"]
    assert h.log[0:3] == ["cost", "stage", "checkpoint"]
    assert h.log.index("support") < h.log.index("fit") < h.log.index("replay") < h.log.index("raw_post")
    assert h.log.count("labels") == 2 and h.log.count("close") == 2
    clocks = [v[1] for v in h.log if isinstance(v, tuple)]
    assert len(set(clocks)) == 1
    bundle = h.root / "results/bundle"
    assert (bundle / "run/summary.json.gz").is_file()
    assert not list(bundle.rglob("*.npy"))
    assert not (bundle / "receipt.json").exists()
    supervisor = supervisor_for(h)
    h.m.publish(h.root / "results", supervisor, sha(supervisor), guard=lambda: None)
    assert json.loads((bundle / "receipt.json").read_text())["status"] == "complete"


@pytest.mark.parametrize(
    "fault",
    [
        "cost",
        "inventory",
        "remaining_cost",
        "coverage",
        "coverage_membership",
        "coverage_fraction",
        "support",
        "template",
        "replay",
        "postcheck",
    ],
)
def test_failed_gate_stops_later_stages_and_never_publishes_complete(tmp_path, monkeypatch, fault):
    h = fixture(tmp_path, monkeypatch, fault)
    with pytest.raises((ValueError, RuntimeError, TimeoutError)):
        run(h)
    receipt = json.loads((h.root / "results/session-receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["original_error"]
    if fault in ("cost", "remaining_cost"):
        assert h.log == ["cost"]
    if fault == "inventory":
        assert not h.log
    if fault not in ("support", "replay", "postcheck"):
        assert "labels" not in h.log
    if fault not in ("replay", "postcheck"):
        assert "fit" not in h.log
    assert not (h.root / "results/bundle/receipt.json").exists()


@pytest.mark.parametrize(
    "changes",
    [
        {"session_started": 0},
        {"engineering_started": 0},
        {"exit_code": 1},
        {"phase": "dense_native"},
        {"status": "failed"},
    ],
)
def test_publication_requires_exact_successful_outer_session(tmp_path, monkeypatch, changes):
    h = fixture(tmp_path, monkeypatch)
    run(h)
    supervisor = supervisor_for(h, **changes)
    with pytest.raises(ValueError, match="whole-session"):
        h.m.publish(h.root / "results", supervisor, sha(supervisor), guard=lambda: None)
    receipt = json.loads((h.root / "results/session-receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["phase"] == "publication"
    assert not (h.root / "results/bundle/receipt.json").exists()


@pytest.mark.parametrize("change", ["extra", "symlink"])
def test_publication_rejects_bundle_membership_drift(tmp_path, monkeypatch, change):
    h = fixture(tmp_path, monkeypatch)
    run(h)
    bundle = h.root / "results/bundle"
    if change == "extra":
        (bundle / "unexpected.json").write_text("{}")
    else:
        (bundle / "unexpected.json").symlink_to(bundle / "run/summary.json.gz")
    supervisor = supervisor_for(h)
    with pytest.raises(ValueError, match="membership changed"):
        h.m.publish(h.root / "results", supervisor, sha(supervisor), guard=lambda: None)
    assert not (bundle / "receipt.json").exists()


def test_original_clock_exhaustion_preserved_before_authority(tmp_path, monkeypatch):
    h = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(time, "monotonic", lambda: 100_000.0)  # fresh CI runners have short uptime
    with pytest.raises(TimeoutError):
        h.m.run_session(
            h.declaration,
            h.digest,
            root=h.root,
            input_root=h.inputs,
            data_root=h.inputs,
            session_started=time.monotonic() - 41101,
            storage_check=lambda: dict(allocated_bytes=0, logical_bytes=0),
        )
    assert not h.log
    receipt = json.loads((h.root / "results/session-receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["error_type"] == "TimeoutError"


def test_source_drift_fails_before_staging_or_labels(tmp_path, monkeypatch):
    h = fixture(tmp_path, monkeypatch)
    (h.root / h.m.ENTRY).write_text("changed source")
    with pytest.raises(ValueError, match="changed"):
        run(h)
    assert not h.log


def test_main_uses_one_outer_group_original_clock_and_table_exclusion(tmp_path, monkeypatch):
    h = fixture(tmp_path, monkeypatch)
    real_file = h.m.__file__
    real_load = h.m.load_script
    monkeypatch.setattr(h.m, "__file__", str(h.root / h.m.ENTRY))
    captured = {}

    def supervise(command, **kwargs):
        captured.update(command=command, **kwargs)
        raise RuntimeError("synthetic launch intercepted")

    def load_script(root, name):
        if name == "run_ecgfounder_study":
            return real_load(Path(real_file).parents[1], name)
        if name == "kaggle_cycle":
            return SimpleNamespace(dataset_bytes=lambda *a, **k: 0)
        assert name == "kaggle_ecgfounder_engineering"
        return SimpleNamespace(supervise=supervise)

    monkeypatch.setattr(h.m, "load_script", load_script)
    original = h.m.BOOTSTRAP_STARTED - 10
    argv = [
        "--root",
        str(h.root),
        "--input-root",
        str(h.inputs),
        "--data-root",
        str(h.inputs),
        "--declaration",
        str(h.declaration),
        "--declaration-sha256",
        h.digest,
        "--session-start-monotonic",
        str(original),
    ]
    with pytest.raises(RuntimeError, match="synthetic launch"):
        h.m.main(argv)
    assert captured["session_started"] == original
    assert captured["engineering_started"] is None
    assert captured["phase"] == "staging_identity"
    assert captured["table_root"] == h.root / "results/extraction/tables"
    assert captured["command"][-2:] == ["--worker-parent-pid", str(h.m.os.getpid())]


@pytest.mark.parametrize("name", ["../outside", "/absolute", "data//cache", "data/./cache"])
def test_noncanonical_paths_rejected(tmp_path, name):
    with pytest.raises(ValueError, match="Canonical"):
        load().path(tmp_path, name)


def test_streamed_hash_checks_guard_between_blocks(tmp_path):
    m = load()
    target = tmp_path / "generated.bin"
    target.write_bytes(b"x" * (1024 * 1024 + 1))
    calls = []

    def guard():
        calls.append(True)
        if len(calls) == 2:
            raise TimeoutError("original work deadline")

    with pytest.raises(TimeoutError, match="original work deadline"):
        m.sha(target, guard=guard)
    assert len(calls) == 2


def test_session_guard_keeps_original_calendar_cleanup_deadline(monkeypatch):
    m = load()
    original = time.monotonic() - 10
    deadline = time.monotonic() + 1
    origins = []

    def work_deadline(start):
        origins.append(start)
        return deadline

    monkeypatch.setattr(m, "load_script", lambda *a: SimpleNamespace(work_deadline=work_deadline))
    guard, actual = m.session_guard(original)
    guard()
    monkeypatch.setattr(m.time, "monotonic", lambda: deadline + 1)
    with pytest.raises(TimeoutError, match="calendar work deadline"):
        guard()
    assert actual == deadline and origins == [original]


@needs_addendum
def test_prospective_calendar_addendum_propagates_exact_cutoff():
    from datetime import datetime, timedelta

    from cardiotwin import ecgfounder_runner as runner

    name = Path(__file__).resolve().parents[1] / runner.CYCLE / "calendar-addendum-20261021.json"
    assert sha(name) == "d47849b254697c97e70b37f9b57671cfcc32e9a20ce6ee73733255fd15b646b5"
    cutoff = datetime.fromisoformat(json.loads(name.read_text())["new_work_calendar_deadline_utc"])
    assert runner.DEADLINE == cutoff
    bootstrap = load().load_script(Path(__file__).resolve().parents[1], "kaggle_ecgfounder_session")
    assert bootstrap.CALENDAR == str(name.relative_to(Path(__file__).resolve().parents[1]))
    assert bootstrap.CALENDAR_SHA256 == sha(name)
    engineering = load().load_script(Path(__file__).resolve().parents[1], "kaggle_ecgfounder_engineering")
    assert engineering.DEADLINE == cutoff
    before = runner.ExecutionGuard(
        session_started=1,
        stage_seconds=41100,
        monotonic=lambda: 2,
        now=lambda: cutoff - timedelta(microseconds=1),
    )
    before()
    at_cutoff = runner.ExecutionGuard(
        session_started=1, stage_seconds=41100, monotonic=lambda: 2, now=lambda: cutoff
    )
    with pytest.raises(TimeoutError, match="calendar deadline"):
        at_cutoff()


@needs_addendum
def test_previous_calendar_addendum_remains_immutable():
    root = Path(__file__).resolve().parents[1]
    old = root / "reports/ml-investigation/next-cycle/ecgfounder-cycle-20261013/calendar-addendum-20261020.json"
    assert sha(old) == "817f3d31addf8f17741221b66d29ea88dd29c941cee71d8b04ad48e6da18e376"
    assert json.loads(old.read_text())["new_work_calendar_deadline_utc"] == "2026-10-21T03:30:00+00:00"


def test_mutated_uniform_source_or_extra_dynamic_hash_rejected_before_cost(tmp_path, monkeypatch):
    h = fixture(tmp_path, monkeypatch)
    h.template["files_sha256"]["unapproved.json"] = None
    h.value["files_sha256"]["template.json"] = write(h.root / "template.json", h.template)
    h.digest = write(h.declaration, h.value)
    with pytest.raises(ValueError):
        run(h)
    assert not h.log


def test_failed_initial_helper_import_preserves_original_error_receipt(tmp_path, monkeypatch):
    h = fixture(tmp_path, monkeypatch)

    def fail(*args):
        raise ValueError("original helper import")

    monkeypatch.setattr(h.m, "load_apis", fail)
    with pytest.raises(ValueError, match="original helper import"):
        run(h)
    receipt = json.loads((h.root / "results/session-receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["original_error"] == "original helper import"


def test_final_bundle_mutation_after_worker_verification_cannot_publish(tmp_path, monkeypatch):
    h = fixture(tmp_path, monkeypatch)
    run(h)
    (h.root / "results/bundle/run/summary.json.gz").write_text('{"fixture":false}')
    supervisor = supervisor_for(h)
    with pytest.raises(ValueError, match="changed"):
        h.m.publish(h.root / "results", supervisor, sha(supervisor), guard=lambda: None)
    assert not (h.root / "results/bundle/receipt.json").exists()
