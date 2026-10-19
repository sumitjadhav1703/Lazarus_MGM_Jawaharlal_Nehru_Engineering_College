"""Tiny generated engineering tests only; never run a real or full-size phase."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from conftest import HistoricalECGFounderDatetime

ROOT = Path(__file__).parents[1]


def module(name="run_ecgfounder_engineering"):
    path = ROOT / "scripts" / (name + ".py")
    assert path.exists(), "Source-only engineering driver absent"
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    if hasattr(result, "datetime"):
        result.datetime = HistoricalECGFounderDatetime
    return result


def fixture_rows():
    rows = [
        dict(
            row_index=i,
            record_id="record_000",
            group_id="g0",
            patient_id="p0",
            timestamp=float(i * 30),
            accepted=i >= 20,
        )
        for i in range(61)
    ]
    request = dict(rows[40], slot="normal", status="selected", fold=0, role="fit", decision_sample=240000)
    record = dict(
        record_id="record_000",
        group_id="g0",
        patient_id="p0",
        fs=200,
        duration_seconds=1800.0,
        signal_parts=[dict(samples=360000)],
    )
    return rows, request, record


def test_dense_metadata_preflight_preserves_exact_identity_and_flags():
    m = module()
    rows, request, record = fixture_rows()
    rows[35]["accepted"] = False
    selected = m.dense_metadata_preflight(rows, [request], {record["record_id"]: record}, expected_slots=1)
    assert [r["row_index"] for r in selected[0]["rows"]] == list(range(30, 41))
    assert selected[0]["rows"][5]["accepted"] is False
    assert selected[0]["request"] == request


@pytest.mark.parametrize("change", ["missing", "duplicate", "time", "identity", "early", "terminal"])
def test_dense_preflight_fails_closed_without_replacements(change):
    m = module()
    rows, request, record = fixture_rows()
    if change == "missing":
        rows.pop(35)
    elif change == "duplicate":
        rows.append(dict(rows[35]))
    elif change == "time":
        request["timestamp"] = 1200.5
    elif change == "identity":
        request["patient_id"] = "different"
    elif change == "early":
        request.update(timestamp=600.0, decision_sample=120000)
    else:
        record["duration_seconds"] = 1199.5
    with pytest.raises(ValueError):
        m.dense_metadata_preflight(rows, [request], {record["record_id"]: record}, expected_slots=1)


def test_dense_adapter_matches_complete_core_and_counts_rejected_gap():
    m = module()
    from cardiotwin.ecgfounder_extraction import IncrementalExtractor

    rows, request, record = fixture_rows()
    rows[35]["accepted"] = False
    calls = []

    def reader(record, start, end):
        calls.append((start, end))
        x = np.arange(round(start * 200), round(end * 200), dtype=np.float32)
        x[x < 6000] = np.nan
        return np.column_stack((x, x))

    def prep(values, **kwargs):
        from cardiotwin.ecgfounder import InvalidChunk

        if not np.isfinite(values).all():
            raise InvalidChunk("calibration_or_missing")
        return np.full(5000, values[0] / 100000.0, dtype=np.float32)

    def encoder(model, chunks):
        return np.repeat(chunks[:, :1], 1024, axis=1).astype(np.float32)

    full = list(
        IncrementalExtractor(None, reader, encoder=encoder, preprocessor=prep).iter_record(rows, record)
    )
    calls.clear()
    got, ledger = m.dense_slot(rows[30:41], record, None, reader, encoder=encoder, preprocessor=prep)
    assert len(got) == 11 and ledger["rows_processed"] == 11
    assert ledger["new_chunks"] == 90 and ledger["fills"][6]["new_chunks"] == 6
    assert ledger["fills"][5]["new_chunks"] == 0
    assert ledger["pool_seconds"] == sum(r["timings"]["pool_seconds"] for r in got)
    assert ledger["pool_seconds"] > 0
    assert all(300 <= start < end <= 1200 for start, end in calls)
    for a, b in zip(got, full[30:41], strict=True):
        assert a["row_index"] == b["row_index"] and a["available"] == b["available"]
        assert a["rejections"] == b["rejections"]
        if a["available"]:
            np.testing.assert_array_equal(a["delta"], b["delta"])
            np.testing.assert_array_equal(a["nuisance"], b["nuisance"])


def test_streamed_raw_manifest_requires_known_and_aggregate_identity(tmp_path):
    m = module()
    names = ["record_000_ecg_00.h5", "record_000_ecg_01.h5", "record_000_ecg_labels.csv"]
    for name in names:
        (tmp_path / name).write_bytes(name.encode())
    pins = {name: m.sha(tmp_path / name) for name in names}
    aggregate = m.aggregate_signal([pins[n] for n in names[:2]])
    records = [
        dict(
            record_id="record_000",
            parts=names[:2],
            signal_sha256=aggregate,
            annotation=names[2],
            annotation_sha256=pins[names[2]],
        )
    ]
    got = m.raw_manifest(tmp_path, records, {names[0]: pins[names[0]]}, guard=lambda: None)
    assert got["files_sha256"] == pins and got["digest_passes"] == 3
    assert got["new_individual_digests"] == [names[1]]
    records[0]["signal_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="aggregate"):
        m.raw_manifest(tmp_path, records, {}, guard=lambda: None)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../b", "a//b"])
def test_bound_paths_refuse_escape(tmp_path, name):
    with pytest.raises(ValueError):
        module().bound_path(tmp_path, name)


def test_ordinary_cache_copy_refuses_symlink_and_existing(tmp_path):
    m = module()
    source = tmp_path / "inputs"
    source.mkdir()
    (source / "x").write_bytes(b"generated")
    root = tmp_path / "project"
    root.mkdir()
    pins = {"cache/x": m.sha(source / "x")}
    m.stage_files(source, root, {"cache/x": "x"}, pins, guard=lambda: None)
    assert (root / "cache/x").read_bytes() == b"generated" and not (root / "cache/x").is_symlink()
    with pytest.raises(FileExistsError):
        m.stage_files(source, root, {"cache/x": "x"}, pins, guard=lambda: None)
    (source / "y").symlink_to(source / "x")
    with pytest.raises(ValueError, match="symlink"):
        m.stage_files(source, root, {"cache/y": "y"}, {"cache/y": pins["cache/x"]}, guard=lambda: None)


def test_tiny_real_writer_reader_once_restores_checkpoint_pins(tmp_path):
    m = module()
    from cardiotwin import ecgfounder_compare as compare

    pins = compare.CHECKPOINT_BYTES, compare.CHECKPOINT_SHA256
    got = m.synthetic_table(
        tmp_path / "fixture",
        count=7,
        storage_check=lambda: dict(allocated_bytes=0, logical_bytes=0),
        guard=lambda: None,
    )
    assert got["rows"] == 7 and got["writer_table_hash_passes"] == 2 and got["reader_table_hash_passes"] == 2
    assert got["allocations"] == 1 and got["reader_opens"] == 1
    assert (compare.CHECKPOINT_BYTES, compare.CHECKPOINT_SHA256) == pins


def test_ceilings_intersect_without_renewing_bootstrap_or_engineering():
    m = module("kaggle_ecgfounder_engineering")
    assert (
        m.remaining_seconds(
            session_started=0,
            engineering_started=50,
            phase_started=100,
            phase_limit=600,
            now=200,
            calendar_remaining=1000,
        )
        == 500
    )
    assert (
        m.remaining_seconds(
            session_started=0,
            engineering_started=50,
            phase_started=100,
            phase_limit=600,
            now=41101,
            calendar_remaining=1000,
        )
        <= 0
    )
    assert (
        m.remaining_seconds(
            session_started=0,
            engineering_started=0,
            phase_started=3550,
            phase_limit=600,
            now=3590,
            calendar_remaining=1000,
        )
        == 10
    )


def test_process_group_rss_counts_descendants_and_unavailable_fails():
    m = module("kaggle_ecgfounder_engineering")
    assert m.parse_group_rss("1 10 11\n2 10 20\n3 11 90\n", 10) == 31 * 1024
    with pytest.raises(RuntimeError):
        m.parse_group_rss("", 10)


@pytest.mark.parametrize("minute,second,remaining", [(25, 0, 300), (30, 1, -1)])
def test_prospective_calendar_cutoff_still_stops_fresh_session(monkeypatch, minute, second, remaining):
    from datetime import datetime, timezone
    from types import SimpleNamespace

    m = module("kaggle_ecgfounder_engineering")
    monkeypatch.setattr(
        m,
        "datetime",
        SimpleNamespace(now=lambda tz: datetime(2026, 10, 18, 3, minute, second, tzinfo=timezone.utc)),
    )
    assert (
        m.remaining_seconds(session_started=0, engineering_started=0, phase_started=0, phase_limit=600, now=0)
        == remaining
    )


def test_no_execution_release_is_accepted_from_source_only_contract(tmp_path):
    m = module()
    declaration = dict(version="ecgfounder-intended-worker-engineering-v1", external_action_approved=False)
    with pytest.raises(ValueError, match="release"):
        m.validate_release(declaration, {}, root=tmp_path)


def test_package_contains_only_bound_source_and_detached_release_hash(tmp_path):
    m = module("kaggle_ecgfounder_engineering")
    root = tmp_path / "project"
    (root / "scripts").mkdir(parents=True)
    for name in ("run_ecgfounder_engineering", "kaggle_ecgfounder_probe", "kaggle_ecgfounder_engineering"):
        (root / "scripts" / (name + ".py")).write_bytes((ROOT / "scripts" / (name + ".py")).read_bytes())
    driver = m.load_script(root, "run_ecgfounder_engineering")
    declaration = root / "declaration.json"
    names = [
        "scripts/" + n + ".py"
        for n in ("run_ecgfounder_engineering", "kaggle_ecgfounder_probe", "kaggle_ecgfounder_engineering")
    ]
    declaration.write_text(
        __import__("json").dumps(
            dict(
                version="ecgfounder-intended-worker-engineering-v1",
                source_sha256={n: driver.sha(root / n) for n in names},
            )
        )
    )
    release = root / "release.json"
    review = root / "review.json"
    provider = root / "provider.json"
    review.write_text("Generated source review fixture")
    provider.write_text("Generated provider fixture")
    release.write_text(
        __import__("json").dumps(
            {
                "kind": "one_bounded_private_engineering_attempt",
                "source_review_receipt": {"path": "review.json", "sha256": driver.sha(review)},
                "provider_preflight_receipt": {"path": "provider.json", "sha256": driver.sha(provider)},
            }
        )
    )
    release_sha = driver.sha(release)
    output = root / "prepared"
    payload = m.package(
        root,
        declaration,
        driver.sha(declaration),
        output,
        "sumit1703/cardiotwin-ecgfounder-engineering-20261016-generated",
        release,
        release_sha,
    )
    assert payload["release_sha256"] == release_sha
    assert set(payload["manifest"]) == set(names) | {
        "declaration.json",
        "release.json",
        "review.json",
        "provider.json",
    }
    metadata = __import__("json").loads((output / "kernel-metadata.json").read_text())
    assert metadata["is_private"] is True and metadata["machine_shape"] == "NvidiaTeslaT4"
    compile((output / "train.py").read_text(), str(output / "train.py"), "exec")
    changed = __import__("json").loads(declaration.read_text())
    changed["source_sha256"]["review.json"] = "0" * 64
    declaration.write_text(__import__("json").dumps(changed))
    with pytest.raises(ValueError, match="Conflicting"):
        m.package(
            root,
            declaration,
            driver.sha(declaration),
            root / "conflict",
            "sumit1703/cardiotwin-ecgfounder-engineering-20261016-generated",
            release,
            release_sha,
        )
    changed["source_sha256"].pop("review.json")
    declaration.write_text(__import__("json").dumps(changed))
    changed["package_files_sha256"] = {"review.json": "0" * 64}
    changed["scientific_files_sha256"] = {"review.json": driver.sha(review)}
    declaration.write_text(__import__("json").dumps(changed))
    with pytest.raises(ValueError, match="Conflicting"):
        m.package(
            root,
            declaration,
            driver.sha(declaration),
            root / "scientific-conflict",
            "sumit1703/cardiotwin-ecgfounder-engineering-20261016-generated",
            release,
            release_sha,
        )
    changed.pop("package_files_sha256")
    changed.pop("scientific_files_sha256")
    declaration.write_text(__import__("json").dumps(changed))
    with pytest.raises(FileExistsError):
        m.package(
            root,
            declaration,
            driver.sha(declaration),
            output,
            "sumit1703/cardiotwin-ecgfounder-engineering-20261016-generated",
            release,
            release_sha,
        )


def test_supervisor_kills_stalled_child_on_independent_clock_and_saves_error(tmp_path, monkeypatch):
    import types

    m = module("kaggle_ecgfounder_engineering")

    class FakeProcess:
        pid = 123
        returncode = None
        stdout = None

        def poll(self):
            return self.returncode

    process = FakeProcess()
    stopped = []
    cycle = types.SimpleNamespace(
        dataset_bytes=lambda *a, **k: 0,
        _stop_group=lambda p: (stopped.append(p.pid), setattr(p, "returncode", -9)),
    )

    def write(path, value):
        Path(path).write_text(__import__("json").dumps(value))

    helper = types.SimpleNamespace(write_json=write, sanitized=str)
    monkeypatch.setattr(m, "load_script", lambda project, name: cycle if name == "kaggle_cycle" else helper)
    monkeypatch.setattr(m.subprocess, "Popen", lambda *a, **k: process)
    # The launch check succeeds; the separate parent's next check expires while child stalls.
    values = iter([1.0, -1.0])
    monkeypatch.setattr(m, "remaining_seconds", lambda **k: next(values))

    class NoThread:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def join(self, **kwargs):
            pass

        def is_alive(self):
            return False

    monkeypatch.setattr(m.threading, "Thread", NoThread)
    with pytest.raises(TimeoutError):
        m.supervise(
            ["generated-stalled-child"],
            project=tmp_path,
            output=tmp_path / "guard",
            roots=[tmp_path],
            session_started=0.0,
            engineering_started=None,
            phase="dense_native",
        )
    assert stopped == [123]
    saved = __import__("json").loads((tmp_path / "guard/receipt.json").read_text())
    assert saved["status"] == "failed" and "ceiling exhausted" in saved["original_error"]


def test_full_size_table_is_rejected_locally_before_creating_directory(tmp_path):
    m = module()
    with pytest.raises(ValueError, match="remote-only"):
        m.synthetic_table(tmp_path / "forbidden", storage_check=lambda: None, guard=lambda: None)
    assert not (tmp_path / "forbidden").exists()


def test_independent_storage_checks_allocated_diagnostics_separately(tmp_path, monkeypatch):
    import types

    m = module("kaggle_ecgfounder_engineering")
    project = tmp_path / "project"
    project.mkdir()

    def sizes(roots, allocated=False):
        if roots == [project]:
            return 300000000 if allocated else 1200000000
        if len(roots) == 3:
            return 0 if allocated else 1000000000
        return 1300000000

    cycle = types.SimpleNamespace(dataset_bytes=sizes)
    helper = types.SimpleNamespace(
        write_json=lambda p, v: Path(p).write_text(__import__("json").dumps(v)), sanitized=str
    )
    monkeypatch.setattr(m, "load_script", lambda project, name: cycle if name == "kaggle_cycle" else helper)
    monkeypatch.setattr(
        m.subprocess, "Popen", lambda *a, **k: pytest.fail("Storage violation must stop before launch")
    )
    with pytest.raises(RuntimeError, match="Diagnostic/source"):
        m.supervise(
            ["never-start"],
            project=project,
            output=project / "guard",
            roots=[tmp_path / "mounted"],
            session_started=0.0,
            engineering_started=None,
            phase="dense_native",
        )
    saved = __import__("json").loads((project / "guard/receipt.json").read_text())
    assert saved["diagnostic_storage"] == {"apparent": 200000000, "allocated": 300000000}
    assert saved["storage"] == {"apparent": 1300000000, "allocated": 1300000000}


@pytest.mark.parametrize("global_cap", [False, True])
def test_remote_dependency_wheel_temp_uses_dependency_allocation_only(tmp_path, monkeypatch, global_cap):
    import base64
    import hashlib
    import json
    import sys
    import time
    import types

    m = module("kaggle_ecgfounder_engineering")
    working, inputs = tmp_path / "working", tmp_path / "inputs"
    working.mkdir()
    inputs.mkdir()
    files = {
        "pyproject.toml": b'[project]\ndependencies = ["generated-only"]\n',
        "declaration.json": b"{}",
        "release.json": b"{}",
    }
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    payload = dict(
        identity_name="sumit1703/cardiotwin-ecgfounder-engineering-20261017-generated",
        identity=hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        files={name: base64.b64encode(data).decode() for name, data in files.items()},
        manifest=manifest,
        declaration="declaration.json",
        declaration_sha256=manifest["declaration.json"],
        release="release.json",
        release_sha256=manifest["release.json"],
    )
    project = working / payload["identity_name"].split("/")[1]

    def mapped_path(value):
        path = Path(value)
        return (
            working if path == Path("/kaggle/working") else inputs if path == Path("/kaggle/input") else path
        )

    driver = types.SimpleNamespace(
        bound_path=lambda root, name: root / name,
        sha=lambda path: hashlib.sha256(path.read_bytes()).hexdigest(),
        validate_release=lambda *a, **k: None,
        HASH_ACCOUNTING={},
        COMPONENTS=[],
    )
    helper, cycle = module("kaggle_ecgfounder_probe"), module("kaggle_cycle")
    monkeypatch.setattr(m, "Path", mapped_path)
    monkeypatch.setattr(
        m,
        "load_script",
        lambda project, name: {
            "run_ecgfounder_engineering": driver,
            "kaggle_ecgfounder_probe": helper,
            "kaggle_cycle": cycle,
        }[name],
    )
    monkeypatch.setattr(m, "DIAGNOSTIC_BYTES", 65536)
    if global_cap:
        monkeypatch.setattr(m, "PROJECT_BYTES", 180000)
    monkeypatch.setattr(m.sys, "path", list(m.sys.path))
    for key in ("TMPDIR", "PYTHONPATH", "OMP_NUM_THREADS", "PYTHONUNBUFFERED"):
        monkeypatch.setenv(key, "generated-test-original")
    real_supervise = m.supervise
    real_popen, child = m.subprocess.Popen, {}

    def start_child(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        child["pid"] = process.pid
        return process

    monkeypatch.setattr(m.subprocess, "Popen", start_child)
    # Sandbox blocks ps; this test targets real child files and storage, not RSS measurement.
    monkeypatch.setattr(
        m.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout=f"1 {child['pid']} 1\n")
    )

    def generated_dependency(command, **kwargs):
        if kwargs["output"].name != "dependencies-guard":
            assert m.os.environ["TMPDIR"] == str(project / "scratch")
            raise RuntimeError("Generated stop before any engineering phase")
        # Replace only the external pip operation; exercise the real parent storage guard.
        return real_supervise(
            [
                sys.executable,
                "-c",
                "import os,pathlib; "
                "pathlib.Path(os.environ['TMPDIR'],'generated-wheel.whl').write_bytes(b'x'*131072)",
            ],
            **kwargs,
        )

    monkeypatch.setattr(m, "supervise", generated_dependency)
    with pytest.raises(RuntimeError, match="Global storage" if global_cap else "Generated stop"):
        m.remote(payload, session_started=time.monotonic())
    saved = json.loads((project / "engineering-results/bootstrap-receipt.json").read_text())
    if global_cap:
        guard = json.loads((project / "engineering-results/dependencies-guard/receipt.json").read_text())
        assert max(guard["storage"].values()) + 65536 > 180000
        assert guard["status"] == "failed"
    else:
        assert saved["dependency_bootstrap"]["status"] == "complete"
    assert (project / "dependencies/.pip-tmp/generated-wheel.whl").stat().st_size == 131072
    assert not list((project / "scratch").iterdir())


def test_failed_native_attempt_is_in_partial_progress(tmp_path):
    m = module()
    rows, request, record = fixture_rows()
    snapshots = []

    def failed_reader(record, start, end):
        raise OSError("original synthetic read failure")

    with pytest.raises(OSError, match="original synthetic read failure"):
        m.dense_slot(
            rows[30:41],
            record,
            None,
            failed_reader,
            encoder=lambda *a: None,
            preprocessor=lambda *a: None,
            progress=snapshots.append,
        )
    assert snapshots[-1]["native_reads_attempted"] == 1 and snapshots[-1]["rows_attempted"] == 1
    assert snapshots[-1]["native_reads_succeeded"] == 0


def test_signed_url_redaction_covers_split_child_logs_and_bootstrap(monkeypatch, capsys):
    m = module("kaggle_ecgfounder_engineering")
    chunks = [
        b"error https://cdn.example/file?X-Amz-",
        b"Signature=secretvalue&token=private\nnext token=anothersecret\n",
    ]
    safe = "".join(m.safe_lines(chunks))
    assert "secretvalue" not in safe and "private" not in safe and "anothersecret" not in safe
    assert "<URL redacted>" in safe

    def failed_remote(*a, **k):
        raise OSError("https://cdn.example/file?token=checkpointsecret")

    monkeypatch.setattr(m, "remote", failed_remote)
    with pytest.raises(SystemExit) as caught:
        m.bootstrap({}, session_started=0.0)
    assert caught.value.code == 1
    stdout = capsys.readouterr().out
    assert "checkpointsecret" not in stdout and "<URL redacted>" in stdout


def test_supervisor_original_signed_failure_receipt_and_raise_are_safe(tmp_path, monkeypatch):
    import types

    m = module("kaggle_ecgfounder_engineering")
    cycle = types.SimpleNamespace(dataset_bytes=lambda *a, **k: 0)
    helper = types.SimpleNamespace(
        write_json=lambda p, v: Path(p).write_text(__import__("json").dumps(v)), sanitized=m.safe_text
    )
    monkeypatch.setattr(m, "load_script", lambda project, name: cycle if name == "kaggle_cycle" else helper)
    monkeypatch.setattr(m, "remaining_seconds", lambda **k: 100.0)

    def fail(*a, **k):
        raise OSError("https://cdn.example/file?token=privatesignature")

    monkeypatch.setattr(m.subprocess, "Popen", fail)
    with pytest.raises(RuntimeError) as caught:
        m.supervise(
            ["generated"],
            project=tmp_path,
            output=tmp_path / "guard",
            roots=[tmp_path],
            session_started=0.0,
            engineering_started=None,
            phase="dense_native",
        )
    assert "privatesignature" not in str(caught.value)
    saved = (tmp_path / "guard/receipt.json").read_text()
    assert "privatesignature" not in saved and "OSError" in saved


def test_authorization_bearer_and_oversized_child_line_are_redacted():
    m = module("kaggle_ecgfounder_engineering")
    assert "privatebearer" not in m.safe_text("Authorization: Bearer privatebearer")
    safe = "".join(m.safe_lines([b"x" * 65537, b"https://cdn.example/?token=tailsecret\nokay\n"]))
    assert "tailsecret" not in safe and safe.endswith("okay\n")


def test_source_hash_ledger_counts_actual_passes_and_bytes(tmp_path):
    m = module()
    path = tmp_path / "generated.bin"
    path.write_bytes(b"only-generated-input")
    assert m.sha(path) == m.sha(path)
    assert m.HASH_ACCOUNTING[str(path)]["calls"] == 2
    assert m.HASH_ACCOUNTING[str(path)]["bytes"] == 2 * path.stat().st_size


def test_cache_copy_partial_failure_preserves_attempted_file(tmp_path):
    m = module()
    source, target = tmp_path / "input", tmp_path / "project"
    source.mkdir()
    (source / "generated").write_bytes(b"synthetic")
    snapshots = []
    with pytest.raises(ValueError, match="checksum"):
        m.stage_files(
            source,
            target,
            {"cache/generated": "generated"},
            {"cache/generated": "0" * 64},
            guard=lambda: None,
            progress=snapshots.append,
        )
    assert snapshots[-1]["attempted_file"] == "cache/generated"
    assert m.HASH_ACCOUNTING[str(target / "cache/generated")]["calls"] == 1


def test_bound_path_refuses_symlinked_output_parent(tmp_path):
    m = module()
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        m.bound_path(root, "linked/output")


def release_fixture(tmp_path, monkeypatch):
    import json

    m = module()
    modules = (
        "__init__ data labels ecgfounder ecgfounder_extraction ecgfounder_compare "
        "ecgfounder_tables ecgfounder_runner ecgfounder_probe gpu_telemetry precursor "
        "progress modeling train investigation evaluation alerts config features"
    ).split()
    scripts = (
        "run_ecgfounder_engineering kaggle_ecgfounder_engineering benchmark_ecgfounder_cost "
        "kaggle_cycle kaggle_ecgfounder_probe"
    ).split()
    names = ["src/cardiotwin/" + name + ".py" for name in modules]
    names += ["scripts/" + name + ".py" for name in scripts]
    names += [
        "pyproject.toml",
        "uv.lock",
        "configs/precursor-cycle.json",
        "reports/ml-investigation/monitor-storage.py",
        m.ATTEMPT + "/phase-contract.json",
        m.CYCLE + "/repair-panel-20261014-b/full-cost-20261015/requirements.json",
        m.CYCLE + "/vendor-source/net1d.py",
        m.CYCLE + "/vendor-source/LICENSE",
    ]
    special_pins = {
        "scripts/benchmark_ecgfounder_cost.py": m.BENCHMARK_SHA,
        m.CYCLE
        + "/vendor-source/net1d.py": "832bf90e05e20e62db6f9c7007acf7f0b1f739938ab80fa9b2c1ad438e8da7f7",
        m.CYCLE
        + "/vendor-source/LICENSE": "b872b97f0675d67a350880e22e21ffd9fb76dfcb8989f89d950bd3f49eca959e",
        m.CYCLE + "/spec.md": "76f4d1e2bd69f983440465d9e69fb864fe0b347bb69b02c78cf7dc96dc87c7b0",
        m.CYCLE + "/probe-spec.json": "4a07656820108f8a12694b8e8c8452876956321d26b339f060ecd92855ab5fd2",
        m.CYCLE
        + "/declaration-binding.json": "35653e952d542e2615e5bacada4191ad3fa8a8092dd77db6c3a2d452c1b8ebeb",
    }
    real_sha = m.sha
    # Fixed-pin fixtures avoid reading actual experiment/data artifacts.
    monkeypatch.setattr(
        m, "sha", lambda p: special_pins.get(str(Path(p).relative_to(tmp_path)), None) or real_sha(p)
    )
    for name in set(names) | set(special_pins):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("generated source only")
    contract = dict(
        phase_order=["staging_identity", "synthetic_cpu", "synthetic_table", "dense_native"],
        limits=dict(rss_bytes=10000000000),
    )
    (tmp_path / m.ATTEMPT / "phase-contract.json").write_text(json.dumps(contract))
    sources = {name: m.sha(tmp_path / name) for name in names}
    fixed = {
        name: special_pins[name]
        for name in special_pins
        if name.endswith(("spec.md", "probe-spec.json", "declaration-binding.json"))
    }
    declaration = dict(
        version="ecgfounder-intended-worker-engineering-v1",
        _sha256="a" * 64,
        source_sha256=sources,
        scientific_files_sha256=fixed,
        data_root="/kaggle/input/generated-raw",
        cache_source_root="/kaggle/input/generated-cache",
    )
    release = dict(
        kind="one_bounded_private_engineering_attempt",
        source_review="passed",
        declaration_sha256="a" * 64,
        source_sha256=sources.copy(),
        provider_private=True,
        provider_quota_checked=True,
        phase_contract_sha256=sources[m.ATTEMPT + "/phase-contract.json"],
        phases=contract["phase_order"],
        limits=contract["limits"],
        fixed_scientific_pins=fixed,
    )
    for key in ("source_review_receipt", "provider_preflight_receipt"):
        path = tmp_path / (key + ".json")
        path.write_text(json.dumps(dict(status="generated fixture", no_execution_authority=True)))
        release[key] = dict(path=path.name, sha256=m.sha(path))
    return m, declaration, release


def test_detached_release_binds_declaration_sources_scientific_phases_and_limits(tmp_path, monkeypatch):
    import copy

    m, declaration, release = release_fixture(tmp_path, monkeypatch)
    m.validate_release(declaration, release, root=tmp_path)
    for key in (
        "declaration_sha256",
        "source_sha256",
        "fixed_scientific_pins",
        "phases",
        "limits",
        "phase_contract_sha256",
        "kind",
        "source_review_receipt",
        "provider_preflight_receipt",
    ):
        changed = copy.deepcopy(release)
        changed[key] = None
        with pytest.raises(ValueError):
            m.validate_release(declaration, changed, root=tmp_path)


@pytest.mark.parametrize(
    "change", ["outside_input", "omitted_module", "vendor_pin", "license_pin", "changed_source"]
)
def test_release_refuses_root_source_and_vendor_gaps(tmp_path, monkeypatch, change):
    m, declaration, release = release_fixture(tmp_path, monkeypatch)
    if change == "outside_input":
        declaration["data_root"] = "/tmp/undeclared"
    elif change == "omitted_module":
        (tmp_path / "scripts/unbound.py").write_text("generated")
    elif change in ("vendor_pin", "license_pin"):
        name = m.CYCLE + "/vendor-source/" + ("net1d.py" if change == "vendor_pin" else "LICENSE")
        declaration["source_sha256"][name] = "0" * 64
        release["source_sha256"][name] = "0" * 64
    else:
        (tmp_path / "src/cardiotwin/progress.py").write_text("modified source")
    with pytest.raises(ValueError):
        m.validate_release(declaration, release, root=tmp_path)


def test_dense_direct_parity_checks_resources_before_and_after_final_slot(tmp_path, monkeypatch):
    import json
    import types
    from contextlib import contextmanager

    torch = pytest.importorskip("torch")

    from cardiotwin import data, ecgfounder, ecgfounder_probe, gpu_telemetry, progress

    m = module()
    rows, request, record = fixture_rows()
    cache = tmp_path / m.CACHE
    cache.mkdir(parents=True)
    (cache / "index.json").write_text(json.dumps(dict(records=[dict(record=record)])))
    (cache / "compact.json").write_text(
        json.dumps(dict(records=[{k: record[k] for k in ("record_id", "group_id", "patient_id")}]))
    )
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(dict(requests=[request])))
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m, "sha", lambda p: m.SELECTION_SHA)
    monkeypatch.setattr(m, "dense_metadata_preflight", lambda *a: [dict(request=request, rows=rows[30:41])])
    monkeypatch.setattr(
        m.np, "load", lambda *a, **k: types.SimpleNamespace(_mmap=types.SimpleNamespace(close=lambda: None))
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda n: "Tesla T4 generated fixture")
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda n: 0)
    monkeypatch.setattr(data, "RecordInfo", lambda **k: k)
    monkeypatch.setattr(ecgfounder_probe, "relocate_record", lambda r, root: r)
    monkeypatch.setattr(ecgfounder, "load_frozen_encoder", lambda *a, **k: (None, {}))
    monkeypatch.setattr(m, "load_script", lambda name: types.SimpleNamespace(peak_rss_bytes=lambda: 0))
    events = []

    @contextmanager
    def reader():
        yield lambda *a: events.append("parity_read") or np.zeros((120000, 2))

    class FakeProgress:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def update(self, *a, **k):
            pass

    @contextmanager
    def telemetry(**kwargs):
        yield

    monkeypatch.setattr(data, "cached_signal_reader", reader)
    monkeypatch.setattr(progress, "Progress", FakeProgress)
    monkeypatch.setattr(gpu_telemetry, "GpuTelemetry", telemetry)
    result = dict(
        rows[40],
        available=True,
        valid_chunks=60,
        rejections={},
        delta=np.zeros(1024, dtype=np.float32),
        nuisance=np.zeros(6, dtype=np.float32),
    )
    monkeypatch.setattr(m, "dense_slot", lambda *a, **k: ([result], dict(new_chunks=60)))

    class FakeHistoryEncoder:
        def __init__(self, model):
            pass

        def encode(self, *a, **k):
            events.append("parity_encode")
            return dict(result, encoded_chunks=60)

    monkeypatch.setattr(ecgfounder_probe, "HistoryEncoder", FakeHistoryEncoder)
    got = m.real_dense(
        dict(cache_sha256={}, selection="selection.json", data_root="/generated"),
        guard=lambda: events.append("resources_or_guard"),
    )
    start = events.index("parity_read")
    assert events[start - 1] == "resources_or_guard"
    assert events[start + 1 : start + 3] == ["parity_encode", "resources_or_guard"]
    assert got["parity_native_reads_attempted"] == got["parity_native_reads_completed"] == 1
    assert got["parity"][0]["passed"]


@pytest.mark.parametrize("key", ["source_review_receipt", "provider_preflight_receipt"])
def test_release_requires_actual_detached_evidence_bytes(tmp_path, monkeypatch, key):
    m, declaration, release = release_fixture(tmp_path, monkeypatch)
    (tmp_path / release[key]["path"]).write_text("tampered generated receipt")
    with pytest.raises(ValueError, match="receipt"):
        m.validate_release(declaration, release, root=tmp_path)
