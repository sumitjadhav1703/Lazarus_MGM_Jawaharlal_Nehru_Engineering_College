import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cardiotwin import ecgfounder_runner as runner
from cardiotwin.data import RecordInfo
from cardiotwin.ecgfounder_probe import OVERHEAD_COMPONENTS, relocate_record
from cardiotwin.ecgfounder_tables import identity_digest
from cardiotwin.investigation import _record_contract
from cardiotwin.precursor import compact_cache_binding

pytestmark = pytest.mark.usefixtures("historical_ecgfounder_clock")

NOW = datetime(2026, 10, 15, 12, tzinfo=timezone.utc)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture(tmp_path, monkeypatch):
    root = tmp_path
    cycle = root / runner.CYCLE
    cache = root / "data/cache"
    audit = root / "reports/ml-investigation/next-cycle/audit"
    raw = root / "data/raw"
    for path in (cycle, cache, audit, raw):
        path.mkdir(parents=True)
    identities = []
    entries = []
    metadata = []
    expected = []
    raw_hashes = {}
    for i in range(5):
        rid = f"record_{i:03}"
        pid = f"p{i}"
        group = f"g{i}"
        old = Path("/original") / rid
        record = dict(
            record_id=rid,
            patient_id=pid,
            path=str(old),
            fs=200.0,
            duration_seconds=660.0,
            annotated=True,
            clinical={},
            metadata_audit=[],
            source="iridia-af",
            signal_parts=[dict(path=str(old / f"{rid}_ecg_00.h5"), key="ecg", samples=132000)],
        )
        for name in (f"{rid}_ecg_00.h5", f"{rid}_ecg_labels.csv"):
            path = raw / name
            path.write_text("opaque synthetic " + name)
            raw_hashes[name] = digest(path)
        entries.append(
            dict(record=record, input_contract=_record_contract(RecordInfo(**relocate_record(record, raw))))
        )
        identities.append(dict(record_id=rid, patient_id=pid, group_id=group))
        for j in range(23):
            accepted = j >= 21
            metadata.append((i, j * 30.0, 660.0, accepted, 99))
            expected.append(
                dict(row_index=len(expected), **identities[-1], timestamp=j * 30.0, accepted=accepted)
            )
    values = np.array(
        metadata,
        dtype=[
            ("record", "i4"),
            ("timestamp", "f8"),
            ("duration_seconds", "f8"),
            ("accepted", "i1"),
            ("label", "i1"),
        ],
    )
    np.save(cache / "metadata.npy", values)
    np.save(cache / "features.npy", np.zeros((115, 1), np.float32))
    index = dict(
        records=entries,
        row_count=115,
        feature_fields=["timing"],
        rows_sha256="a" * 64,
        group_ids={f"p{i}": f"g{i}" for i in range(5)},
        allowed_groups=[f"g{i}" for i in range(5)],
    )
    (cache / "index.json").write_text(json.dumps(index))
    compact = dict(
        records=identities,
        features=["timing"],
        rows_sha256="a" * 64,
        checksums={name: digest(cache / name) for name in ("features.npy", "metadata.npy")},
    )
    (cache / "compact.json").write_text(json.dumps(compact))
    binding = compact_cache_binding(cache)
    roles = [dict(fold=f, **{name: [f"g{i}"] for i, name in enumerate(runner.ROLE_NAMES)}) for f in range(3)]
    (audit / "roles.json").write_text(
        json.dumps(dict(cache_index_sha256=binding["index_sha256"], roles=roles))
    )
    (audit / "pairs.jsonl").write_text("opaque synthetic pair identities\n")
    (cycle / "spec.md").write_text("fixed synthetic specification")
    (cycle / "probe-spec.json").write_text("{}")
    original_files = {
        str(p.relative_to(root)): digest(p)
        for p in (
            cycle / "spec.md",
            cycle / "probe-spec.json",
            cache / "index.json",
            audit / "roles.json",
            audit / "pairs.jsonl",
        )
    }
    original = cycle / "declaration-binding.json"
    original.write_text(json.dumps(dict(files_sha256=original_files, compact_binding=binding)))
    for key, path in [
        ("SPEC_SHA256", cycle / "spec.md"),
        ("PROBE_SHA256", cycle / "probe-spec.json"),
        ("ORIGINAL_DECLARATION_SHA256", original),
    ]:
        monkeypatch.setattr(runner, key, digest(path))
    monkeypatch.setattr(runner, "EXPECTED_ROWS", 115)
    monkeypatch.setattr(runner, "EXPECTED_PEOPLE", 5)
    monkeypatch.setattr(runner, "EXPECTED_RECORDS", 5)
    source = {}
    for name in runner.REQUIRED_SOURCE:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic source " + name)
        source[name] = digest(path)
    monkeypatch.setattr(runner, "source_manifest", lambda: dict(files=source))
    evidence = cycle / "evidence.json"
    evidence.write_text('{"engineering_evidence":"synthetic"}')
    panel = cycle / "panel.json"
    panel.write_text(
        json.dumps(
            dict(
                status="passed",
                numerical_gate="passed",
                normal_available_by_fold=[12, 12, 12],
                dense_dispatch=dict(status="passed", unique_normal_histories=36, chunks=2160),
            )
        )
    )
    cost = cycle / "cost.json"
    cost_payload = dict(
        version="ecgfounder-complete-cost-v1",
        cohort_rows=115,
        encoding_upper_bound=1895675,
        device_name="Tesla T4",
        cold_aggregate_chunks_per_second=100.0,
        cold_tenth_chunks_per_second=100.0,
        dispatch_aggregate_chunks_per_second=100.0,
        dispatch_tenth_chunks_per_second=100.0,
        rate_scope="dense_causal_full_timeline",
        rate_evidence_files=[str(evidence.relative_to(root))],
        components={
            name: dict(upper_seconds=10.0, evidence_files=[str(evidence.relative_to(root))], scope=name)
            for name in OVERHEAD_COMPONENTS
        },
        work_accounting={
            name: dict(
                charged_to="chunk_rate",
                evidence_files=[str(evidence.relative_to(root))],
                reason="joint dense bound",
            )
            for name in ("dense_cold_io", "scheduler", "per_row_pooling", "table_writing")
        },
        calendar_contingency_seconds=60.0,
        extraction_stage_seconds=30000.0,
    )
    cost.write_text(json.dumps(cost_payload))
    review = cycle / "cost-review.json"
    review.write_text(
        json.dumps(
            dict(
                status="passed",
                cost_sha256=digest(cost),
                source_sha256=source,
                compact_binding_sha256=hashlib.sha256(
                    json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            )
        )
    )
    checkpoint = cycle / "weights/1_lead_ECGFounder.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic pinned checkpoint")
    monkeypatch.setattr(runner, "CHECKPOINT_SHA256", digest(checkpoint), raising=False)
    monkeypatch.setattr(runner, "CHECKPOINT_BYTES", checkpoint.stat().st_size, raising=False)
    raw_manifest = cycle / "raw-manifest.json"
    raw_manifest.write_text(json.dumps(dict(files_sha256=raw_hashes)))
    names = [
        cycle / "spec.md",
        cycle / "probe-spec.json",
        original,
        cache / "index.json",
        cache / "compact.json",
        cache / "metadata.npy",
        cache / "features.npy",
        audit / "roles.json",
        audit / "pairs.jsonl",
        evidence,
        panel,
        cost,
        review,
        raw_manifest,
        checkpoint,
    ]
    declaration = cycle / "full-execution.json"
    payload = dict(
        version="ecgfounder-full-extraction-v1",
        files_sha256={str(p.relative_to(root)): digest(p) for p in names},
        source_sha256=source,
        compact_binding=binding,
        cache=str(cache.relative_to(root)),
        roles=str((audit / "roles.json").relative_to(root)),
        panel_receipt=str(panel.relative_to(root)),
        cost_receipt=str(cost.relative_to(root)),
        cost_review=str(review.relative_to(root)),
        raw_manifest=str(raw_manifest.relative_to(root)),
        metadata_identity_sha256=identity_digest(expected),
        table_byte_cap=1_000_000,
    )
    declaration.write_text(json.dumps(payload))
    return SimpleNamespace(
        root=root,
        cycle=cycle,
        cache=cache,
        audit=audit,
        raw=raw,
        declaration=declaration,
        payload=payload,
        cost=cost,
        review=review,
        storage=lambda: dict(allocated_bytes=0, logical_bytes=0),
        device=lambda: dict(device="cuda:0", name="Tesla T4", reserved=0),
    )


def preflight(f):
    return runner.preflight(
        f.declaration,
        digest(f.declaration),
        root=f.root,
        data_root=f.raw,
        storage_check=f.storage,
        device_info=f.device,
        now=NOW,
    )


def rebind(f, path):
    f.payload["files_sha256"][str(path.relative_to(f.root))] = digest(path)
    f.declaration.write_text(json.dumps(f.payload))


def test_bound_preflight_is_outcome_blind_and_preserves_original_global_order(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    prepared = preflight(f)
    assert prepared["receipt"]["status"] == "preflight_passed_not_executed"
    assert prepared["table_contract"]["row_count"] == 115
    assert prepared["spans"] == [(i * 23, (i + 1) * 23) for i in range(5)]
    assert [row["row_index"] for row in prepared["rows"]] == list(range(115))
    assert len(prepared["coverage_cells"]) == 15
    assert not prepared["receipt"]["fits_allowed"]


@pytest.mark.parametrize(
    "fault",
    [
        "wrong_hash",
        "source",
        "original",
        "role_overlap",
        "row_order",
        "acceptance",
        "terminal",
        "cost_missing",
        "cost_unbounded",
        "cost_scope",
        "cost_rate",
        "cost_review",
        "deadline",
        "device",
        "storage",
        "raw_missing",
    ],
)
def test_preflight_fails_before_model_or_waveform_execution(tmp_path, monkeypatch, fault):
    f = fixture(tmp_path, monkeypatch)
    if fault == "wrong_hash":
        expected = "0" * 64
    else:
        expected = digest(f.declaration)
    if fault == "source":
        (f.root / next(iter(f.payload["source_sha256"]))).write_text("changed")
    elif fault == "original":
        (f.cycle / "declaration-binding.json").write_text("changed")
    elif fault == "role_overlap":
        path = f.audit / "roles.json"
        p = json.loads(path.read_text())
        p["roles"][0]["stop"].append("g0")
        path.write_text(json.dumps(p))
        rebind(f, path)
    elif fault in ("row_order", "acceptance", "terminal"):
        p = np.load(f.cache / "metadata.npy")
        if fault == "row_order":
            p["record"][0] = 1
        elif fault == "acceptance":
            p["accepted"][0] = 2
        else:
            p["timestamp"][-1] -= 0.005
        np.save(f.cache / "metadata.npy", p)
        rebind(f, f.cache / "metadata.npy")
    elif fault.startswith("cost_") and fault != "cost_review":
        p = json.loads(f.cost.read_text())
        if fault == "cost_missing":
            del p["components"]["six_primary_fits"]
        elif fault == "cost_unbounded":
            p["components"]["six_primary_fits"]["evidence_files"] = []
        elif fault == "cost_scope":
            del p["work_accounting"]["dense_cold_io"]
        else:
            p["dispatch_tenth_chunks_per_second"] = 1.0
        f.cost.write_text(json.dumps(p))
        rebind(f, f.cost)
        reviewed = json.loads(f.review.read_text())
        reviewed["cost_sha256"] = digest(f.cost)
        f.review.write_text(json.dumps(reviewed))
        rebind(f, f.review)
    elif fault == "cost_review":
        f.review.write_text("{}")
        rebind(f, f.review)
    elif fault == "deadline":
        monkeypatch.setattr(runner, "DEADLINE", NOW - timedelta(seconds=1))
    elif fault == "device":
        f.device = lambda: dict(device="cpu", name="cpu", reserved=0)
    elif fault == "storage":
        f.storage = lambda: dict(allocated_bytes=24_900_000_000, logical_bytes=0)
    elif fault == "raw_missing":
        next(f.raw.glob("*.h5")).unlink()
    if fault != "wrong_hash":
        expected = digest(f.declaration)
    with pytest.raises((ValueError, RuntimeError, OSError)):
        runner.preflight(
            f.declaration,
            expected,
            root=f.root,
            data_root=f.raw,
            storage_check=f.storage,
            device_info=f.device,
            now=NOW,
        )


def test_small_core_writer_runner_coverage_integration_and_no_auto_fit(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    prepared = preflight(f)
    reads = []

    def reader(record, start, end):
        reads.append((record.record_id, start, end))
        v = np.sin(np.arange(round(start * 200), round(end * 200)) / 20.0)
        return np.column_stack([v, v])

    def prepare(v, *, timings):
        return np.full(5000, v.mean(), np.float32)

    def encode(model, chunks):
        return np.broadcast_to(chunks[:, :1], (len(chunks), 1024)).copy()

    result = runner.extract(
        prepared,
        tmp_path / "run",
        reader=reader,
        model=None,
        encoder=encode,
        preprocessor=prepare,
        storage_check=f.storage,
        resource_check=lambda: dict(rss=1, reserved=0),
        now=lambda: NOW,
        session_started=runner.time.monotonic(),
    )
    assert result["status"] == "complete" and result["coverage_passed"] and not result["fits_allowed"]
    assert result["rows_completed"] == 115 and len(reads) == 105
    coverage = json.loads((tmp_path / "run/coverage.json").read_text())
    assert len(coverage["cells"]) == 15 and all(
        c["original_accepted"] == 2 and c["common_available"] == 2 for c in coverage["cells"]
    )
    assert (tmp_path / "run/tables/receipt.json").is_file()


def test_coverage_uses_original_denominators_equality_and_zero_a():
    assert runner.coverage_cell(20, 20, 19)["passed"]
    assert not runner.coverage_cell(20, 20, 18)["passed"]
    assert not runner.coverage_cell(20, 0, 0)["passed"]


def test_session_guard_reserves_cleanup_and_preserves_stage_caps():
    guard = runner.ExecutionGuard(
        session_started=0.0, stage_seconds=100.0, now=lambda: NOW, monotonic=lambda: 101.0
    )
    with pytest.raises(TimeoutError):
        guard()
    guard = runner.ExecutionGuard(
        session_started=0.0, stage_seconds=41100.0, now=lambda: NOW, monotonic=lambda: 41101.0
    )
    with pytest.raises(TimeoutError):
        guard()


def refresh_metadata_bindings(f, monkeypatch):
    compact = json.loads((f.cache / "compact.json").read_text())
    compact["checksums"]["metadata.npy"] = digest(f.cache / "metadata.npy")
    (f.cache / "compact.json").write_text(json.dumps(compact))
    binding = compact_cache_binding(f.cache)
    original = f.cycle / "declaration-binding.json"
    p = json.loads(original.read_text())
    p["compact_binding"] = binding
    original.write_text(json.dumps(p))
    monkeypatch.setattr(runner, "ORIGINAL_DECLARATION_SHA256", digest(original))
    f.payload["compact_binding"] = binding
    for path in (original, f.cache / "metadata.npy", f.cache / "compact.json"):
        rebind(f, path)
    review = json.loads(f.review.read_text())
    review["compact_binding_sha256"] = runner._canonical_sha(binding)
    f.review.write_text(json.dumps(review))
    rebind(f, f.review)


@pytest.mark.parametrize("fault", ["record_order", "native_coordinate", "accepted_warmup", "bad_flag"])
def test_refrozen_malformed_metadata_reaches_independent_cadence_validation(tmp_path, monkeypatch, fault):
    f = fixture(tmp_path, monkeypatch)
    values = np.load(f.cache / "metadata.npy")
    if fault == "record_order":
        values["record"][0] = 1
    elif fault == "native_coordinate":
        values["timestamp"][-1] -= 0.005
    elif fault == "accepted_warmup":
        values["accepted"][0] = 1
    else:
        values["accepted"][0] = 2
    np.save(f.cache / "metadata.npy", values)
    refresh_metadata_bindings(f, monkeypatch)
    with pytest.raises(ValueError, match="cadence|metadata"):
        preflight(f)


def test_compact_record_codes_may_differ_from_index_order_without_reindexing_rows(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    compact = json.loads((f.cache / "compact.json").read_text())
    compact["records"].reverse()
    (f.cache / "compact.json").write_text(json.dumps(compact))
    values = np.load(f.cache / "metadata.npy")
    values["record"] = 4 - values["record"]
    np.save(f.cache / "metadata.npy", values)
    refresh_metadata_bindings(f, monkeypatch)
    prepared = preflight(f)
    assert [record["record_id"] for record in prepared["records"]] == [f"record_{i:03}" for i in range(5)]
    assert prepared["rows"][0]["record_id"] == "record_000" and prepared["rows"][114]["row_index"] == 114


def test_decomposed_cost_accepts_evidenced_missing_work_and_never_relabels_sparse_rate(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    cost = json.loads(f.cost.read_text())
    cost["rate_scope"] = "decomposed_chunk_processing_with_sparse_cold_cap"
    for item in cost["work_accounting"].values():
        item["charged_to"] = "feature_transfer_persistence"
    f.cost.write_text(json.dumps(cost))
    rebind(f, f.cost)
    review = json.loads(f.review.read_text())
    review["cost_sha256"] = digest(f.cost)
    f.review.write_text(json.dumps(review))
    rebind(f, f.review)
    assert preflight(f)["receipt"]["cost"]["rate_scope"] == "decomposed_chunk_processing_with_sparse_cold_cap"
    cost["work_accounting"]["dense_cold_io"]["charged_to"] = "chunk_rate"
    f.cost.write_text(json.dumps(cost))
    rebind(f, f.cost)
    review["cost_sha256"] = digest(f.cost)
    f.review.write_text(json.dumps(review))
    rebind(f, f.review)
    with pytest.raises(ValueError, match="charged"):
        preflight(f)


def fake_runtime(f, reader):
    return dict(
        reader=reader,
        model=None,
        encoder=lambda model, chunks: np.broadcast_to(chunks[:, :1], (len(chunks), 1024)).copy(),
        preprocessor=lambda v, **k: np.full(5000, v.mean(), np.float32),
        storage_check=f.storage,
        resource_check=lambda: dict(rss=1, reserved=0),
        now=lambda: NOW,
        session_started=runner.time.monotonic(),
    )


def test_complete_runner_persists_actual_postchecks_and_exact_preflight_identity(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    prepared = preflight(f)

    def reader(record, start, end):
        values = np.sin(np.arange(round(start * 200), round(end * 200)) / 20.0)
        return np.column_stack([values, values])

    output = tmp_path / "run"
    result = runner.extract(prepared, output, **fake_runtime(f, reader))
    saved = json.loads((output / "receipt.json").read_text())
    assert saved == result and saved["status"] == "complete"
    assert saved["preflight_sha256"] == digest(output / "preflight.json")
    assert saved["hashes_after"] == dict(
        files_sha256={name: digest(f.root / name) for name in {**prepared["files"], **prepared["source"]}},
        hash_errors={},
    )
    assert saved["raw_hashes_after"] == dict(
        files_sha256={name: digest(f.raw / name) for name in prepared["raw_hashes"]}, hash_errors={}
    )
    declaration = str(f.declaration.relative_to(f.root))
    assert saved["hashes_after"]["files_sha256"][declaration] == digest(f.declaration)


def test_preflight_drift_after_processing_withholds_runner_complete(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    prepared = preflight(f)
    output = tmp_path / "run"
    reads = 0

    def reader(record, start, end):
        nonlocal reads
        reads += 1
        if reads == 105:
            (output / "preflight.json").write_text("{}")
        values = np.sin(np.arange(round(start * 200), round(end * 200)) / 20.0)
        return np.column_stack([values, values])

    with pytest.raises(ValueError, match="preflight"):
        runner.extract(prepared, output, **fake_runtime(f, reader))
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "failed" and not receipt["fits_allowed"]
    assert json.loads((output / "tables/receipt.json").read_text())["complete"]


def test_raw_drift_after_processing_withholds_runner_complete_even_if_tables_complete(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    prepared = preflight(f)
    calls = []

    def reader(record, start, end):
        calls.append((record.record_id, start, end))
        values = np.sin(np.arange(round(start * 200), round(end * 200)) / 20.0)
        if len(calls) == 105:
            next(f.raw.glob("*.h5")).write_text("changed after final selected slice")
        return np.column_stack([values, values])

    with pytest.raises(ValueError, match="binding"):
        runner.extract(prepared, tmp_path / "run", **fake_runtime(f, reader))
    receipt = json.loads((tmp_path / "run/receipt.json").read_text())
    assert receipt["status"] == "failed" and not receipt["fits_allowed"]
    assert json.loads((tmp_path / "run/tables/receipt.json").read_text())["complete"]
    changed = next(f.raw.glob("*.h5"))
    assert receipt["raw_hashes_after"]["files_sha256"][changed.name] == digest(changed)


def test_reader_original_failure_retains_attempt_success_complete_and_hash_errors(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    prepared = preflight(f)

    def reader(*args):
        next(f.raw.glob("*.h5")).unlink()
        raise OSError("original runner read failure")

    with pytest.raises(OSError, match="original runner read failure"):
        runner.extract(prepared, tmp_path / "run", **fake_runtime(f, reader))
    receipt = json.loads((tmp_path / "run/receipt.json").read_text())
    assert receipt["core"]["native_reads_attempted"] == 1 and receipt["core"]["native_reads_succeeded"] == 0
    assert receipt["core"]["native_reads_completed"] == 0 and receipt["core"]["rows_completed"] == 21
    assert receipt["raw_hashes_after"]["hash_errors"]
    assert receipt["original_error"] == "original runner read failure"


def cli_module():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts/run_ecgfounder_extract.py"
    spec = importlib.util.spec_from_file_location("synthetic_extraction_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_help_and_extraction_bootstrap_requirement_without_execution():
    module = cli_module()
    with pytest.raises(SystemExit) as status:
        module.main(["--help"])
    assert status.value.code == 0
    with pytest.raises(SystemExit) as status:
        module.main(
            [
                "--root",
                ".",
                "--declaration",
                "x",
                "--declaration-sha256",
                "0" * 64,
                "--data-root",
                "x",
                "--output",
                "x",
                "--mode",
                "extract",
            ]
        )
    assert status.value.code == 2


def test_cli_preflight_only_never_loads_checkpoint_or_invokes_reader(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    module = cli_module()
    monkeypatch.setattr(module, "preflight", lambda *a, **k: preflight(f))
    monkeypatch.setattr(module, "extract", lambda *a, **k: pytest.fail("extraction called"))
    assert (
        module.main(
            [
                "--root",
                str(f.root),
                "--declaration",
                str(f.declaration),
                "--declaration-sha256",
                digest(f.declaration),
                "--data-root",
                str(f.raw),
                "--output",
                str(tmp_path / "cli"),
                "--mode",
                "preflight",
            ]
        )
        == 0
    )
    assert (
        json.loads((tmp_path / "cli/receipt.json").read_text())["status"] == "preflight_passed_not_executed"
    )


@pytest.mark.parametrize("reserved", [float("nan"), -1.0, float("inf")])
def test_device_memory_measurement_must_be_finite_before_model_loading(tmp_path, monkeypatch, reserved):
    f = fixture(tmp_path, monkeypatch)
    f.device = lambda: dict(device="cuda:0", name="Tesla T4", reserved=reserved)
    with pytest.raises(RuntimeError, match="device|memory|T4"):
        preflight(f)


def test_extraction_stage_ceiling_cannot_substitute_for_completion_bound(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    cost = json.loads(f.cost.read_text())
    cost["extraction_stage_seconds"] = 1.0
    f.cost.write_text(json.dumps(cost))
    rebind(f, f.cost)
    review = json.loads(f.review.read_text())
    review["cost_sha256"] = digest(f.cost)
    f.review.write_text(json.dumps(review))
    rebind(f, f.review)
    with pytest.raises(RuntimeError, match="bound"):
        preflight(f)


def test_raw_drift_between_preflight_and_extract_stops_before_native_reads(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    prepared = preflight(f)
    next(f.raw.glob("*.h5")).write_text("changed between stages")
    with pytest.raises(ValueError, match="binding"):
        runner.extract(
            prepared, tmp_path / "run", **fake_runtime(f, lambda *a: pytest.fail("raw read after drift"))
        )
    receipt = json.loads((tmp_path / "run/receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["native_reads_attempted"] == 0


def test_invalid_stage_start_cannot_reset_whole_session_budget():
    with pytest.raises(ValueError, match="start"):
        runner.ExecutionGuard(session_started=10.0, stage_started=9.0, stage_seconds=100.0, now=lambda: NOW)


def test_runtime_resources_checked_before_model_load(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    prepared = preflight(f)
    monkeypatch.setattr(
        runner, "load_frozen_encoder", lambda *a, **k: pytest.fail("load before resource gate")
    )
    with pytest.raises(RuntimeError, match="resource"):
        runner.extract(
            prepared,
            tmp_path / "run",
            session_started=runner.time.monotonic(),
            storage_check=f.storage,
            resource_check=lambda: dict(rss=10_000_000_001, reserved=0),
            now=lambda: NOW,
        )


def test_coverage_failure_retains_completed_tables_and_original_role_denominators(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    prepared = preflight(f)

    def reader(record, start, end):
        return np.full((round((end - start) * 200), 2), np.nan)

    runtime = fake_runtime(f, reader)
    runtime["preprocessor"] = runner.preprocess_chunk
    result = runner.extract(prepared, tmp_path / "run", **runtime)
    assert result["status"] == "coverage_failed" and result["tables_complete"] and not result["fits_allowed"]
    cells = json.loads((tmp_path / "run/coverage.json").read_text())["cells"]
    assert all(c["original_accepted"] == 2 and c["common_available"] == 0 and not c["passed"] for c in cells)


def test_cli_preserves_original_preflight_failure_and_never_extracts(tmp_path, monkeypatch):
    module = cli_module()

    def failed(*args, **kwargs):
        raise OSError("original preflight failure")

    monkeypatch.setattr(module, "preflight", failed)
    monkeypatch.setattr(module, "extract", lambda *a, **k: pytest.fail("extraction called"))
    output = tmp_path / "failed-cli"
    with pytest.raises(OSError, match="original preflight failure"):
        module.main(
            [
                "--root",
                str(tmp_path),
                "--declaration",
                "x",
                "--declaration-sha256",
                "0" * 64,
                "--data-root",
                str(tmp_path),
                "--output",
                str(output),
                "--mode",
                "preflight",
            ]
        )
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["original_error"] == "original preflight failure"
    assert "original preflight failure" in (output / "original-error.txt").read_text()


def test_cli_extract_uses_first_bootstrap_start_before_preflight(tmp_path, monkeypatch):
    module = cli_module()
    monkeypatch.setattr(runner.time, "monotonic", lambda: 100_000.0)  # fresh CI runners have short uptime
    monkeypatch.setattr(
        module, "preflight", lambda *a, **k: pytest.fail("preflight after whole-session deadline")
    )
    output = tmp_path / "expired-cli"
    with pytest.raises(TimeoutError, match="cleanup"):
        module.main(
            [
                "--root",
                str(tmp_path),
                "--declaration",
                "x",
                "--declaration-sha256",
                "0" * 64,
                "--data-root",
                str(tmp_path),
                "--output",
                str(output),
                "--mode",
                "extract",
                "--session-start-monotonic",
                str(runner.time.monotonic() - 41101),
            ]
        )
    assert json.loads((output / "receipt.json").read_text())["error_type"] == "TimeoutError"


@pytest.mark.parametrize("fault", ["missing", "wrong_pin", "wrong_size"])
def test_checkpoint_must_be_an_explicit_pinned_input_before_accepting_preflight(tmp_path, monkeypatch, fault):
    f = fixture(tmp_path, monkeypatch)
    checkpoint = f.cycle / "weights/1_lead_ECGFounder.pth"
    monkeypatch.setattr(runner, "CHECKPOINT_SHA256", digest(checkpoint), raising=False)
    monkeypatch.setattr(runner, "CHECKPOINT_BYTES", checkpoint.stat().st_size, raising=False)
    if fault == "missing":
        del f.payload["files_sha256"][str(checkpoint.relative_to(f.root))]
        f.declaration.write_text(json.dumps(f.payload))
    if fault == "wrong_pin":
        monkeypatch.setattr(runner, "CHECKPOINT_SHA256", "0" * 64)
    elif fault == "wrong_size":
        monkeypatch.setattr(runner, "CHECKPOINT_BYTES", checkpoint.stat().st_size + 1)
    with pytest.raises(ValueError, match="checkpoint"):
        preflight(f)


def test_work_timeout_saves_original_failure_and_skips_full_posthash_pass(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    prepared = preflight(f)
    clock = [41101.0]
    hash_calls = []
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    original_guard = runner.ExecutionGuard
    monkeypatch.setattr(runner, "ExecutionGuard", lambda **k: original_guard(**k, monotonic=lambda: clock[0]))

    def expensive_hash(path):
        hash_calls.append(str(path))
        clock[0] += 100.0
        return digest(path)

    monkeypatch.setattr(runner, "file_hash", expensive_hash)
    with pytest.raises(TimeoutError, match="cleanup"):
        runner.extract(
            prepared, tmp_path / "timed-out", session_started=0.0, storage_check=f.storage, now=lambda: NOW
        )
    receipt = json.loads((tmp_path / "timed-out/receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["error_type"] == "TimeoutError"
    assert not hash_calls and clock[0] == 41101.0
    assert receipt["cleanup"]["posthash_status"] == "skipped_work_budget_exhausted"


def test_failure_receipt_precedes_bounded_early_error_posthashes(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    prepared = preflight(f)
    output = tmp_path / "early-error"
    clock = [100.0]
    calls = []
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    original_guard = runner.ExecutionGuard
    monkeypatch.setattr(runner, "ExecutionGuard", lambda **k: original_guard(**k, monotonic=lambda: clock[0]))

    def failed_verify(*a, **k):
        raise OSError("original provenance failure")

    monkeypatch.setattr(runner, "verify_files", failed_verify)

    def expensive_hash(path):
        receipt = json.loads((output / "receipt.json").read_text())
        assert receipt["original_error"] == "original provenance failure"
        calls.append(str(path))
        clock[0] += 100.0
        return digest(path)

    monkeypatch.setattr(runner, "file_hash", expensive_hash)
    with pytest.raises(OSError, match="original provenance failure"):
        runner.extract(prepared, output, session_started=0.0, storage_check=f.storage, now=lambda: NOW)
    assert len(calls) == 3 and clock[0] == 400.0
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["cleanup"]["deadline_monotonic"] == 400.0
    assert receipt["cleanup"]["posthash_status"] == "partial_cleanup_deadline"


@pytest.mark.parametrize("reverse_codes", [False, True])
def test_completed_runner_tables_open_in_reviewed_consumer(tmp_path, monkeypatch, reverse_codes):
    from cardiotwin import ecgfounder_compare as consumer

    f = fixture(tmp_path, monkeypatch)
    if reverse_codes:
        compact = json.loads((f.cache / "compact.json").read_text())
        compact["records"].reverse()
        (f.cache / "compact.json").write_text(json.dumps(compact))
        values = np.load(f.cache / "metadata.npy")
        values["record"] = 4 - values["record"]
        np.save(f.cache / "metadata.npy", values)
        refresh_metadata_bindings(f, monkeypatch)
    prepared = preflight(f)

    def reader(record, start, end):
        v = np.sin(np.arange(round(start * 200), round(end * 200)) / 20.0)
        return np.column_stack([v, v])

    result = runner.extract(prepared, tmp_path / "run", **fake_runtime(f, reader))
    monkeypatch.setattr(consumer, "LINEAR_SCHEMA", ("timing",))
    monkeypatch.setattr(consumer, "CHECKPOINT_SHA256", runner.CHECKPOINT_SHA256)
    monkeypatch.setattr(consumer, "CHECKPOINT_BYTES", runner.CHECKPOINT_BYTES)
    with consumer.VerifiedTables.open(
        tmp_path / "run/tables",
        iter(prepared["rows"]),
        prepared["table_contract"],
        root=f.root,
        receipt_sha256=result["table_receipt_sha256"],
        compact_path=str((f.cache / "compact.json").relative_to(f.root)),
        checkpoint_path=runner.CHECKPOINT_PATH,
        feature_schema=("timing",),
    ) as tables:
        assert tables.delta.shape == (115, 1024)
    assert prepared["receipt"]["checkpoint"]["sha256"] == runner.CHECKPOINT_SHA256
    assert result["checkpoint"] == prepared["receipt"]["checkpoint"]


def test_cli_checkpoint_override_must_equal_fixed_bound_path(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    module = cli_module()

    def actual_preflight(*a, **k):
        return runner.preflight(
            f.declaration,
            digest(f.declaration),
            root=f.root,
            data_root=f.raw,
            storage_check=f.storage,
            device_info=f.device,
            now=NOW,
            checkpoint_path=k["checkpoint_path"],
        )

    monkeypatch.setattr(module, "preflight", actual_preflight)
    with pytest.raises(ValueError, match="checkpoint"):
        module.main(
            [
                "--root",
                str(f.root),
                "--declaration",
                str(f.declaration),
                "--declaration-sha256",
                digest(f.declaration),
                "--data-root",
                str(f.raw),
                "--output",
                str(tmp_path / "cli-checkpoint"),
                "--mode",
                "preflight",
                "--checkpoint",
                "alternate.pth",
            ]
        )


def test_cli_work_alarm_arms_nonrenewable_cleanup_alarm(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    prepared = preflight(f)
    module = cli_module()
    clock = [runner.time.monotonic()]
    active = [module.signal.SIG_DFL]
    timers = []
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])

    def set_handler(signum, handler):
        old = active[0]
        active[0] = handler
        return old

    monkeypatch.setattr(module.signal, "signal", set_handler)
    monkeypatch.setattr(module.signal, "setitimer", lambda which, value: timers.append(value))
    monkeypatch.setattr(module, "preflight", lambda *a, **k: prepared)

    def alarm_extract(*a, **k):
        clock[0] += 41100.0
        active[0](module.signal.SIGALRM, None)

    monkeypatch.setattr(module, "extract", alarm_extract)
    with pytest.raises(TimeoutError, match="cleanup reserved"):
        module.main(
            [
                "--root",
                str(f.root),
                "--declaration",
                str(f.declaration),
                "--declaration-sha256",
                digest(f.declaration),
                "--data-root",
                str(f.raw),
                "--output",
                str(tmp_path / "cli-alarm"),
                "--mode",
                "extract",
                "--session-start-monotonic",
                str(clock[0]),
            ]
        )
    assert timers == [41100.0, 300.0, 300.0, 0]
    assert active[0] == module.signal.SIG_DFL


def test_cli_cleanup_alarm_cannot_be_swallowed_by_nested_best_effort_code(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    prepared = preflight(f)
    module = cli_module()
    clock = [runner.time.monotonic()]
    active = [module.signal.SIG_DFL]
    output = tmp_path / "hard-cleanup"
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])

    def set_handler(signum, handler):
        old = active[0]
        active[0] = handler
        return old

    monkeypatch.setattr(module.signal, "signal", set_handler)
    monkeypatch.setattr(module.signal, "setitimer", lambda *a: None)
    monkeypatch.setattr(module, "preflight", lambda *a, **k: prepared)

    class HardExit(BaseException):
        pass

    def hard_exit(code):
        assert code == 124
        raise HardExit()

    import os

    monkeypatch.setattr(os, "_exit", hard_exit)

    def nested_writer_cleanup(*a, **k):
        clock[0] += 41100.0
        with pytest.raises(TimeoutError, match="cleanup reserved"):
            active[0](module.signal.SIGALRM, None)
        receipt = json.loads((output / "receipt.json").read_text())
        assert receipt["error_type"] == "TimeoutError" and receipt["status"] == "failed"
        clock[0] += 300.0
        active[0](module.signal.SIGALRM, None)
        pytest.fail("cleanup watchdog returned")

    monkeypatch.setattr(module, "extract", nested_writer_cleanup)
    with pytest.raises(HardExit):
        module.main(
            [
                "--root",
                str(f.root),
                "--declaration",
                str(f.declaration),
                "--declaration-sha256",
                digest(f.declaration),
                "--data-root",
                str(f.raw),
                "--output",
                str(output),
                "--mode",
                "extract",
                "--session-start-monotonic",
                str(clock[0]),
            ]
        )


def test_cleanup_timeout_stops_auxiliary_hashes_and_preserves_primary_error(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    prepared = preflight(f)
    calls = []

    def failed_verify(*a, **k):
        raise OSError("primary failure before cleanup")

    monkeypatch.setattr(runner, "verify_files", failed_verify)

    def cleanup_timeout(path):
        calls.append(str(path))
        raise TimeoutError("cleanup deadline")

    monkeypatch.setattr(runner, "file_hash", cleanup_timeout)
    with pytest.raises(OSError, match="primary failure before cleanup"):
        runner.extract(prepared, tmp_path / "cleanup-timeout", **fake_runtime(f, lambda *a: None))
    assert len(calls) == 1
    receipt = json.loads((tmp_path / "cleanup-timeout/receipt.json").read_text())
    assert receipt["original_error"] == "primary failure before cleanup"
    assert receipt["cleanup"]["posthash_status"] == "partial_cleanup_deadline"


def test_historical_clock_is_module_local_and_default_calendar_expiry_still_fails(monkeypatch):
    import datetime as standard_datetime

    assert standard_datetime.datetime is datetime
    assert runner.datetime is not datetime
    assert runner.datetime.now(timezone.utc) == NOW
    guard = runner.ExecutionGuard(session_started=runner.time.monotonic(), stage_seconds=100)
    guard()
    monkeypatch.setattr(runner, "datetime", SimpleNamespace(now=lambda tz: runner.DEADLINE))
    with pytest.raises(TimeoutError, match="calendar deadline"):
        guard()
