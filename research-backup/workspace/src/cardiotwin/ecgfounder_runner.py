"""Outcome-blind full extraction orchestration; no fitting, submission or retry."""

import hashlib
import importlib.util
import json
import math
import resource
import sys
import time
from collections.abc import Sequence
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .data import RecordInfo, cached_signal_reader, file_hash
from .ecgfounder import (
    CHECKPOINT_BYTES,
    CHECKPOINT_SHA256,
    encode_chunks,
    load_frozen_encoder,
    preprocess_chunk,
)
from .ecgfounder_extraction import IncrementalExtractor
from .ecgfounder_probe import OVERHEAD_COMPONENTS, read_bound_manifest, relocate_record, verify_files
from .ecgfounder_tables import (
    RECEIPT_RESERVE,
    ROW_DTYPE,
    _npy_bytes,
    _snapshot,
    _verify,
    identity_digest,
    write_tables,
)
from .gpu_telemetry import GpuTelemetry
from .investigation import _record_contract, _write
from .precursor import compact_cache_binding, fresh_output, source_manifest
from .progress import Progress

CYCLE = "reports/ml-investigation/next-cycle/ecgfounder-cycle-20261013"
CHECKPOINT_PATH = CYCLE + "/weights/1_lead_ECGFounder.pth"
SPEC_SHA256 = "76f4d1e2bd69f983440465d9e69fb864fe0b347bb69b02c78cf7dc96dc87c7b0"
PROBE_SHA256 = "4a07656820108f8a12694b8e8c8452876956321d26b339f060ecd92855ab5fd2"
ORIGINAL_DECLARATION_SHA256 = "35653e952d542e2615e5bacada4191ad3fa8a8092dd77db6c3a2d452c1b8ebeb"
EXPECTED_ROWS, EXPECTED_PEOPLE, EXPECTED_RECORDS = 629444, 121, 132
ROLE_NAMES = ("fit", "stop", "calibration", "policy", "outer")
DEADLINE = datetime(2026, 10, 22, 12, 30, tzinfo=timezone.utc)
REQUIRED_SOURCE = (
    "src/cardiotwin/ecgfounder.py",
    "src/cardiotwin/ecgfounder_extraction.py",
    "src/cardiotwin/ecgfounder_tables.py",
    "src/cardiotwin/ecgfounder_runner.py",
    "src/cardiotwin/data.py",
    "src/cardiotwin/precursor.py",
    "src/cardiotwin/progress.py",
    "src/cardiotwin/gpu_telemetry.py",
    "scripts/run_ecgfounder_extract.py",
    "reports/ml-investigation/monitor-storage.py",
    "pyproject.toml",
    "uv.lock",
)
WORK = ("dense_cold_io", "scheduler", "per_row_pooling", "table_writing")


def _canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _path(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Required path escapes project root")
    return path


def _bound_json(root, name, bindings):
    if name not in bindings:
        raise ValueError("Required gate/input file is not hash bound")
    return read_bound_manifest(_path(root, name), bindings[name])


def _storage(check, additional=0):
    value = check()
    values = [value[k] for k in ("allocated_bytes", "logical_bytes")]
    if (
        any(not math.isfinite(v) or v < 0 for v in values)
        or max(values) + additional + 250_000_000 > 25_000_000_000
    ):
        raise RuntimeError("Project storage cap/reserve exceeded")
    return value


def storage_measurement(root):
    root = Path(root).resolve()
    path = root / "reports/ml-investigation/monitor-storage.py"
    spec = importlib.util.spec_from_file_location("ecgfounder_runner_storage", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.storage_usage([root / name for name in ("data", "reports", "artifacts")])


def runtime_device_info():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU fallback")
    return dict(
        device="cuda:0", name=torch.cuda.get_device_name(0), reserved=torch.cuda.max_memory_reserved(0)
    )


class ExecutionGuard:
    def __init__(
        self, *, session_started, stage_seconds, now=None, monotonic=time.monotonic, stage_started=None
    ):
        self.session_started, self.stage_seconds = session_started, stage_seconds
        self.stage_started = session_started if stage_started is None else stage_started
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.monotonic = monotonic
        if (
            not math.isfinite(session_started)
            or session_started < 0
            or not math.isfinite(self.stage_started)
            or self.stage_started < session_started
            or not 0 < stage_seconds <= 41100
        ):
            raise ValueError("Finite bootstrap start and bounded extraction stage required")

    def __call__(self):
        current = self.monotonic()
        if current < self.session_started or current - self.session_started > 41100:
            raise TimeoutError("41400-second session requires300-second cleanup reserve")
        if current - self.stage_started > self.stage_seconds or self.now() >= DEADLINE:
            raise TimeoutError("Extraction stage/calendar deadline exceeded")


class MetadataRows(Sequence):
    """Expose only original global row identity, time and acceptance from readonly metadata."""

    def __init__(self, metadata, identities):
        self.metadata, self.identities = metadata, identities

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        value = self.metadata[index]
        code = value["record"]
        accepted = value["accepted"]
        if (
            not np.isfinite(code)
            or code != int(code)
            or not 0 <= code < len(self.identities)
            or accepted not in (0, 1)
        ):
            raise ValueError("Invalid original metadata record/acceptance")
        identity = self.identities[int(code)]
        return dict(
            row_index=index,
            **{k: identity[k] for k in ("record_id", "group_id", "patient_id")},
            timestamp=float(value["timestamp"]),
            accepted=bool(accepted),
        )


def coverage_cell(total, accepted, available):
    if not 0 <= available <= accepted <= total:
        raise ValueError("Invalid common availability counts")
    return dict(
        total_rows=total,
        original_accepted=accepted,
        common_available=available,
        rejected=total - accepted,
        unavailable_accepted=accepted - available,
        fraction=available / accepted if accepted else None,
        passed=accepted > 0 and 20 * available >= 19 * accepted,
    )


def _roles(roles, groups):
    if any(type(r["fold"]) is not int for r in roles) or sorted(r["fold"] for r in roles) != [0, 1, 2]:
        raise ValueError("Original three folds required")
    membership, cells = {}, {}
    for role in roles:
        assigned = set()
        for name in ROLE_NAMES:
            members = role[name]
            if (
                len(members) != len(set(members))
                or assigned.intersection(members)
                or not set(members) <= groups
            ):
                raise ValueError("Original roles must be disjoint and known")
            assigned.update(members)
            cells[role["fold"], name] = [0, 0, 0]
            for group in members:
                membership[role["fold"], group] = name
        if assigned != groups:
            raise ValueError("Original role partition incomplete")
    return membership, cells


def _cost(cost, review, bindings, source, binding, now):
    if (
        cost["version"] != "ecgfounder-complete-cost-v1"
        or cost["cohort_rows"] != EXPECTED_ROWS
        or cost["encoding_upper_bound"] != 1895675
        or "T4" not in cost["device_name"]
        or review["status"] != "passed"
        or review["source_sha256"] != source
        or review["compact_binding_sha256"] != _canonical_sha(binding)
    ):
        raise ValueError("Reviewed cost/cohort/source/device identity differs")

    def evidence(names):
        if not names or any(name not in bindings for name in names):
            raise ValueError("Cost evidence missing or unbound")

    evidence(cost["rate_evidence_files"])
    rates = [
        cost[k]
        for k in (
            "cold_aggregate_chunks_per_second",
            "cold_tenth_chunks_per_second",
            "dispatch_aggregate_chunks_per_second",
            "dispatch_tenth_chunks_per_second",
        )
    ]
    if any(not math.isfinite(v) or v <= 0 for v in rates):
        raise ValueError("Measured positive cost rates required")
    components = cost["components"]
    if set(components) != set(OVERHEAD_COMPONENTS):
        raise ValueError("All11 fixed overhead components required")
    for item in components.values():
        if not math.isfinite(item["upper_seconds"]) or item["upper_seconds"] <= 0 or not item["scope"]:
            raise ValueError("Applicable positive overhead bounds required")
        evidence(item["evidence_files"])
    if set(cost["work_accounting"]) != set(WORK):
        raise ValueError("Dense I/O and every row cost must be charged")
    scope = cost["rate_scope"]
    if scope not in ("dense_causal_full_timeline", "decomposed_chunk_processing_with_sparse_cold_cap"):
        raise ValueError("Unknown measured rate scope")
    for work in cost["work_accounting"].values():
        evidence(work["evidence_files"])
        if not work["reason"]:
            raise ValueError("Cost applicability reasoning required")
        if (
            scope == "dense_causal_full_timeline"
            and work["charged_to"] != "chunk_rate"
            or scope != "dense_causal_full_timeline"
            and work["charged_to"] not in components
        ):
            raise ValueError("Dense work must be charged exactly once to applicable rate or overhead")
    estimate = 1.5 * (1895675 / min(rates) + sum(v["upper_seconds"] for v in components.values()))
    contingency = cost["calendar_contingency_seconds"]
    stage = cost["extraction_stage_seconds"]
    if (
        not math.isfinite(contingency)
        or contingency <= 0
        or not math.isfinite(stage)
        or not 0 < stage <= 41100
        or stage < 1.5 * 1895675 / min(rates)
        or estimate > 41100
        or (DEADLINE - now).total_seconds() < estimate + contingency
    ):
        raise RuntimeError("Complete cost/session/calendar bound failed")
    return dict(
        estimated_complete_seconds=estimate,
        planning_chunks_per_second=min(rates),
        calendar_contingency_seconds=contingency,
        extraction_stage_seconds=stage,
        rate_scope=scope,
    )


def preflight(
    declaration,
    expected_sha256,
    *,
    root,
    data_root,
    storage_check,
    device_info=runtime_device_info,
    now=None,
    guard=None,
    checkpoint_path=CHECKPOINT_PATH,
):
    root, data_root = Path(root).resolve(), Path(data_root).resolve()
    now = now or datetime.now(timezone.utc)
    guard = guard or (lambda: None)
    guard()
    declaration = Path(declaration).resolve()
    if not declaration.is_relative_to(root):
        raise ValueError("Execution declaration must be within bound root")
    payload = read_bound_manifest(declaration, expected_sha256)
    if payload["version"] != "ecgfounder-full-extraction-v1":
        raise ValueError("Fixed extraction declaration required")
    files, source = payload["files_sha256"], payload["source_sha256"]
    if not files or not source:
        raise ValueError("Required source/input bindings missing")
    checkpoint = _path(root, CHECKPOINT_PATH)
    if (
        _path(root, checkpoint_path) != checkpoint
        or files.get(CHECKPOINT_PATH) != CHECKPOINT_SHA256
        or checkpoint.stat().st_size != CHECKPOINT_BYTES
    ):
        raise ValueError("Explicit pinned checkpoint path/bytes/SHA256 binding required")
    verify_files(files, root=root)
    verify_files(source, root=root)
    guard()
    original_name = CYCLE + "/declaration-binding.json"
    fixed = {
        CYCLE + "/spec.md": SPEC_SHA256,
        CYCLE + "/probe-spec.json": PROBE_SHA256,
        original_name: ORIGINAL_DECLARATION_SHA256,
    }
    if any(files.get(k) != v for k, v in fixed.items()):
        raise ValueError("Original frozen scientific pins differ")
    original = _bound_json(root, original_name, files)
    if any(files.get(k) != v for k, v in original["files_sha256"].items()):
        raise ValueError("Inherited inputs differ")
    current = source_manifest()["files"]
    if any(source.get(k) != v for k, v in current.items()) or any(
        name not in source for name in REQUIRED_SOURCE
    ):
        raise ValueError("Current executed sources omitted or changed")
    cache = _path(root, payload["cache"])
    binding = compact_cache_binding(cache, expected=original["compact_binding"])
    if binding != payload["compact_binding"]:
        raise ValueError("Execution compact identity differs")
    for name in ("index.json", "compact.json", "metadata.npy", "features.npy"):
        key = str((cache / name).relative_to(root))
        if files.get(key) != file_hash(cache / name):
            raise ValueError("Required compact input is not explicitly bound")
    cost = _bound_json(root, payload["cost_receipt"], files)
    review = _bound_json(root, payload["cost_review"], files)
    if review.get("cost_sha256") != files[payload["cost_receipt"]]:
        raise ValueError("Cost review binds a different cost file")
    cost_result = _cost(cost, review, files, source, binding, now)
    panel = _bound_json(root, payload["panel_receipt"], files)
    normal = panel["normal_available_by_fold"]
    dispatch = panel["dense_dispatch"]
    if (
        panel["status"] != "passed"
        or panel["numerical_gate"] != "passed"
        or len(normal) != 3
        or any(type(v) is not int or not 11 <= v <= 12 for v in normal)
        or sum(normal) < 33
        or dispatch["status"] != "passed"
        or not 1 <= dispatch["unique_normal_histories"] <= 36
        or dispatch["chunks"] != 60 * dispatch["unique_normal_histories"]
    ):
        raise ValueError("Required numerical/availability/dispatch panel did not pass")
    device = device_info()
    if (
        device["device"] != "cuda:0"
        or "T4" not in device["name"]
        or not math.isfinite(device["reserved"])
        or not 0 <= device["reserved"] <= 12_000_000_000
    ):
        raise RuntimeError("Declared actualT4/CUDA:0 device required")
    index = json.loads((cache / "index.json").read_text())
    compact = json.loads((cache / "compact.json").read_text())
    role_names = [name for name in original["files_sha256"] if name.endswith("/roles.json")]
    if role_names != [payload["roles"]]:
        raise ValueError("Passed roles are not the frozen original file")
    roles = _bound_json(root, payload["roles"], files)
    if roles["cache_index_sha256"] != binding["index_sha256"]:
        raise ValueError("Roles/cache identity mismatch")
    metadata = np.load(cache / "metadata.npy", mmap_mode="r", allow_pickle=False)
    rows = MetadataRows(metadata, compact["records"])
    records = []
    raw_records = {}
    seen = set()
    bijection = {}
    reverse = {}
    identities = {r["record_id"]: r for r in compact["records"]}
    if len(identities) != len(compact["records"]):
        raise ValueError("Duplicate compact records")
    for entry in index["records"]:
        record = entry["record"]
        rid = record["record_id"]
        identity = identities[rid]
        pid = identity["patient_id"]
        group = identity["group_id"]
        if (
            rid in seen
            or record["patient_id"] != pid
            or index["group_ids"][pid] != group
            or record["fs"] != 200
            or bijection.setdefault(group, pid) != pid
            or reverse.setdefault(pid, group) != group
        ):
            raise ValueError("Acquisition/group/patient identity differs")
        seen.add(rid)
        records.append({**identity, **{k: record[k] for k in ("fs", "duration_seconds", "signal_parts")}})
    if (
        seen != set(identities)
        or set(bijection) != set(index["allowed_groups"])
        or len(records) != EXPECTED_RECORDS
        or len(bijection) != EXPECTED_PEOPLE
        or len(rows) != EXPECTED_ROWS
        or index["row_count"] != EXPECTED_ROWS
    ):
        raise ValueError("Original cohort/row counts differ")
    membership, cells = _roles(roles["roles"], set(bijection))
    spans = []
    cursor = 0
    with Progress("ecgfounder.full_preflight_metadata", total=EXPECTED_ROWS, device="cpu") as progress:
        for record in records:
            start = cursor
            parts = record["signal_parts"]
            if not parts or any(type(p["samples"]) is not int or p["samples"] <= 0 for p in parts):
                raise ValueError("Invalid acquisition sample counts")
            total = sum(p["samples"] for p in parts)
            duration = record["duration_seconds"]
            if not math.isfinite(duration) or abs(duration * 200 - total) > 200 * math.ulp(
                duration
            ) + math.ulp(duration * 200):
                raise ValueError("Acquisition duration/sample counts differ")
            for number in range((total + 5999) // 6000 + 1):
                guard()
                row = rows[cursor]
                end = min(number * 6000, total)
                timestamp = row["timestamp"]
                if (
                    any(row[k] != record[k] for k in ("record_id", "group_id", "patient_id"))
                    or metadata[cursor]["duration_seconds"] != duration
                    or not math.isfinite(timestamp)
                    or abs(timestamp * 200 - end) > 200 * math.ulp(duration) + math.ulp(timestamp * 200)
                    or row["accepted"]
                    and end < 120000
                ):
                    raise ValueError("Original cadence/terminal/row identity differs")
                for fold in range(3):
                    cell = cells[fold, membership[fold, record["group_id"]]]
                    cell[0] += 1
                    cell[1] += row["accepted"]
                cursor += 1
                progress.update(cursor)
            spans.append((start, cursor))
    if cursor != len(rows):
        raise ValueError("Original row stream incomplete or extra")
    identity = identity_digest(rows)
    if identity != payload["metadata_identity_sha256"]:
        raise ValueError("Frozen metadata identity digest differs")
    planned = (
        sum(
            (
                _npy_bytes((EXPECTED_ROWS, 1024), np.dtype("<f4")),
                _npy_bytes((EXPECTED_ROWS, 6), np.dtype("<f4")),
                _npy_bytes((EXPECTED_ROWS,), ROW_DTYPE),
            )
        )
        + RECEIPT_RESERVE
    )
    if type(payload["table_byte_cap"]) is not int or payload["table_byte_cap"] < planned:
        raise RuntimeError("Provisioned table bytes insufficient")
    storage = _storage(storage_check, planned)
    raw_manifest = _bound_json(root, payload["raw_manifest"], files)
    required = set()
    for entry in index["records"]:
        record = entry["record"]
        rid = record["record_id"]
        required.update(Path(p["path"]).name for p in record["signal_parts"])
        required.add(f"{rid}_ecg_labels.csv")
    if set(raw_manifest["files_sha256"]) != required:
        raise ValueError("Full raw/annotation identity manifest membership differs")
    verify_files(raw_manifest["files_sha256"], root=data_root)
    guard()
    for entry in index["records"]:
        guard()
        record = RecordInfo(**relocate_record(entry["record"], data_root))
        if _record_contract(record) != entry["input_contract"]:
            raise ValueError("Acquisition source contract differs")
        raw_records[record.record_id] = record
    all_files = {**files, str(declaration.relative_to(root)): expected_sha256}
    table_contract = dict(
        row_count=EXPECTED_ROWS,
        identity_sha256=identity,
        records=[{k: r[k] for k in ("record_id", "group_id", "patient_id")} for r in compact["records"]],
        files_sha256=all_files,
        source_sha256=source,
    )
    verify_files(all_files, root=root)
    verify_files(source, root=root)
    guard()
    receipt = dict(
        status="preflight_passed_not_executed",
        declaration_sha256=expected_sha256,
        row_count=EXPECTED_ROWS,
        records=EXPECTED_RECORDS,
        people=EXPECTED_PEOPLE,
        identity_sha256=identity,
        cost=cost_result,
        device=device,
        planned_table_bytes=planned,
        storage=storage,
        checkpoint=dict(path=CHECKPOINT_PATH, sha256=CHECKPOINT_SHA256, bytes=CHECKPOINT_BYTES),
        fits_allowed=False,
        full_run_eligible=False,
    )
    return dict(
        receipt=receipt,
        root=root,
        data_root=data_root,
        files=all_files,
        source=source,
        payload=payload,
        binding=binding,
        cache=cache,
        rows=rows,
        records=records,
        raw_records=raw_records,
        raw_hashes=dict(raw_manifest["files_sha256"]),
        index=index,
        spans=spans,
        table_contract=table_contract,
        membership=membership,
        coverage_cells=cells,
    )


def _verify_raw(prepared, guard):
    guard()
    snapshot = _snapshot(prepared["raw_hashes"], prepared["data_root"])
    _verify(snapshot, prepared["raw_hashes"])
    for entry in prepared["index"]["records"]:
        guard()
        if _record_contract(prepared["raw_records"][entry["record"]["record_id"]]) != entry["input_contract"]:
            raise ValueError("Acquisition source changed during extraction")
    return snapshot


def extract(
    prepared,
    output,
    *,
    session_started,
    storage_check,
    reader=None,
    model=None,
    encoder=encode_chunks,
    preprocessor=preprocess_chunk,
    resource_check=None,
    now=None,
    cleanup_watchdog=None,
):
    output = fresh_output(output)
    now = now or (lambda: datetime.now(timezone.utc))
    started = time.monotonic()
    guard = ExecutionGuard(
        session_started=session_started,
        stage_started=started,
        stage_seconds=prepared["receipt"]["cost"]["extraction_stage_seconds"],
        now=now,
    )
    result = dict(
        status="running",
        checkpoint=prepared["receipt"]["checkpoint"],
        fits_allowed=False,
        full_run_eligible=False,
        rows_completed=0,
        native_reads_attempted=0,
    )
    core = None
    try:
        guard()
        verify_files(prepared["files"], root=prepared["root"])
        verify_files(prepared["source"], root=prepared["root"])
        _verify_raw(prepared, guard)
        _write(output / "preflight.json", prepared["receipt"])
        result["preflight_sha256"] = file_hash(output / "preflight.json")

        def supervised():
            guard()
            if resource_check is not None:
                measured = resource_check()
            else:
                import torch

                measured = dict(
                    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * (1 if sys.platform == "darwin" else 1024),
                    reserved=torch.cuda.max_memory_reserved(0),
                )
            if any(
                not math.isfinite(measured[key]) or not 0 <= measured[key] <= cap
                for key, cap in (("rss", 10_000_000_000), ("reserved", 12_000_000_000))
            ):
                raise RuntimeError("Declared extraction resource cap exceeded")
            return measured

        supervised()
        if model is None and encoder is encode_chunks:
            model, load = load_frozen_encoder(
                prepared["root"] / CYCLE / "vendor-source/net1d.py",
                prepared["root"] / prepared["receipt"]["checkpoint"]["path"],
                device="cuda:0",
            )
            result["checkpoint_load"] = load
        supervised()

        cells = {key: list(value) for key, value in prepared["coverage_cells"].items()}
        with (
            cached_signal_reader() if reader is None else nullcontext(reader) as native_reader,
            GpuTelemetry() if encoder is encode_chunks else nullcontext(),
        ):
            core = IncrementalExtractor(
                model,
                lambda record, a, b: native_reader(prepared["raw_records"][record["record_id"]], a, b),
                encoder=encoder,
                preprocessor=preprocessor,
                resource_check=supervised,
            )
            with Progress("ecgfounder.full_extraction", total=EXPECTED_ROWS, device="cuda:0") as progress:

                def produced():
                    for record, (a, b) in zip(prepared["records"], prepared["spans"], strict=True):
                        for row in core.iter_record((prepared["rows"][i] for i in range(a, b)), record):
                            for fold in range(3):
                                cells[fold, prepared["membership"][fold, record["group_id"]]][2] += row[
                                    "available"
                                ]
                            result["rows_completed"] += 1
                            if result["rows_completed"] % 4096 == 0:
                                _write(output / "execution-state.json", {**result, "core": core.ledger})
                            progress.update(result["rows_completed"], core=core.ledger)
                            yield row

                table = write_tables(
                    output / "tables",
                    produced(),
                    iter(prepared["rows"]),
                    prepared["table_contract"],
                    root=prepared["root"],
                    byte_cap=prepared["payload"]["table_byte_cap"],
                    storage_check=lambda: _storage(storage_check, 0),
                )
        guard()
        bindings = {**prepared["files"], **prepared["source"]}
        result["hashes_after"] = _snapshot(bindings, prepared["root"])
        _verify(result["hashes_after"], bindings)
        compact_cache_binding(prepared["cache"], expected=prepared["binding"])
        result["raw_hashes_after"] = _verify_raw(prepared, guard)
        if file_hash(output / "preflight.json") != result["preflight_sha256"]:
            raise ValueError("Saved preflight identity changed during extraction")
        supervised()
        coverage = dict(
            cells=[
                dict(fold=fold, role=name, **coverage_cell(*cells[fold, name]))
                for fold in range(3)
                for name in ROLE_NAMES
            ],
            rows=EXPECTED_ROWS,
            identity_sha256=prepared["receipt"]["identity_sha256"],
            table_receipt_sha256=file_hash(output / "tables/receipt.json"),
        )
        coverage["passed"] = all(c["passed"] for c in coverage["cells"])
        coverage["support_audit_authorized"] = False
        _write(output / "coverage.json", coverage)
        result.update(
            status="complete" if coverage["passed"] else "coverage_failed",
            coverage_passed=coverage["passed"],
            tables_complete=table["complete"],
            table_receipt_sha256=coverage["table_receipt_sha256"],
            coverage_sha256=file_hash(output / "coverage.json"),
            core=core.ledger,
            storage_after=_storage(storage_check, 0),
            elapsed_seconds=time.monotonic() - started,
            native_reads_attempted=core.ledger["native_reads_attempted"],
        )
        _write(output / "receipt.json", result)
        return result
    except BaseException as error:
        result.update(
            status="failed",
            error_type=type(error).__name__,
            original_error=str(error),
            elapsed_seconds=time.monotonic() - started,
        )
        if core is not None:
            result["core"] = core.ledger
        failed_at = time.monotonic()
        cleanup_deadline = min(failed_at + 300, session_started + 41400)
        exhausted = failed_at - session_started >= 41100
        result["cleanup"] = dict(
            deadline_monotonic=cleanup_deadline,
            posthash_status="skipped_work_budget_exhausted" if exhausted else "pending",
        )
        for key in ("hashes_after", "raw_hashes_after"):
            result[key] = dict(files_sha256={}, hash_errors={})
        try:
            _write(output / "receipt.json", result)
        except BaseException:
            pass  # An unwritable diagnostic cannot replace the original processing exception.
        if not exhausted:
            try:
                if cleanup_watchdog is not None:
                    cleanup_watchdog(cleanup_deadline)
                stopped = False
                for key, bindings, root in (
                    ("hashes_after", {**prepared["files"], **prepared["source"]}, prepared["root"]),
                    ("raw_hashes_after", prepared["raw_hashes"], prepared["data_root"]),
                ):
                    for name in bindings:
                        if time.monotonic() >= cleanup_deadline:
                            stopped = True
                            break
                        try:
                            result[key]["files_sha256"][name] = file_hash(_path(root, name))
                        except BaseException as failure:
                            result[key]["hash_errors"][name] = f"{type(failure).__name__}: {failure}"
                            if isinstance(failure, TimeoutError):
                                stopped = True
                                break
                    if stopped:
                        break
                result["cleanup"]["posthash_status"] = "partial_cleanup_deadline" if stopped else "complete"
            except BaseException as failure:
                result["cleanup"].update(
                    posthash_status="partial_diagnostic_failure",
                    diagnostic_error=f"{type(failure).__name__}: {failure}",
                )
            try:
                _write(output / "receipt.json", result)
            except BaseException:
                pass
        raise
