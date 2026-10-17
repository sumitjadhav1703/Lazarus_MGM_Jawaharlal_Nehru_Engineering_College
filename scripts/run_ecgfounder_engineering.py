#!/usr/bin/env python3
"""Four fixed intended-worker engineering phases; no fitting or submission authority."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
if os.environ["OMP_NUM_THREADS"] != "1":
    raise ValueError("OMP_NUM_THREADS=1 required before imports")

import argparse
import hashlib
import importlib.util
import json
import math
import time
import traceback
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CYCLE = "reports/ml-investigation/next-cycle/ecgfounder-cycle-20261013"
ATTEMPT = (
    CYCLE + "/repair-panel-20261014-b/intended-worker-engineering-20261016/repair-engineering-20261017-b"
)
CACHE = "data/iridia-development-cache-recovered"
SELECTION_SHA = "d32a60ba44ccafd419912e46d47630bf809fd9ce8f6ca2472ecfb77e62bab76f"
BENCHMARK_SHA = "217e05ff1bfe963cff21cf8b0a3584cf48180c46b252d9771a2e84c9754882bf"
EXPECTED_ROWS, TABLE_BYTES = 629444, 2619117060
HASH_ACCOUNTING = {}
COMPONENTS = (
    "source_data_hashing",
    "bootstrap_checkpoint_transfer_loading",
    "feature_transfer_persistence",
    "six_primary_fits",
    "null_nuisance_fits",
    "six_calibrators",
    "policy_selection",
    "complete_timelines",
    "person_bootstrap",
    "artifact_verification_download",
    "review",
)


def load_script(name):
    path = ROOT / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    path = Path(path)
    started = time.monotonic()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        size = stream.tell()
    item = HASH_ACCOUNTING.setdefault(str(path), dict(calls=0, bytes=0, seconds=0.0))
    item["calls"] += 1
    item["bytes"] += size
    item["seconds"] += time.monotonic() - started
    return digest


def bound_path(root, name):
    root, name = Path(root).resolve(), str(name)
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or str(relative) != name or name in ("", "."):
        raise ValueError("Noncanonical bound path")
    path = root / relative
    if any(parent.is_symlink() for parent in [path, *path.parents] if parent.is_relative_to(root)):
        raise ValueError("Bound path contains a symlink")
    if not path.resolve().is_relative_to(root):
        raise ValueError("Bound path escapes root")
    return path


def stage_files(source, root, mapping, pins, *, guard, progress=None):
    if set(mapping) != set(pins):
        raise ValueError("Exact staging source/pin membership required")
    checks = []
    for name, original in mapping.items():
        origin, target = bound_path(source, original), bound_path(root, name)
        if target.exists():
            raise FileExistsError("Fresh ordinary cache files required")
        if not origin.is_file():
            raise ValueError("Ordinary staging source missing")
        checks.append((name, origin, target))
    result = dict(
        files={},
        copy_bytes=0,
        checksum_passes=0,
        checksum_bytes=0,
        cache_state="uncontrolled; no cold-cache claim",
    )
    for name, origin, target in checks:
        if progress:
            progress(dict(result, attempted_file=name))
        guard()
        target.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        copied = 0
        with origin.open("rb") as src, target.open("xb") as dst:
            while block := src.read(1024 * 1024):
                guard()
                dst.write(block)
                copied += len(block)
            dst.flush()
            os.fsync(dst.fileno())
        copied_seconds = time.monotonic() - started
        started = time.monotonic()
        checksum = sha(target)
        if checksum != pins[name]:
            raise ValueError("Staged cache checksum differs")
        result["files"][name] = dict(
            bytes=copied,
            sha256=checksum,
            copy_seconds=copied_seconds,
            checksum_seconds=time.monotonic() - started,
        )
        result["copy_bytes"] += copied
        result["checksum_bytes"] += copied
        result["checksum_passes"] += 1
        if progress:
            progress(dict(result, completed_file=name))
    return result


def aggregate_signal(parts):
    return (
        parts[0]
        if len(parts) == 1
        else hashlib.sha256(b"".join(bytes.fromhex(value) for value in parts)).hexdigest()
    )


def raw_manifest(data_root, records, known, *, guard, progress=None):
    names = []
    for record in records:
        names.extend(record["parts"])
        names.append(record["annotation"])
    if len(names) != len(set(names)) or not set(known) <= set(names):
        raise ValueError("Raw membership duplicates/unknown pin")
    result = dict(
        files_sha256={},
        files_bytes={},
        digest_passes=0,
        digest_bytes=0,
        digest_seconds=0.0,
        new_individual_digests=[],
        annotation_bytes_parsed=0,
        aggregate_identity_verified=False,
        cache_state="uncontrolled; sequential pass may warm filesystem",
    )
    for name in names:
        if progress:
            progress(dict(result, attempted_file=name))
        guard()
        path = bound_path(data_root, name)
        started = time.monotonic()
        digest = sha(path)
        result["digest_seconds"] += time.monotonic() - started
        result["files_sha256"][name] = digest
        result["files_bytes"][name] = path.stat().st_size
        result["digest_passes"] += 1
        result["digest_bytes"] += path.stat().st_size
        if progress:
            progress(dict(result, completed_file=name))
        if name in known and known[name] != digest:
            raise ValueError("Known raw pin differs")
    for record in records:
        actual = aggregate_signal([result["files_sha256"][name] for name in record["parts"]])
        if actual != record["signal_sha256"]:
            raise ValueError(
                "Original signal aggregate differs; new digests cannot establish source identity"
            )
        if result["files_sha256"][record["annotation"]] != record["annotation_sha256"]:
            raise ValueError("Original annotation identity differs")
        result["new_individual_digests"].extend(name for name in record["parts"] if name not in known)
    result["aggregate_identity_verified"] = True
    if progress:
        progress(result)
    return result


def dense_metadata_preflight(rows, requests, records, *, expected_slots=36):
    selected = [dict(r) for r in requests if r.get("slot") == "normal"]
    if len(selected) != expected_slots or any(r.get("status") != "selected" for r in selected):
        raise ValueError("Exact original normal slots required")
    if expected_slots == 36 and [sum(r["fold"] == fold for r in selected) for fold in range(3)] != [
        12,
        12,
        12,
    ]:
        raise ValueError("Original fold slot counts differ")
    wanted = {}
    for slot, request in enumerate(selected):
        rid, t = request["record_id"], request["timestamp"]
        record = records[rid]
        if (
            not math.isfinite(t)
            or t - 300 < 600
            or t * 200 != request["decision_sample"]
            or t * 200 % 6000
            or t > record["duration_seconds"]
            or any(request[k] != record[k] for k in ("record_id", "group_id", "patient_id"))
        ):
            raise ValueError("Exact regular dense window unavailable; no replacement")
        for offset in range(-300, 1, 30):
            wanted.setdefault((rid, t + offset), []).append(slot)
    found = {}
    for row in rows:
        key = row["record_id"], row["timestamp"]
        if key in wanted:
            if key in found:
                raise ValueError("Duplicate exact original decision")
            if type(row["row_index"]) is not int or not isinstance(row["accepted"], (bool, np.bool_)):
                raise ValueError("Invalid original row/flag")
            found[key] = dict(row)
    if set(found) != set(wanted):
        raise ValueError("Missing exact original dense decision; no replacement")
    result = []
    for request in selected:
        window = [found[request["record_id"], request["timestamp"] + offset] for offset in range(-300, 1, 30)]
        if any(any(row[k] != request[k] for k in ("record_id", "group_id", "patient_id")) for row in window):
            raise ValueError("Dense original person identity differs")
        if any(b["row_index"] != a["row_index"] + 1 for a, b in zip(window, window[1:])):
            raise ValueError("Dense original cadence indices differ")
        result.append(dict(request=request, rows=window))
    return result


def dense_slot(
    rows, record, model, reader, *, encoder=None, preprocessor=None, resource_check=None, progress=None
):
    from cardiotwin.ecgfounder import chunk_bounds, encode_chunks, pool_delta, preprocess_chunk
    from cardiotwin.ecgfounder_extraction import IncrementalExtractor

    encoder, preprocessor = encoder or encode_chunks, preprocessor or preprocess_chunk
    # Private frozen helpers preserve exact chunk semantics; this is not full iter_record scheduler timing.
    core = IncrementalExtractor(
        model, reader, encoder=encoder, preprocessor=preprocessor, resource_check=resource_check
    )
    _, masks = core._geometry(record)
    results, fills, batches = [], [], []
    native_seconds = [0.0]

    def traced_reader(record, start, end):
        if progress:
            progress(dict(core.ledger, current_native_interval=[start, end]))
        started = time.monotonic()
        try:
            return reader(record, start, end)
        finally:
            native_seconds[0] += time.monotonic() - started

    core.reader = traced_reader

    def traced_encoder(model, chunks):
        values = encoder(model, chunks)
        batches.extend(min(8, len(chunks) - start) for start in range(0, len(chunks), 8))
        return values

    core.encoder = traced_encoder
    for row in rows:
        core._resources()
        core.ledger["rows_attempted"] += 1
        core.ledger["current_row"] = dict(row_index=row["row_index"], record_id=record["record_id"])
        if progress:
            progress(dict(core.ledger))
        end = round(row["timestamp"] * 200)
        timing = dict(read_seconds=0.0, preprocess_seconds=0.0, infer_seconds=0.0, pool_seconds=0.0)
        result = dict(
            row,
            decision_sample=end,
            available=False,
            delta=None,
            nuisance=None,
            valid_chunks=None,
            rejections={"waveform_not_evaluated_original_rejection": 1},
            timings=timing,
        )
        new = 0
        if row["accepted"]:
            core.ledger["original_accepted"] += 1
            bounds = chunk_bounds(row["timestamp"], record["duration_seconds"])
            core.cache = {k: v for k, v in core.cache.items() if k in bounds}
            new, _ = core._fill(record, bounds, end, masks, timing)
            reasons = {}
            for key in bounds:
                reason = core.cache[key][1]
                if reason is not None:
                    reasons[reason] = reasons.get(reason, 0) + 1
            result.update(valid_chunks=60 - sum(reasons.values()), rejections=reasons)
            if not reasons:
                started = time.monotonic()
                result["delta"] = pool_delta(np.asarray([core.cache[k][0] for k in bounds], dtype=np.float32))
                scalars = np.asarray([core.cache[k][2] for k in bounds])
                result["nuisance"] = np.r_[
                    scalars.mean(0), scalars[30:].mean(0) - scalars[:30].mean(0)
                ].astype(np.float32)
                result["available"] = True
                core.ledger["available_rows"] += 1
                timing["pool_seconds"] = time.monotonic() - started
                core.ledger["pool_seconds"] += timing["pool_seconds"]
        if not row["accepted"]:
            core.ledger["original_rejected"] += 1
        core.ledger["rows_completed"] += 1
        fills.append(dict(row_index=row["row_index"], new_chunks=new, accepted=bool(row["accepted"])))
        results.append(result)
        if progress:
            progress(dict(core.ledger))
    core._resources()
    return results, dict(
        core.ledger,
        rows_processed=len(results),
        fills=fills,
        actual_encoder_batch_sizes=batches,
        actual_native_reader_seconds=native_seconds[0],
        measurement="frozen chunk helper/native I/O; excludes full iter_record scheduler",
    )


def synthetic_table(output, *, count=EXPECTED_ROWS, storage_check, guard):
    if count == EXPECTED_ROWS and not Path(output).resolve().is_relative_to(Path("/kaggle/working")):
        raise ValueError("Full-size synthetic table is remote-only")
    from cardiotwin import ecgfounder_tables as tables

    benchmark = load_script("benchmark_ecgfounder_cost")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = [dict(record_id="synthetic0", group_id="synthetic0", patient_id="synthetic0")]

    def metadata():
        for i in range(count):
            if i % 4096 == 0:
                guard()
            yield dict(row_index=i, **records[0], timestamp=float(600 + i * 30), accepted=True)

    delta, nuisance = np.linspace(-1, 1, 1024, dtype=np.float32), np.arange(6, dtype=np.float32)

    def producer():
        for row in metadata():
            yield dict(
                row,
                available=True,
                decision_sample=round(row["timestamp"] * 200),
                valid_chunks=60,
                rejections={},
                delta=delta,
                nuisance=nuisance,
            )

    (output / "checkpoint.marker").write_bytes(benchmark.MARKER)
    (output / "source.marker").write_text("Fixed whole-table synthetic I/O; never a real checkpoint")
    (output / "compact.json").write_text(
        json.dumps(dict(features=list(benchmark.compare.LINEAR_SCHEMA), records=records))
    )
    contract = dict(
        row_count=count,
        identity_sha256=tables.identity_digest(metadata()),
        records=records,
        files_sha256={n: sha(output / n) for n in ("checkpoint.marker", "compact.json")},
        source_sha256={"source.marker": sha(output / "source.marker")},
    )
    planned = sum(
        (
            tables._npy_bytes((count, 1024), "<f4"),
            tables._npy_bytes((count, 6), "<f4"),
            tables._npy_bytes((count,), tables.ROW_DTYPE),
        )
    )
    if count == EXPECTED_ROWS and planned != TABLE_BYTES:
        raise ValueError("Fixed whole-table allocation differs")
    timed = benchmark.TimedProducer(producer())
    with benchmark.hash_accounting() as writing:
        started = time.monotonic()
        receipt = tables.write_tables(
            output / "tables",
            timed,
            metadata(),
            contract,
            root=output,
            byte_cap=planned + 250000,
            storage_check=storage_check,
        )
        writer_seconds = time.monotonic() - started
    args = dict(
        output=output / "tables",
        metadata_rows=metadata,
        contract=contract,
        root=output,
        receipt_sha256=sha(output / "tables/receipt.json"),
        compact_path="compact.json",
        checkpoint_path="checkpoint.marker",
        feature_schema=list(benchmark.compare.LINEAR_SCHEMA),
    )
    with benchmark.hash_accounting() as reading:
        started = time.monotonic()
        with benchmark.open_synthetic(args):
            pass
        reader_seconds = time.monotonic() - started
    guard()
    return dict(
        rows=count,
        allocations=1,
        reader_opens=1,
        table_bytes=receipt["table_bytes"],
        writer_table_hash_passes=writing["table_hash_passes"] // 3,
        reader_table_hash_passes=reading["table_hash_passes"] // 3,
        writer=writing,
        reader=reading,
        writer_seconds=writer_seconds,
        producer_seconds=timed.seconds,
        writer_residual_seconds=max(
            0.0,
            writer_seconds
            - timed.seconds
            - writing["input_hash_seconds"]
            - writing["table_hash_seconds"]
            - writing["fsync_seconds"]
            - writing["rename_seconds"],
        ),
        reader_seconds=reader_seconds,
        validation_scan_residual_seconds=max(
            0.0, reader_seconds - reading["input_hash_seconds"] - reading["table_hash_seconds"]
        ),
        fixture_generation_is_encoder_throughput=False,
    )


def release_receipts(root, release):
    pins = {}
    for key in ("source_review_receipt", "provider_preflight_receipt"):
        value = release.get(key)
        if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
            raise ValueError("Actual detached review/provider receipt identity required")
        path = bound_path(root, value["path"])
        if path.suffix != ".json" or not path.is_file() or sha(path) != value["sha256"]:
            raise ValueError("Detached review/provider receipt bytes differ")
        pins[value["path"]] = value["sha256"]
    return pins


def validate_release(declaration, release, *, root):
    if (
        declaration.get("version") != "ecgfounder-intended-worker-engineering-v1"
        or release.get("kind") != "one_bounded_private_engineering_attempt"
        or release.get("source_review") != "passed"
        or release.get("declaration_sha256") != declaration.get("_sha256")
        or release.get("source_sha256") != declaration.get("source_sha256")
        or release.get("provider_private") is not True
        or release.get("provider_quota_checked") is not True
    ):
        raise ValueError("Concrete reviewed external execution release required")
    release_receipts(root, release)
    sources = declaration["source_sha256"]
    required = {
        "src/cardiotwin/" + name + ".py"
        for name in (
            "__init__",
            "data",
            "labels",
            "ecgfounder",
            "ecgfounder_extraction",
            "ecgfounder_compare",
            "ecgfounder_tables",
            "ecgfounder_runner",
            "ecgfounder_probe",
            "gpu_telemetry",
            "precursor",
            "progress",
            "modeling",
            "train",
            "investigation",
            "evaluation",
            "alerts",
            "config",
            "features",
        )
    }
    required.update(
        "scripts/" + name + ".py"
        for name in (
            "run_ecgfounder_engineering",
            "kaggle_ecgfounder_engineering",
            "benchmark_ecgfounder_cost",
            "kaggle_cycle",
            "kaggle_ecgfounder_probe",
        )
    )
    required.update(
        (
            "pyproject.toml",
            "uv.lock",
            "configs/precursor-cycle.json",
            "reports/ml-investigation/monitor-storage.py",
            ATTEMPT + "/phase-contract.json",
            CYCLE + "/repair-panel-20261014-b/full-cost-20261015/requirements.json",
            CYCLE + "/vendor-source/net1d.py",
            CYCLE + "/vendor-source/LICENSE",
        )
    )
    if (
        sources.get(CYCLE + "/vendor-source/net1d.py")
        != "832bf90e05e20e62db6f9c7007acf7f0b1f739938ab80fa9b2c1ad438e8da7f7"
        or sources.get(CYCLE + "/vendor-source/LICENSE")
        != "b872b97f0675d67a350880e22e21ffd9fb76dfcb8989f89d950bd3f49eca959e"
    ):
        raise ValueError("Frozen vendor source/license pins differ")
    physical = {
        str(path.relative_to(root))
        for base in ("src/cardiotwin", "scripts")
        for path in (Path(root) / base).glob("*.py")
    }
    if not physical <= set(sources):
        raise ValueError("Outer source map omits a physically staged imported source")
    for key in ("data_root", "cache_source_root"):
        if not Path(declaration[key]).resolve().is_relative_to(Path("/kaggle/input")):
            raise ValueError("Only bound mounted private input roots permitted")
    if not required <= set(sources) or sources["scripts/benchmark_ecgfounder_cost.py"] != BENCHMARK_SHA:
        raise ValueError("Frozen benchmark/entrypoint source identity differs")
    for name, pin in sources.items():
        if sha(bound_path(root, name)) != pin:
            raise ValueError("Execution source differs")
    phase_name = ATTEMPT + "/phase-contract.json"
    phase_contract = json.loads(bound_path(root, phase_name).read_text())
    fixed = {
        CYCLE + "/spec.md": "76f4d1e2bd69f983440465d9e69fb864fe0b347bb69b02c78cf7dc96dc87c7b0",
        CYCLE + "/probe-spec.json": "4a07656820108f8a12694b8e8c8452876956321d26b339f060ecd92855ab5fd2",
        CYCLE
        + "/declaration-binding.json": "35653e952d542e2615e5bacada4191ad3fa8a8092dd77db6c3a2d452c1b8ebeb",
    }
    if (
        release.get("phase_contract_sha256") != sources[phase_name]
        or release.get("phases") != phase_contract["phase_order"]
        or release.get("limits") != phase_contract["limits"]
        or release.get("fixed_scientific_pins") != fixed
    ):
        raise ValueError("Release must bind exact scientific/phase/stage/limit contracts")
    for name, pin in fixed.items():
        if (
            declaration.get("scientific_files_sha256", {}).get(name) != pin
            or sha(bound_path(root, name)) != pin
        ):
            raise ValueError("Original scientific declaration identity differs")


def staging_identity(declaration, *, guard, progress=None):
    from cardiotwin import precursor
    from cardiotwin.ecgfounder_probe import relocate_record
    from cardiotwin.precursor import compact_cache_binding

    if (
        set(declaration["cache_sha256"])
        != {CACHE + "/" + n for n in ("index.json", "compact.json", "features.npy", "metadata.npy")}
        or len(declaration["known_raw_sha256"]) != 291
    ):
        raise ValueError("Exact cache/known raw membership required before staging")
    copied = stage_files(
        declaration["cache_source_root"],
        ROOT,
        declaration["cache_source_files"],
        declaration["cache_sha256"],
        guard=guard,
        progress=(lambda value: progress("cache_staging", value)) if progress else None,
    )
    if (
        set(declaration["cache_sha256"])
        != {CACHE + "/" + n for n in ("index.json", "compact.json", "features.npy", "metadata.npy")}
        or copied["copy_bytes"] != 581150286
    ):
        raise ValueError("Exact four-cache membership/bytes differ")
    started = time.monotonic()
    with patch.object(precursor, "file_hash", sha):
        binding = compact_cache_binding(ROOT / CACHE, expected=declaration["compact_binding"])
    index = json.loads((ROOT / CACHE / "index.json").read_text())
    compact = json.loads((ROOT / CACHE / "compact.json").read_text())
    if (
        index["row_count"] != EXPECTED_ROWS
        or len(index["records"]) != 132
        or len(compact["records"]) != 132
        or len({r["group_id"] for r in compact["records"]}) != 121
    ):
        raise ValueError("Original cohort/cache contract differs")
    raw_records = []
    for entry in index["records"]:
        record = relocate_record(entry["record"], declaration["data_root"])
        rid = record["record_id"]
        parts = [
            str(Path(p["path"]).relative_to(Path(declaration["data_root"]).resolve()))
            for p in record["signal_parts"]
        ]
        annotation = str(
            (Path(record["path"]) / (rid + "_ecg_labels.csv")).relative_to(
                Path(declaration["data_root"]).resolve()
            )
        )
        raw_records.append(
            dict(
                record_id=rid,
                parts=parts,
                annotation=annotation,
                signal_sha256=entry["input_contract"]["signal_sha256"],
                annotation_sha256=entry["input_contract"]["annotation_sha256"],
            )
        )
    compact_seconds = time.monotonic() - started
    raw = raw_manifest(
        declaration["data_root"],
        raw_records,
        declaration["known_raw_sha256"],
        guard=guard,
        progress=(lambda value: progress("raw_checksums", value)) if progress else None,
    )
    if (
        raw["digest_passes"] != 359
        or raw["digest_bytes"] != 8577562285
        or sum(len(r["parts"]) for r in raw_records) != 227
    ):
        raise ValueError("Exact development raw membership/bytes differ")
    return dict(
        cache=copied,
        compact_binding=binding,
        compact_validation_seconds=compact_seconds,
        compact_hash_accounting="See exact checksum calls/bytes in phase receipt",
        raw=raw,
        annotation_bytes_parsed=0,
    )


def real_dense(declaration, *, guard, progress=None):
    import torch

    from cardiotwin import ecgfounder, ecgfounder_compare
    from cardiotwin.data import RecordInfo, cached_signal_reader
    from cardiotwin.ecgfounder import load_frozen_encoder
    from cardiotwin.ecgfounder_probe import HistoryEncoder, relocate_record
    from cardiotwin.ecgfounder_runner import MetadataRows
    from cardiotwin.gpu_telemetry import GpuTelemetry
    from cardiotwin.progress import Progress

    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Actual T4 CUDA:0 required; no CPU fallback")
    if (ecgfounder_compare.CHECKPOINT_BYTES, ecgfounder_compare.CHECKPOINT_SHA256) != (
        ecgfounder.CHECKPOINT_BYTES,
        ecgfounder.CHECKPOINT_SHA256,
    ):
        raise ValueError("Synthetic checkpoint overrides were not restored")
    for name, pin in declaration["cache_sha256"].items():
        if sha(bound_path(ROOT, name)) != pin:
            raise ValueError("Staged cache changed before dense preflight")
    selected_path = bound_path(ROOT, declaration["selection"])
    if sha(selected_path) != SELECTION_SHA:
        raise ValueError("Original panel-b selection changed")
    selected = json.loads(selected_path.read_text())
    index = json.loads((ROOT / CACHE / "index.json").read_text())
    compact = json.loads((ROOT / CACHE / "compact.json").read_text())
    identities = {r["record_id"]: r for r in compact["records"]}
    records = {
        e["record"]["record_id"]: dict(e["record"], **identities[e["record"]["record_id"]])
        for e in index["records"]
    }
    metadata = np.load(ROOT / CACHE / "metadata.npy", mmap_mode="r", allow_pickle=False)
    try:
        windows = dense_metadata_preflight(
            MetadataRows(metadata, compact["records"]), selected["requests"], records
        )
    finally:
        metadata._mmap.close()
    guard()
    started = time.monotonic()
    model, loaded = load_frozen_encoder(
        bound_path(ROOT, CYCLE + "/vendor-source/net1d.py"),
        bound_path(ROOT, CYCLE + "/weights/1_lead_ECGFounder.pth"),
        device="cuda:0",
    )
    benchmark = load_script("benchmark_ecgfounder_cost")

    def resources():
        guard()
        measured = dict(rss=benchmark.peak_rss_bytes(), reserved=torch.cuda.max_memory_reserved(0))
        if measured["rss"] > 10000000000 or measured["reserved"] > 12000000000:
            raise RuntimeError("Actual RSS/CUDA reserved ceiling exceeded")
        return measured

    summary = dict(
        checkpoint_load=loaded,
        checkpoint_load_seconds=time.monotonic() - started,
        device="cuda:0",
        device_name=torch.cuda.get_device_name(0),
        slots=[],
        processed_rows=0,
        new_chunks=0,
        parity=[],
        parity_native_reads_attempted=0,
        parity_native_reads_completed=0,
        source_cache_state="Full raw checksum pass precedes these reads; OS cache uncontrolled; fresh per-slot HDF5 caches only",
    )
    publish = progress
    with (
        GpuTelemetry(interval_seconds=10),
        Progress("ecgfounder.engineering_dense", total=396, device="cuda:0") as progress,
    ):
        for slot, window in enumerate(windows):
            guard()
            if publish:
                publish("dense_slot_start", dict(summary, attempted_slot=slot))
            record = records[window["request"]["record_id"]]
            native_record = RecordInfo(
                **{
                    k: v
                    for k, v in relocate_record(
                        {k: v for k, v in record.items() if k != "group_id"}, declaration["data_root"]
                    ).items()
                }
            )
            started = time.monotonic()
            with cached_signal_reader() as read:
                results, ledger = dense_slot(
                    window["rows"],
                    record,
                    model,
                    lambda r, a, b: read(native_record, a, b),
                    resource_check=resources,
                    progress=(
                        lambda value: publish(
                            "dense_current_slot",
                            dict(slot=slot, previous_completed_rows=summary["processed_rows"], ledger=value),
                        )
                    )
                    if publish
                    else None,
                )
            summary["processed_rows"] += len(results)
            summary["new_chunks"] += ledger["new_chunks"]
            if summary["processed_rows"] > 396 or summary["new_chunks"] > 3240:
                raise RuntimeError("Dense fixed scope cap exceeded")
            summary["slots"].append(
                dict(
                    slot=slot,
                    request=window["request"],
                    ledger=ledger,
                    helper_wall_seconds=time.monotonic() - started,
                    available_rows=sum(r["available"] for r in results),
                    result_hashes=[
                        dict(
                            row_index=r["row_index"],
                            accepted=bool(r["accepted"]),
                            available=r["available"],
                            valid_chunks=r["valid_chunks"],
                            rejections=r["rejections"],
                            delta_sha256=hashlib.sha256(r["delta"].tobytes()).hexdigest()
                            if r["available"]
                            else None,
                        )
                        for r in results
                    ],
                )
            )
            # Additional last600s native/parity work is outside the dense timing/counts.
            resources()
            started = time.monotonic()
            t = window["request"]["timestamp"]
            summary["parity_native_reads_attempted"] += 1
            if publish:
                publish("parity_native_attempt", summary)
            with cached_signal_reader() as read:
                native = read(native_record, t - 600, t)[:, 0]
            summary["parity_native_reads_completed"] += 1
            direct = HistoryEncoder(model).encode(
                native,
                timestamp=t,
                record_id=record["record_id"],
                first_sample=round((t - 600) * 200),
                duration=record["duration_seconds"],
            )
            resources()
            last = results[-1]
            if direct["available"] != last["available"] or direct["rejections"] != last["rejections"]:
                raise ValueError("Direct/incremental rejection parity differs")
            if last["available"]:
                np.testing.assert_allclose(direct["delta"], last["delta"], rtol=1e-4, atol=1e-5)
                np.testing.assert_allclose(direct["nuisance"], last["nuisance"], rtol=1e-4, atol=1e-5)
            parity_errors = (
                {name: float(np.max(np.abs(direct[name] - last[name]))) for name in ("delta", "nuisance")}
                if last["available"]
                else {}
            )
            summary["parity"].append(
                dict(
                    slot=slot,
                    native_samples=len(native),
                    native_read_calls=1,
                    considered_chunks=60,
                    encoded_chunks=direct["encoded_chunks"],
                    seconds=time.monotonic() - started,
                    passed=True,
                    max_absolute_errors=parity_errors,
                )
            )
            if publish:
                publish("dense_completed_slot", summary)
            progress.update(summary["processed_rows"], force=True)
    summary["final_resources"] = resources()
    for name, pin in declaration["cache_sha256"].items():
        if sha(bound_path(ROOT, name)) != pin:
            raise ValueError("Staged cache changed during dense work")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=["staging_identity", "synthetic_table", "dense_native"], required=True
    )
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--declaration-sha256", required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not ROOT.is_relative_to(Path("/kaggle/working")):
        raise ValueError("Execution is intended-worker-only")
    if sha(args.declaration) != args.declaration_sha256 or sha(args.release) != args.release_sha256:
        raise ValueError("Declaration/release digest differs")
    declaration = json.loads(args.declaration.read_text())
    declaration["_sha256"] = args.declaration_sha256
    validate_release(declaration, json.loads(args.release.read_text()), root=ROOT)
    output = bound_path(ROOT, str(args.output.relative_to(ROOT)))
    output.mkdir(parents=True, exist_ok=False)
    benchmark = load_script("benchmark_ecgfounder_cost")
    cycle = load_script("kaggle_cycle")

    def global_storage():
        roots = [Path("/kaggle/input"), Path("/kaggle/working")]
        return dict(
            allocated_bytes=cycle.dataset_bytes(roots, allocated=True),
            logical_bytes=cycle.dataset_bytes(roots),
        )

    receipt = dict(
        status="running",
        phase=args.phase,
        full_run_eligible=False,
        actual_outcome_fits=0,
        accepted_remote_upper_seconds=dict.fromkeys(benchmark.COMPONENTS),
    )
    started = time.monotonic()

    def publish(stage, value):
        receipt["progress"] = dict(stage=stage, counts=value)
        receipt["checksum_calls"] = HASH_ACCOUNTING
        benchmark.atomic_json(output / "receipt.json", receipt)

    def guard():
        if max(global_storage().values()) + 250000000 > 25000000000:
            raise RuntimeError("Global storage/reserve exceeded")

    try:
        guard()
        if args.phase == "staging_identity":
            result = staging_identity(declaration, guard=guard, progress=publish)
        elif args.phase == "synthetic_table":
            result = synthetic_table(output / "fixture", storage_check=global_storage, guard=guard)
        else:
            result = real_dense(declaration, guard=guard, progress=publish)
        validate_release(declaration, json.loads(args.release.read_text()), root=ROOT)
        receipt.update(status="complete_engineering_measurement_only", result=result)
    except BaseException as error:
        receipt.update(
            status="failed",
            original_error=load_script("kaggle_ecgfounder_engineering").safe_text(error),
            sanitized_traceback=load_script("kaggle_ecgfounder_engineering").safe_text(
                traceback.format_exc()
            ),
            error_type=type(error).__name__,
        )
        raise RuntimeError(load_script("kaggle_ecgfounder_engineering").safe_text(error)) from None
    finally:
        receipt["checksum_calls"] = HASH_ACCOUNTING
        receipt["elapsed_seconds"] = time.monotonic() - started
        benchmark.atomic_json(output / "receipt.json", receipt)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(load_script("kaggle_ecgfounder_engineering").safe_text(error), flush=True)
        raise SystemExit(1) from None
