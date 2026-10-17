"""Declare or execute the fixed outcome-blind ECGFounder engineering panel; never fit."""

import argparse
import hashlib
import importlib.util
import json
import resource
import sys
import time
import traceback
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from cardiotwin.data import RecordInfo, cached_signal_reader, file_hash
from cardiotwin.ecgfounder import (
    InvalidChunk,
    encode_chunks,
    load_frozen_encoder,
    pool_delta,
    preprocess_chunk,
    preprocess_history,
)
from cardiotwin.ecgfounder_probe import (
    OLD_SELECTION_SHA,
    PANEL_DESIGN_SHA,
    PANEL_IDENTITY,
    PANEL_SPEC_SHA,
    HistoryEncoder,
    acquisition_calibration_mask,
    check_resources,
    cost_gate,
    read_bound_manifest,
    relocate_record,
    select_metadata_supported_requests,
    select_probe_requests,
    verify_files,
)
from cardiotwin.gpu_telemetry import GpuTelemetry
from cardiotwin.investigation import _record_contract, _write
from cardiotwin.precursor import compact_cache_binding, fresh_output, source_manifest
from cardiotwin.progress import Progress

PANEL_TIMING_SHA = "73d24e5f94c97ffd8768be56923a0236bf1d453e911c2c045c89a94730683fb9"


class SelectionRows(Sequence):
    """Expose only acquisition identities, accepted flags and times; no feature access."""

    def __init__(self, cache, identities):
        self.metadata = np.load(Path(cache) / "metadata.npy", mmap_mode="r", allow_pickle=False)
        self.identities = identities

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        row = self.metadata[index]
        record = row["record"]
        if not np.isfinite(record) or record != int(record) or not 0 <= record < len(self.identities):
            raise ValueError("Invalid original acquisition index")
        if row["accepted"] not in (0, 1):
            raise ValueError("Invalid original acceptance flag")
        return dict(self.identities[int(record)], timestamp=float(row["timestamp"]),
                    accepted=bool(row["accepted"]))


def panel_attempt_inputs(args, inputs):
    """Verify the one fixed repair declaration; old source identities remain historical."""
    cycle = Path(args.cycle).resolve().relative_to(Path.cwd().resolve())
    attempt = cycle / "repair-panel-20261014-b/spec.md"
    if Path(args.attempt).resolve() != attempt.resolve():
        raise ValueError("Only the fixed panel-b attempt path is supported")
    paths = {attempt: PANEL_SPEC_SHA, cycle / "repair-design-20261014.md": PANEL_DESIGN_SHA,
             attempt.parent / "dense-dispatch-timing-addendum.md": PANEL_TIMING_SHA,
             cycle / "selection-20261014/selected.json": OLD_SELECTION_SHA}
    for path, expected in paths.items():
        if file_hash(path) != expected:
            raise ValueError(f"Fixed panel declaration/selection changed: {path}")
    declaration_path = attempt.parent / "declaration-binding.json"
    declaration = json.loads(declaration_path.read_text())
    files = declaration["files_sha256"]
    required = {**inputs["files_sha256"], **inputs["source_sha256"],
                **{str(p): digest for p, digest in paths.items()},
                str(cycle / "declaration-binding.json"): inputs["declaration_sha256"]}
    if any(files.get(name) != digest for name, digest in required.items()):
        raise ValueError("Panel declaration omits or changes required source/input bindings")
    verify_files(files)
    old = read_bound_manifest(cycle / "selection-20261014/selected.json", OLD_SELECTION_SHA)
    old_inputs = old["inputs"]
    if any(old_inputs[key] != inputs[key] for key in ("files_sha256", "compact_binding", "declaration_sha256")):
        raise ValueError("Original boundary selection input identities differ")
    metadata = dict(identity=PANEL_IDENTITY, spec=str(attempt), spec_sha256=PANEL_SPEC_SHA,
                    design_sha256=PANEL_DESIGN_SHA, timing_addendum_sha256=PANEL_TIMING_SHA,
                    declaration=str(declaration_path), declaration_sha256=file_hash(declaration_path),
                    old_selection_sha256=OLD_SELECTION_SHA,
                    historical_source_sha256=old_inputs["source_sha256"])
    return metadata, {**files, str(declaration_path): metadata["declaration_sha256"]}, old["requests"]


def selection_inputs(args):
    cycle, cache, audit = Path(args.cycle), Path(args.cache), Path(args.audit)
    declaration = json.loads((cycle / "declaration-binding.json").read_text())
    verify_files(declaration["files_sha256"])
    with Progress("ecgfounder.compact_binding", device="cpu"):
        binding = compact_cache_binding(cache, expected=declaration["compact_binding"])
    index = json.loads((cache / "index.json").read_text())
    compact = json.loads((cache / "compact.json").read_text())
    role_payload = json.loads((audit / "roles.json").read_text())
    role_hashes = [value for name, value in declaration["files_sha256"].items() if name.endswith("/roles.json")]
    if len(role_hashes) != 1 or file_hash(audit / "roles.json") != role_hashes[0]:
        raise ValueError("Passed audit role file is not the frozen original")
    if role_payload["cache_index_sha256"] != binding["index_sha256"]:
        raise ValueError("Original role/cache identity differs")
    identities = {r["record_id"]: r for r in compact["records"]}
    records = {}
    for entry in index["records"]:
        record = entry["record"]
        rid = record["record_id"]
        records[rid] = dict(identities[rid], duration_seconds=record["duration_seconds"], fs=record["fs"],
                            signal_parts=record["signal_parts"])
    source_paths = [Path("src/cardiotwin/ecgfounder.py"), Path("src/cardiotwin/ecgfounder_probe.py"),
                    Path("scripts/run_ecgfounder_probe.py"), Path("src/cardiotwin/data.py"),
                    Path("src/cardiotwin/precursor.py"), Path("src/cardiotwin/investigation.py"),
                    Path("src/cardiotwin/progress.py"), Path("src/cardiotwin/gpu_telemetry.py"),
                    Path("reports/ml-investigation/monitor-storage.py")]
    source = source_manifest()["files"]
    source.update({str(path): file_hash(path) for path in source_paths})
    if getattr(args, "attempt", None):
        source["scripts/kaggle_ecgfounder_probe.py"] = file_hash("scripts/kaggle_ecgfounder_probe.py")
    inputs = dict(declaration_sha256=file_hash(cycle / "declaration-binding.json"),
                  files_sha256=declaration["files_sha256"], compact_binding=binding,
                  source_sha256=source)
    rows = SelectionRows(cache, compact["records"])
    if getattr(args, "attempt", None):
        if getattr(args, "additional_budget", None):
            raise ValueError("Panel b has no additional budget and cannot authorize a full run")
        attempt, files, old_requests = panel_attempt_inputs(args, inputs)
        requests, support = select_metadata_supported_requests(rows, records, role_payload["roles"], old_requests)
        inputs["files_sha256"] = files
        verify_files(files)
        verify_files(source)
        compact_cache_binding(cache, expected=binding)
        return dict(version="ecgfounder-engineering-selection-v1", attempt=attempt,
                    inputs=inputs, requests=requests, support=support), index
    requests = select_probe_requests(rows, records, role_payload["roles"])
    return dict(version="ecgfounder-engineering-selection-v1", inputs=inputs, requests=requests), index


def storage_measurement():
    path = Path("reports/ml-investigation/monitor-storage.py")
    spec = importlib.util.spec_from_file_location("ecgfounder_storage", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.storage_usage([Path(name) for name in ("data", "reports", "artifacts")])


def resources(started, output, device):
    import torch

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    reserved = torch.cuda.max_memory_reserved(device) if device.startswith("cuda") else 0
    output_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    result = dict(elapsed=time.monotonic() - started, rss=rss, reserved=reserved, output_bytes=output_bytes)
    check_resources(**result)
    return result


def numerical_checks(model, native, request, measured, *, dense_dispatch=False, device="cpu"):
    """Reprocess only the already-selected history in RAM; no additional raw reads."""
    reference_started = time.monotonic()
    start = request["decision_sample"] - 120000
    kwargs = dict(timestamp=request["timestamp"], first_sample=start)
    direct = preprocess_history(native, **kwargs)
    eight = encode_chunks(model, direct)
    one = encode_chunks(model, direct, batch_size=1)
    np.testing.assert_allclose(eight, one, atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(measured["delta"], pool_delta(eight), atol=1e-5, rtol=1e-4)
    suffixed = np.r_[native, np.full(37, np.nan)]
    np.testing.assert_allclose(direct, preprocess_history(suffixed, **kwargs), atol=1e-8, rtol=1e-7)
    result = dict(direct_ring=True, batch1_batch8=True, future_suffix=True)
    if dense_dispatch:
        result["reference_check_seconds"] = time.monotonic() - reference_started
        result["dense_dispatch"] = dense_dispatch_check(model, native, request, eight, measured["delta"], device=device)
    return result


def dense_dispatch_check(model, native, request, reference, delta, *, device):
    """Replay exactly sixty in-memory chunks via twenty unchanged encoder calls of three."""
    import torch

    started = time.monotonic()
    prepared = preprocess_history(native, timestamp=request["timestamp"],
                                  first_sample=request["decision_sample"] - 120000)
    preprocess_seconds = time.monotonic() - started
    if prepared.shape != (60, 5000):
        raise ValueError("Dense dispatch requires exactly sixty native chunks")
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    infer_started = time.monotonic()
    # encode_chunks keeps its frozen batch8 API; each three-row call has exactly one forward.
    vectors = np.concatenate([encode_chunks(model, prepared[i:i + 3]) for i in range(0, 60, 3)])
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    infer_seconds = time.monotonic() - infer_started
    pool_started = time.monotonic()
    pooled = pool_delta(vectors)
    pool_seconds = time.monotonic() - pool_started
    total_seconds = time.monotonic() - started
    compare_started = time.monotonic()
    np.testing.assert_allclose(vectors, reference, atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(pooled, delta, atol=1e-5, rtol=1e-4)
    return dict(status="passed", timing_kind="in_memory_reprocessing", device=device, chunks=60,
                forward_batches=20, preprocess_seconds=preprocess_seconds, infer_seconds=infer_seconds,
                pool_seconds=pool_seconds, total_seconds=total_seconds,
                comparison_seconds=time.monotonic() - compare_started)


def dispatch_summary(results, selected):
    normal = {(r["record_id"], r["decision_sample"]) for r in selected if r["slot"] == "normal"}
    records, histories = {}, 0
    for key in sorted(normal):
        measured = results[key]
        if not measured["available"]:
            continue
        replay = measured.get("numerical_checks", {}).get("dense_dispatch", {})
        if (replay.get("status") != "passed" or replay.get("chunks") != 60
                or not np.isfinite(replay.get("total_seconds", np.nan)) or replay["total_seconds"] <= 0):
            raise ValueError("Missing or invalid mandatory dense dispatch evidence")
        histories += 1
        record = records.setdefault(key[0], dict(record_id=key[0], unique_chunks=0, histories=0,
                                    elapsed_seconds=0., preprocess_seconds=0., infer_seconds=0., pool_seconds=0.))
        record["histories"] += 1
        record["unique_chunks"] += 60
        record["elapsed_seconds"] += replay["total_seconds"]
        for stage in ("preprocess_seconds", "infer_seconds", "pool_seconds"):
            record[stage] += replay[stage]
    if not records:
        raise ValueError("No mandatory dense dispatch evidence")
    values = list(records.values())
    aggregate = sum(r["unique_chunks"] for r in values) / sum(r["elapsed_seconds"] for r in values)
    tenth = float(np.quantile([r["unique_chunks"] / r["elapsed_seconds"] for r in values], .1))
    return dict(status="passed", timing_kind="in_memory_reprocessing", unique_normal_histories=histories,
                chunks=histories * 60, record_timings=values, aggregate_chunks_per_second=aggregate,
                tenth_percentile_chunks_per_second=tenth, planning_chunks_per_second=min(aggregate, tenth))


def check_acquisition_mask(native, signal_parts, decision_sample, slots, available):
    expected = acquisition_calibration_mask(signal_parts, first_sample=decision_sample - 120000,
                                            length=len(native))
    if expected.any() and (available or not np.isnan(native[expected]).all()):
        raise ValueError("Expected acquisition calibration mask was not completely preserved")
    if any(slot in ("initial", "daily") for slot in slots) and not expected.any():
        raise ValueError("Expected acquisition calibration mask has no history intersection")
    clear = "clear" in slots
    if clear and expected.any():
        raise ValueError("t630 acquisition calibration mask unexpectedly intersects history")
    return dict(acquisition_mask_samples=int(expected.sum()), acquisition_mask_complete=True,
                t630_acquisition_clear=clear)


def synthetic_wave(start, end):
    times = np.arange(start, end) / 200
    return np.sin(2 * np.pi * 1.3 * times) + 0.02 * np.cos(2 * np.pi * 29 * times)


def synthetic_daily_check(model):
    offset, end = 16812345, 16818345
    parts = [dict(samples=offset), dict(samples=126000)]
    native = synthetic_wave(end - 120000, end)
    native[-6000:] = np.nan
    check = check_acquisition_mask(native, parts, end, ["daily"], False)
    try:
        preprocess_history(native, timestamp=end / 200, first_sample=end - 120000)
    except InvalidChunk as error:
        if error.reason != "nonfinite_native":
            raise
    else:
        raise ValueError("Synthetic daily calibration did not reject the history")
    clear_end = offset + 126000
    clear = synthetic_wave(offset + 6000, clear_end)
    expected = acquisition_calibration_mask(parts, first_sample=offset + 6000, length=120000)
    if expected.any():
        raise ValueError("Synthetic daily calibration failed exact clearance")
    prepared = preprocess_history(clear, timestamp=clear_end / 200, first_sample=offset + 6000)
    if not np.isfinite(prepared).all():
        raise ValueError("Synthetic daily clear history is not finite")
    return dict(status="passed", daily_sample_offset=offset, decision_sample=end,
                calibration_samples=check["acquisition_mask_samples"], clearance_checked=True)


def synthetic_terminal_check(model):
    native = synthetic_wave(0, 126002)
    ring = HistoryEncoder(model)
    ring.encode(native, timestamp=630., record_id="synthetic-terminal")
    measured = ring.encode(native, timestamp=630.005, record_id="synthetic-terminal")
    if not measured["available"] or measured["new_chunks"] != 60:
        raise ValueError("Synthetic off-grid terminal did not use its exact independent chunks")
    prepared = preprocess_history(native, timestamp=630.005)
    np.testing.assert_array_equal(prepared[0], preprocess_chunk(native[6001:8001]))
    np.testing.assert_array_equal(prepared[-1], preprocess_chunk(native[124001:126001]))
    checks = numerical_checks(model, native[6001:126001],
                              dict(timestamp=630.005, decision_sample=126001), measured)
    native[126001:] = np.nan
    np.testing.assert_array_equal(prepared, preprocess_history(native, timestamp=630.005))
    return dict(status="passed", decision_sample=126001, first_sample=6001, new_chunks=60, checks=checks)


def run_synthetic_fallbacks(model, requests, output, source_sha256):
    """Execute required fixed fixtures in RAM; their work never enters cold raw rates."""
    started = time.monotonic()
    required_slots = [r for r in requests if r.get("synthetic_daily_required") or r.get("synthetic_terminal_required")]
    required = [name for name in ("daily", "terminal") if any(r.get(f"synthetic_{name}_required") for r in required_slots)]
    result = dict(version="ecgfounder-fixed-synthetic-fallback-v1", status="running", required=required,
                  required_slots=required_slots, checks={}, source_sha256=source_sha256, raw_history_requests=0)
    _write(output / "synthetic-checks.json", result)
    for name in required:
        check_started = time.monotonic()
        try:
            check = (synthetic_daily_check if name == "daily" else synthetic_terminal_check)(model)
            if not isinstance(check, dict) or check.get("status") != "passed":
                raise ValueError("missing synthetic fallback passing result")
        except BaseException as error:
            result["checks"][name] = dict(status="failed", error_type=type(error).__name__, error=str(error),
                                          elapsed_seconds=time.monotonic() - check_started)
            result.update(status="failed", elapsed_seconds=time.monotonic() - started)
            _write(output / "synthetic-checks.json", result)
            raise
        result["checks"][name] = {**check, "elapsed_seconds": time.monotonic() - check_started}
        _write(output / "synthetic-checks.json", result)
    result.update(status="passed", elapsed_seconds=time.monotonic() - started)
    _write(output / "synthetic-checks.json", result)
    return result


def execute_probe(args, output, started):
    import torch

    manifest = read_bound_manifest(args.selection, args.selection_sha256)
    current, index = selection_inputs(args)
    if manifest != current:
        raise ValueError("Reviewed selection/source inputs differ from current inputs")
    storage_before = storage_measurement()
    if max(storage_before["logical_bytes"], storage_before["allocated_bytes"]) + 250_000_000 > 25_000_000_000:
        raise RuntimeError("Probe output reservation exceeds project storage cap")
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no CPU fallback permitted")
        device_name = torch.cuda.get_device_name(torch.device(args.device))
        torch.cuda.reset_peak_memory_stats(args.device)
    else:
        device_name = "cpu"
    load_started = time.monotonic()
    model, load_receipt = load_frozen_encoder(Path(args.cycle) / "vendor-source/net1d.py",
                                             Path(args.cycle) / "weights/1_lead_ECGFounder.pth",
                                             device=args.device)
    load_seconds = time.monotonic() - load_started
    selected = [r for r in manifest["requests"] if r["status"] == "selected"]
    chosen = sorted({r["record_id"] for r in selected})
    entries = {entry["record"]["record_id"]: entry for entry in index["records"]}
    runtime_root = getattr(args, "data_root", None)
    if runtime_root:
        staged = read_bound_manifest(args.data_manifest, args.data_manifest_sha256)
        verify_files(staged["files_sha256"], root=runtime_root)
    records = {rid: RecordInfo(**(relocate_record(entries[rid]["record"], runtime_root)
               if runtime_root else entries[rid]["record"])) for rid in chosen}
    overhead_bytes = Path(args.additional_budget).read_bytes() if args.additional_budget else None
    overhead = json.loads(overhead_bytes) if overhead_bytes is not None else {}
    budget_sha256 = hashlib.sha256(overhead_bytes).hexdigest() if overhead_bytes is not None else None
    _write(output / "predeclaration.json", dict(selection_sha256=args.selection_sha256,
           manifest=manifest, checkpoint_load=load_receipt, raw_contracts={rid: entries[rid]["input_contract"]
           for rid in chosen}, device=args.device, device_name=device_name, raw_history_requests=0,
           outcome_associations=False, fits=0, retry=False,
           runtime_data_root=str(Path(runtime_root).resolve()) if runtime_root else None,
           data_manifest_sha256=getattr(args, "data_manifest_sha256", None),
           additional_budget=overhead, additional_budget_sha256=budget_sha256,
           runtime_paths={rid: dict(path=records[rid].path, signal_parts=records[rid].signal_parts) for rid in chosen}))
    state = dict(raw_history_requests=0, successful_raw_history_reads=0, completed_unique_histories=0,
                 current_request=None)
    _write(output / "execution-state.json", state)
    dispatch_state = dict(status="running", timing_kind="in_memory_reprocessing", histories=[],
                          current_request=None, addendum_sha256=manifest.get("attempt", {}).get("timing_addendum_sha256"))
    if manifest.get("attempt"):
        _write(output / "dense-dispatch.json", dispatch_state)
    resources(started, output, args.device)
    synthetic_checks = run_synthetic_fallbacks(model, manifest["requests"], output,
                                               manifest["inputs"]["source_sha256"])
    resources(started, output, args.device)
    hash_started = time.monotonic()
    with Progress("ecgfounder.raw_source_hashing", total=len(chosen), device="cpu") as progress:
        for count, rid in enumerate(chosen, 1):
            if _record_contract(records[rid]) != entries[rid]["input_contract"]:
                raise ValueError(f"Selected raw source identity changed: {rid}")
            resources(started, output, args.device)
            progress.update(count)
    hash_seconds = time.monotonic() - hash_started
    results, record_timings = {}, []
    ring = HistoryEncoder(model)
    with cached_signal_reader() as reader, (GpuTelemetry() if args.device.startswith("cuda") else nullcontext()):
        with (output / "measurements.jsonl").open("x") as handle, Progress(
                "ecgfounder.fixed_panel", total=len({(r["record_id"], r["decision_sample"]) for r in selected}),
                device=args.device, device_name=device_name) as progress:
            for rid in chosen:
                record_seconds, encoded = 0., 0
                requests = sorted({r["decision_sample"]: r for r in selected if r["record_id"] == rid}.values(),
                                  key=lambda r: r["decision_sample"])
                for request in requests:
                    cold_started = time.monotonic()
                    resources(started, output, args.device)
                    state.update(raw_history_requests=state["raw_history_requests"] + 1, current_request=request)
                    _write(output / "execution-state.json", state)
                    read_started = time.monotonic()
                    native = reader(records[rid], (request["decision_sample"] - 120000) / 200,
                                    request["decision_sample"] / 200)
                    read_seconds = time.monotonic() - read_started
                    state["successful_raw_history_reads"] += 1
                    _write(output / "execution-state.json", state)
                    if native.shape != (120000, 2):
                        raise ValueError("Selected history must have exact native two-channel support")
                    native = native[:, 0]
                    matching = [r for r in selected if (r["record_id"], r["decision_sample"])
                                == (rid, request["decision_sample"])]
                    try:
                        mask_checks = check_acquisition_mask(native, records[rid].signal_parts,
                                      request["decision_sample"], [r["slot"] for r in matching], False)
                        measured = ring.encode(native, timestamp=request["timestamp"], record_id=rid,
                                               first_sample=request["decision_sample"] - 120000,
                                               duration=records[rid].duration_seconds)
                        numerical_started = time.monotonic()
                        if mask_checks["acquisition_mask_samples"] and measured["available"]:
                            raise ValueError("Expected acquisition calibration mask did not reject the history")
                        if measured["available"]:
                            if manifest.get("attempt"):
                                replay_normal = any(r["slot"] == "normal" for r in matching)
                                if replay_normal:
                                    dispatch_state["current_request"] = request
                                    _write(output / "dense-dispatch.json", dispatch_state)
                                checks = numerical_checks(model, native, request, measured,
                                    dense_dispatch=replay_normal, device=args.device)
                                if "dense_dispatch" in checks:
                                    checks["dense_dispatch"]["resources"] = resources(started, output, args.device)
                                    dispatch_state["histories"].append(dict(record_id=rid,
                                        decision_sample=request["decision_sample"], **checks["dense_dispatch"]))
                                    dispatch_state["current_request"] = None
                                    _write(output / "dense-dispatch.json", dispatch_state)
                            else:
                                checks = numerical_checks(model, native, request, measured)
                        else:
                            checks = dict(rejection_preserved=True)
                        checks.update(mask_checks)
                        numerical_seconds = time.monotonic() - numerical_started
                    except BaseException as error:
                        if manifest.get("attempt") and dispatch_state["current_request"]:
                            dispatch_state.update(status="failed", error_type=type(error).__name__, error=str(error))
                            _write(output / "dense-dispatch.json", dispatch_state)
                        np.save(output / "failed-selected-native.npy", native, allow_pickle=False)
                        _write(output / "failed-request.json", request)
                        raise
                    measured.update(read_seconds=read_seconds, numerical_checks=checks,
                                    numerical_check_seconds=numerical_seconds)
                    payload = {key: value.tolist() if isinstance(value, np.ndarray) else value
                               for key, value in measured.items()}
                    payload.update(record_id=rid, decision_sample=request["decision_sample"],
                                   request_slots=matching, raw_read_index=state["raw_history_requests"])
                    write_started = time.monotonic()
                    handle.write(json.dumps(payload, allow_nan=False) + "\n")
                    handle.flush()
                    write_seconds = time.monotonic() - write_started
                    measured["write_seconds"] = write_seconds
                    results[rid, request["decision_sample"]] = measured
                    state["completed_unique_histories"] = len(results)
                    _write(output / "execution-state.json", state)
                    encoded += measured["encoded_chunks"]
                    progress.update(len(results), force=True, encoded_chunks=encoded,
                                    raw_history_requests=state["raw_history_requests"],
                                    successful_raw_history_reads=state["successful_raw_history_reads"],
                                    resources=resources(started, output, args.device))
                    record_seconds += time.monotonic() - cold_started - numerical_seconds
                if encoded:
                    record_timings.append(dict(record_id=rid, unique_chunks=encoded, elapsed_seconds=record_seconds))
    dispatch = None
    if manifest.get("attempt"):
        dispatch = dispatch_summary(results, selected)
        dispatch_state.update(status="passed", summary=dispatch)
        _write(output / "dense-dispatch.json", dispatch_state)
    counts = [sum(results[r["record_id"], r["decision_sample"]]["available"] for r in selected
                  if r["fold"] == fold and r["slot"] == "normal") for fold in range(3)]
    if min(counts) < 11 or sum(counts) < 33:
        raise ValueError(f"Fixed panel availability gate failed: {counts}")
    cost = cost_gate(record_timings, overhead, device_name=device_name)
    if manifest.get("attempt"):
        rate = min(cost["planning_chunks_per_second"], dispatch["planning_chunks_per_second"])
        cost.update(planning_chunks_per_second=rate,
                    estimated_seconds=1.5 * (1895675 / rate + sum(overhead.values())),
                    dispatch_chunks_per_second=dispatch["planning_chunks_per_second"], full_run_eligible=False)
    verify_files(manifest["inputs"]["source_sha256"])
    verify_files(manifest["inputs"]["files_sha256"])
    compact_cache_binding(args.cache, expected=manifest["inputs"]["compact_binding"])
    if file_hash(args.selection) != args.selection_sha256:
        raise ValueError("Selection changed during probe")
    if runtime_root and file_hash(args.data_manifest) != args.data_manifest_sha256:
        raise ValueError("Staged data manifest changed during probe")
    if args.additional_budget and file_hash(args.additional_budget) != budget_sha256:
        raise ValueError("Additional whole-run budget changed during probe")
    storage_after = storage_measurement()
    if max(storage_after["logical_bytes"], storage_after["allocated_bytes"]) > 25_000_000_000:
        raise RuntimeError("Project storage cap exceeded")
    return dict(status="passed", numerical_gate="passed", normal_available_by_fold=counts,
                request_slots=len(manifest["requests"]), structural_slots=48 - len(selected),
                raw_history_requests=state["raw_history_requests"],
                successful_raw_history_reads=state["successful_raw_history_reads"],
                completed_unique_histories=state["completed_unique_histories"],
                unique_encoded_chunks=sum(r["unique_chunks"] for r in record_timings),
                synthetic_checks=synthetic_checks, synthetic_check_seconds=synthetic_checks["elapsed_seconds"],
                synthetic_checks_sha256=file_hash(output / "synthetic-checks.json"),
                load_seconds=load_seconds, raw_hash_seconds=hash_seconds, record_timings=record_timings,
                stage_seconds={key: sum(r[key] for r in results.values()) for key in (
                    "read_seconds", "resample_seconds", "filter_seconds", "normalize_seconds",
                    "preprocess_seconds", "infer_seconds", "pool_seconds", "write_seconds")},
                cached_numerical_check_seconds=sum(r["numerical_check_seconds"] for r in results.values()),
                cost=cost, dense_dispatch=dispatch, full_run_eligible=cost["full_run_eligible"],
                resources=resources(started, output, args.device),
                storage_before=storage_before, storage_after=storage_after, outcome_associations=False, fits=0,
                measurements_sha256=file_hash(output / "measurements.jsonl"),
                predeclaration_sha256=file_hash(output / "predeclaration.json"), retry=False)


def failure_binding_hashes(args, output):
    """Best effort after-identities on a failed panel-b run; never replace its original error."""
    paths, errors, hashes = set(), {}, {}
    cycle = Path(args.cycle)
    declarations = [cycle / "declaration-binding.json", cycle / "repair-panel-20261014-b/declaration-binding.json"]
    for path in declarations:
        paths.add(str(path))
        try:
            paths.update(json.loads(path.read_text())["files_sha256"])
        except BaseException as error:
            errors[str(path)] = f"{type(error).__name__}: {error}"
    selection = getattr(args, "selection", None)
    if selection:
        paths.add(str(selection))
    predeclaration = output / "predeclaration.json"
    if predeclaration.is_file():
        try:
            inputs = json.loads(predeclaration.read_text())["manifest"]["inputs"]
            paths.update(inputs["files_sha256"])
            paths.update(inputs["source_sha256"])
        except BaseException as error:
            errors[str(predeclaration)] = f"{type(error).__name__}: {error}"
    for name in sorted(paths):
        try:
            path = Path(name).resolve()
            if not path.is_relative_to(Path.cwd().resolve()):
                raise ValueError("Bound path escapes the project root")
            hashes[name] = file_hash(path)
        except BaseException as error:
            errors[name] = f"{type(error).__name__}: {error}"
    return dict(files_sha256=hashes, hash_errors=errors)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cycle", "cache", "audit", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--mode", required=True, choices=("select", "probe"))
    parser.add_argument("--selection")
    parser.add_argument("--selection-sha256")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--additional-budget", help="JSON file of conservative whole-run overhead seconds")
    parser.add_argument("--attempt", help="Fixed repair-panel-20261014-b/spec.md declaration; default uses original selector")
    parser.add_argument("--data-root", help="Runtime immutable raw root, flat or containing record directories")
    parser.add_argument("--data-manifest", help="Staged raw-root file-hash manifest; required with --data-root")
    parser.add_argument("--data-manifest-sha256")
    args = parser.parse_args(argv)
    if args.mode == "probe" and (not args.selection or not args.selection_sha256):
        parser.error("probe mode requires reviewed --selection and --selection-sha256")
    if args.data_root and (not args.data_manifest or not args.data_manifest_sha256):
        parser.error("--data-root requires bound --data-manifest and --data-manifest-sha256")
    started = time.monotonic()
    output = fresh_output(args.output)
    try:
        if args.mode == "select":
            manifest, _ = selection_inputs(args)
            _write(output / "selected.json", manifest)
            result = dict(status="selected_not_executed", selection_sha256=file_hash(output / "selected.json"),
                          request_slots=len(manifest["requests"]), raw_history_requests=0, fits=0)
            _write(output / "selection-receipt.json", result)
        else:
            result = execute_probe(args, output, started)
        _write(output / "receipt.json", {**result, "elapsed_seconds": time.monotonic() - started})
        return 0
    except BaseException as error:
        (output / "original-error.txt").write_text(traceback.format_exc())
        partial = json.loads((output / "execution-state.json").read_text()) if (output / "execution-state.json").exists() else {}
        if (output / "synthetic-checks.json").exists():
            partial["synthetic_checks"] = json.loads((output / "synthetic-checks.json").read_text())
            partial["synthetic_checks_sha256"] = file_hash(output / "synthetic-checks.json")
        if args.attempt:
            partial["binding_hashes_after"] = failure_binding_hashes(args, output)
            if (output / "dense-dispatch.json").is_file():
                dispatch_state = json.loads((output / "dense-dispatch.json").read_text())
                if dispatch_state["status"] == "running":
                    dispatch_state.update(status="incomplete", original_error=str(error))
                    _write(output / "dense-dispatch.json", dispatch_state)
                partial["dense_dispatch"] = dispatch_state
                partial["dense_dispatch_sha256"] = file_hash(output / "dense-dispatch.json")
        _write(output / "receipt.json", dict(status="failed", full_run_eligible=False, fits=0, retry=False,
               elapsed_seconds=time.monotonic() - started, error_type=type(error).__name__, error=str(error), **partial))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
