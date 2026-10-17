#!/usr/bin/env python3
"""Fixed local synthetic software workload; never grants extraction/fitting authority."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
if os.environ["OMP_NUM_THREADS"] != "1":
    raise ValueError("Set OMP_NUM_THREADS=1 before numerical imports")

import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import platform
import plistlib
import queue
import resource
import runpy
import subprocess
import sys
import time
import traceback
from collections.abc import Sequence
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_info

from cardiotwin import ecgfounder_compare as compare
from cardiotwin import ecgfounder_tables as tables
from cardiotwin import investigation
from cardiotwin.config import load_config
from cardiotwin.ecgfounder_extraction import IncrementalExtractor
from cardiotwin.precursor import EventPair, compare_person_counts, freeze_policy_frontiers
from cardiotwin.progress import Progress

ROOT = Path(__file__).resolve().parents[1]
RECIPE_PATH = "reports/ml-investigation/next-cycle/ecgfounder-cycle-20261013/repair-panel-20261014-b/full-cost-20261015/requirements.json"
DEPENDENCIES = (
    "pyproject.toml",
    "uv.lock",
    "scripts/benchmark_ecgfounder_cost.py",
    "src/cardiotwin/ecgfounder_compare.py",
    "src/cardiotwin/ecgfounder_tables.py",
    "src/cardiotwin/ecgfounder_extraction.py",
    "src/cardiotwin/ecgfounder.py",
    "src/cardiotwin/precursor.py",
    "src/cardiotwin/modeling.py",
    "src/cardiotwin/train.py",
    "src/cardiotwin/investigation.py",
    "src/cardiotwin/evaluation.py",
    "src/cardiotwin/alerts.py",
    "src/cardiotwin/config.py",
    "src/cardiotwin/progress.py",
    "reports/ml-investigation/monitor-storage.py",
    "configs/precursor-cycle.json",
    "src/cardiotwin/features.py",
    RECIPE_PATH,
)
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
MARKER = b"SYNTHETIC COST FIXTURE - NOT A CHECKPOINT - NEVER DESERIALIZE\n"
ROWS = 8192


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".partial")
    with temporary.open("w") as stream:
        json.dump(
            dict(value, timestamp_utc=datetime.now(timezone.utc).isoformat()),
            stream,
            indent=2,
            allow_nan=False,
        )
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def base_receipt(sources, recipe_sha):
    return dict(
        status="prepared",
        evidence_class="local-software-benchmark-only",
        sources_sha256=sources,
        requirements_sha256=recipe_sha,
        full_run_eligible=False,
        actual_outcome_fits=0,
        actual_raw_reads=0,
        actual_checkpoint_loads=0,
        actual_encoder_inference=0,
        network_calls=0,
        jobs=0,
        accepted_remote_upper_seconds=dict.fromkeys(COMPONENTS),
        phases={},
        missing=[
            "intended_worker_cpu_filesystem",
            "dense_source_io",
            "large_table_paging_fsync",
            "checkpoint_bootstrap_network",
            "final_study_drift_diagnostic_adapter",
            "final_model_persistence_replay_adapter",
            "artifact_download",
            "review_calendar",
        ],
    )


def validate_recipe(recipe):
    expected = dict(
        primary_heads=6,
        null_heads=57,
        nuisance_heads=3,
        calibrators=6,
        timing_columns=46,
        augmented_columns=1070,
        nuisance_columns=52,
        scoring_batch_rows=4096,
        bootstrap_samples=10000,
        bootstrap_people=121,
        bootstrap_fields=10,
        bootstrap_comparisons_upper=10,
        policy_candidate_row_visits=26975680,
        outer_frozen_policy_row_visits=12588880,
    )
    if any(recipe["fixed_work"].get(k) != v for k, v in expected.items()):
        raise ValueError("Fixed reviewed workload changed")
    fixed = recipe["synthetic_recipe"]
    if (
        fixed["seed"] != 42
        or fixed["fixture_writer_rows"] != ROWS
        or fixed["fixture_writer_patterns"] != 2
        or fixed["solver_matrix_patterns"] != 2
        or fixed["whole_benchmark_wall_ceiling_seconds"] != 1800
        or fixed["per_phase_wall_ceiling_seconds"] != 1200
        or fixed["process_rss_cap_bytes"] != 10_000_000_000
        or fixed["diagnostic_output_cap_bytes"] != 250_000_000
        or fixed["calibration_logit_max_rows_by_fold"] != [37091, 77030, 59018]
        or fixed["solver_pattern_names"] != ["standard_gaussian", "fixed_correlated_gaussian"]
        or fixed["calibration_patterns"] != ["balanced_synthetic_labels", "one_positive_synthetic_label"]
        or fixed["chronology_patterns"]
        != ["continuous_sawtooth_risk", "fixed_high_low_and_unavailable_blocks"]
        or any(c["accepted_upper_seconds"] is not None for c in recipe["components"])
    ):
        raise ValueError("Fixed synthetic bounds/patterns or missing remote authority changed")
    for role, totals in (
        ("policy", [59020, 62123, 47455]),
        ("stop", [39920, 39821, 26706]),
        ("outer", [195736, 196655, 237053]),
    ):
        if [
            c["total_rows"] for c in recipe["role_shape_maxima_from_verified_census"] if c["role"] == role
        ] != totals:
            raise ValueError("Fixed original aggregate role lengths changed")


def verify_sources(expected, *, root=ROOT):
    if set(expected) != set(DEPENDENCIES):
        raise ValueError("Exact benchmark dependency source manifest required")
    for name, digest in expected.items():
        if not isinstance(digest, str) or len(digest) != 64 or sha(Path(root) / name) != digest:
            raise ValueError(f"Frozen benchmark source differs: {name}")


class Budget:
    def __init__(self, started=None, monotonic=time.monotonic):
        self.monotonic = monotonic
        self.started = monotonic() if started is None else started
        self.phase_started, self.phase = self.started, "bootstrap"

    def stage(self, name, now=None):
        self.phase, self.phase_started = name, self.monotonic() if now is None else now

    def check(self, *, now=None, rss=0, output_bytes=0, storage=None):
        now = self.monotonic() if now is None else now
        if now - self.started > 1800 or now - self.phase_started > 1200:
            raise TimeoutError(f"Synthetic benchmark wall ceiling exceeded: {self.phase}")
        if rss > 10_000_000_000 or output_bytes > 250_000_000:
            raise ValueError("Synthetic benchmark RSS/output ceiling exceeded")
        if (
            storage
            and max(storage["allocated_bytes"], storage["logical_bytes"]) + 250_000_000 > 25_000_000_000
        ):
            raise ValueError("Project cap plus output reserve exceeded")


class TimedProducer:
    def __init__(self, values, monotonic=time.monotonic):
        self.values, self.monotonic = iter(values), monotonic
        self.seconds, self.completed, self.attempted = 0.0, 0, 0

    def __iter__(self):
        return self

    def __next__(self):
        started = self.monotonic()
        self.attempted += 1
        try:
            result = next(self.values)
            self.completed += 1
            return result
        finally:
            self.seconds += self.monotonic() - started


def counted_fit(function, counts, kind, *args, report=None):
    value = counts.setdefault(kind, dict(attempted=0, completed=0))
    value["attempted"] += 1
    if report:
        report("progress", fit_counts={key: dict(item) for key, item in counts.items()})
    result = function(*args)
    value["completed"] += 1
    if report:
        report("progress", fit_counts={key: dict(item) for key, item in counts.items()})
    return result


def counted_evaluation(function, counts, kind, rows, *args, report=None, **kwargs):
    value = counts.setdefault(
        kind, dict(calls_attempted=0, calls_completed=0, rows_attempted=0, rows_completed=0)
    )
    value["calls_attempted"] += 1
    value["rows_attempted"] += len(rows)
    if report:
        report("progress", evaluations={key: dict(item) for key, item in counts.items()})
    result = function(rows, *args, **kwargs)
    value["calls_completed"] += 1
    value["rows_completed"] += len(rows)
    if report:
        report("progress", evaluations={key: dict(item) for key, item in counts.items()})
    return result


def storage_usage(roots):
    return runpy.run_path(str(ROOT / "reports/ml-investigation/monitor-storage.py"))["storage_usage"](roots)


def peak_rss_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)


def filesystem_metadata(output):
    filesystem = os.statvfs(output)
    value = dict(
        type=None,
        device=os.stat(output).st_dev,
        block_size=filesystem.f_frsize,
        available_bytes=filesystem.f_bavail * filesystem.f_frsize,
    )
    try:
        if sys.platform == "darwin":
            listing = subprocess.check_output(
                ["/bin/df", "-P", str(output)], text=True, stderr=subprocess.PIPE, timeout=2
            )
            columns = listing.splitlines()[-1].split(None, 5)
            if len(columns) != 6 or not columns[0].startswith("/dev/"):
                raise ValueError("Filesystem mounted device unavailable")
            device, mount = columns[0], columns[5]
            info = plistlib.loads(
                subprocess.check_output(
                    ["/usr/sbin/diskutil", "info", "-plist", device], stderr=subprocess.PIPE, timeout=2
                )
            )
            if (
                not isinstance(info, dict)
                or not isinstance(info.get("FilesystemType"), str)
                or not info["FilesystemType"]
                or info.get("MountPoint") != mount
                or info.get("DeviceIdentifier") != device.removeprefix("/dev/")
                or os.stat(mount).st_dev != value["device"]
            ):
                raise ValueError("Filesystem plist type/device/mount identity differs")
            value.update(
                type=info["FilesystemType"], mountpoint=mount, device_identifier=info["DeviceIdentifier"]
            )
        else:
            value["type"] = subprocess.check_output(
                ["stat", "-f", "-c", "%T", str(output)], text=True, stderr=subprocess.PIPE, timeout=2
            ).strip()
            if not value["type"]:
                raise ValueError("Filesystem type unavailable")
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as error:
        value.update(type=None, error=f"{type(error).__name__}: {error}")
    return value


def runtime_metadata(output):
    cpu = dict(architecture=platform.machine(), processor=platform.processor())
    if sys.platform == "darwin":
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=2
        )
        cpu["model"] = result.stdout.strip() if result.returncode == 0 else None
        if result.returncode:
            cpu["model_error"] = result.stderr.strip()
    else:
        cpu["model"] = next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            None,
        )
    return dict(
        platform=platform.platform(),
        machine=platform.machine(),
        cpu=cpu,
        cpu_count=os.cpu_count(),
        python=sys.version,
        executable=sys.executable,
        omp_num_threads=os.environ["OMP_NUM_THREADS"],
        numerical_versions={
            name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")
        },
        threadpools=threadpool_info(),
        filesystem=filesystem_metadata(output),
        device="cpu",
    )


class SyntheticRows(Sequence):
    """Deterministic synthetic views; only bounded batches materialize numeric features."""

    def __init__(self, count, record_count=132, people=121):
        self.count, self.record_count, self.people = count, record_count, people
        self.records = [
            dict(record_id=f"r{i:03d}", group_id=f"g{i % people:03d}", patient_id=f"p{i % people:03d}")
            for i in range(record_count)
        ]

    def __len__(self):
        return self.count

    def vectors(self, index):
        pattern = int(index // self.record_count >= 8)
        rng = np.random.default_rng(np.random.SeedSequence([42, pattern, index]))
        values = rng.normal(size=1053)
        if pattern:
            values = rng.normal() + 0.1 * values
        return values[:23], values[23:1047].astype(np.float32), values[1047:].astype(np.float32)

    def __getitem__(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        return dict(
            self.records[index % self.record_count],
            row_index=int(index),
            timestamp=630.0 + (index // self.record_count) * 3000.0,
            accepted=True,
            features=dict(zip(compare.LINEAR_SCHEMA, self.vectors(index)[0], strict=True)),
        )


class TimingRows(Sequence):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]["features"]


@contextmanager
def hash_accounting():
    result = dict(
        table_hash_passes=0,
        table_hash_bytes=0,
        table_hash_seconds=0.0,
        input_hash_passes=0,
        input_hash_bytes=0,
        input_hash_seconds=0.0,
        fsync_calls=0,
        fsync_seconds=0.0,
        rename_calls=0,
        rename_seconds=0.0,
    )
    original_sha, original_fsync, original_replace = tables._sha256, os.fsync, os.replace

    def measured_sha(path):
        table = Path(path).name.split(".npy")[0] in {"delta", "nuisance", "rows"}
        prefix = "table" if table else "input"
        started = time.monotonic()
        try:
            return original_sha(path)
        finally:
            result[prefix + "_hash_passes"] += 1
            result[prefix + "_hash_bytes"] += Path(path).stat().st_size
            result[prefix + "_hash_seconds"] += time.monotonic() - started

    def timed_call(function, prefix, *args):
        started = time.monotonic()
        try:
            return function(*args)
        finally:
            result[prefix + "_calls"] += 1
            result[prefix + "_seconds"] += time.monotonic() - started

    with (
        patch.object(tables, "_sha256", measured_sha),
        patch.object(compare, "_sha256", measured_sha),
        patch.object(os, "fsync", lambda *args: timed_call(original_fsync, "fsync", *args)),
        patch.object(os, "replace", lambda *args: timed_call(original_replace, "rename", *args)),
    ):
        yield result


def write_fixture(root, rows, *, rejected, report=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "checkpoint.marker").write_bytes(MARKER)
    (root / "source.marker").write_text("Fixed synthetic benchmark fixture; no real data")
    (root / "compact.json").write_text(
        json.dumps(dict(features=list(compare.LINEAR_SCHEMA), records=rows.records))
    )

    def metadata():
        for i in range(len(rows)):
            row = rows[i]
            row["accepted"] = not rejected
            yield row

    contract = dict(
        row_count=len(rows),
        identity_sha256=tables.identity_digest(metadata()),
        records=rows.records,
        files_sha256={p: sha(root / p) for p in ("checkpoint.marker", "compact.json")},
        source_sha256={"source.marker": sha(root / "source.marker")},
    )

    def results():
        for i, row in enumerate(metadata()):
            if report and i % 256 == 0:
                report("progress", completed=i, total=len(rows), writer_original_rejected=rejected)
            _, delta, nuisance = rows.vectors(i)
            row = investigation._compact_prediction(row)
            yield dict(
                row,
                available=not rejected,
                decision_sample=round(row["timestamp"] * 200),
                valid_chunks=None if rejected else 60,
                rejections={"waveform_not_evaluated_original_rejection": 1} if rejected else {},
                delta=None if rejected else delta,
                nuisance=None if rejected else nuisance,
            )

    producer = TimedProducer(results())
    output = root / "tables"
    with hash_accounting() as accounting:
        started = time.monotonic()
        tables.write_tables(
            output,
            producer,
            metadata(),
            contract,
            root=root,
            byte_cap=34_337_488,
            storage_check=lambda: storage_usage([ROOT / p for p in ("data", "reports", "artifacts")]),
        )
        accounting["inclusive_wall_seconds"] = time.monotonic() - started
        accounting["producer_seconds"] = producer.seconds
        accounting["rows_completed"] = producer.completed
        accounting["writer_residual_seconds"] = max(
            0.0, accounting["inclusive_wall_seconds"] - producer.seconds - accounting["input_hash_seconds"]
        )
    # Three files per complete pass; store whole-table pass count, not per-file calls.
    accounting["table_hash_file_calls"] = accounting["table_hash_passes"]
    accounting["table_hash_passes"] //= 3
    args = dict(
        output=output,
        metadata_rows=metadata,
        contract=contract,
        root=root,
        receipt_sha256=sha(output / "receipt.json"),
        compact_path="compact.json",
        checkpoint_path="checkpoint.marker",
        feature_schema=list(compare.LINEAR_SCHEMA),
    )
    return output, args, accounting


@contextmanager
def open_synthetic(args):
    args = dict(args)
    root = Path(args["root"])
    if (root / "checkpoint.marker").read_bytes() != MARKER or args["checkpoint_path"] != "checkpoint.marker":
        raise ValueError("Only the generated tiny synthetic marker may replace fixture pins")
    args["metadata_rows"] = args["metadata_rows"]()
    with patch.multiple(
        compare, CHECKPOINT_BYTES=len(MARKER), CHECKPOINT_SHA256=hashlib.sha256(MARKER).hexdigest()
    ):
        data = compare.VerifiedTables.open(**args)
    try:
        yield data
    finally:
        data.close()


def synthetic_pairs(rows, pattern):
    pairs, stop = [], []
    for person in range(121):
        events = 3 if person < 3 else 2
        for event in range(events if person < 119 else 2):
            cycle = 8 * pattern + 2 * event
            negative = cycle * rows.record_count + person
            positive = negative + rows.record_count
            record = rows[positive]
            pair = EventPair(
                record["group_id"],
                record["patient_id"],
                record["record_id"],
                record["timestamp"] + 360.0,
                positive,
                negative,
            )
            (pairs if person < 119 else stop).append(pair)
    return pairs, stop


def chronology(count, accepted, pattern):
    rows, events = [], []
    record_count = min(132, count)
    lengths = [count // record_count + int(i < count % record_count) for i in range(record_count)]
    index = 0
    for record, length in enumerate(lengths):
        for j in range(length):
            original = index < accepted
            available = original and (pattern == 0 or index % 120 < 90)
            reason = "unknown_tail" if j >= length - 2 else "current_AF" if j % 101 == 30 else None
            label = -1 if reason else int(j % 31 == 10)
            risk = ((index % 997) + 0.5) / 997 if pattern == 0 else (0.99 if index // 20 % 2 else 0.01)
            rows.append(
                dict(
                    row_index=index,
                    record_id=f"r{record:03d}",
                    patient_id=f"p{record % 121:03d}",
                    group_id=f"g{record % 121:03d}",
                    timestamp=j * 30.0,
                    duration_seconds=length * 30.0,
                    accepted=available,
                    original_accepted=original,
                    label=label,
                    label_reason=reason,
                    reason=None if available else "synthetic_common_unavailable",
                    risk=risk if available else None,
                )
            )
            index += 1
    for event in range(241):
        record = event % record_count
        events.append(
            dict(
                record_id=f"r{record:03d}",
                patient_id=f"p{record % 121:03d}",
                group_id=f"g{record % 121:03d}",
                onset=600.0 + 300.0 * (event // record_count),
            )
        )
    if {r["label"] for r in rows if r["accepted"] and r["label"] in (0, 1)} != {0, 1}:
        raise ValueError("Fixed synthetic chronology lacks policy class prerequisites")
    return rows, events


def bootstrap_people():
    first, second = [], []
    for i in range(121):
        values = dict(
            group_id=f"g{i:03d}",
            eligible_events=0 if i % 5 == 0 else 2,
            matched_events=0 if i % 5 == 0 else 1,
            false_warnings=i % 3,
            eligible_seconds=0.0 if i == 0 else 3600.0,
            monitored_seconds=7200.0,
            burden_seconds=60.0,
            eligible_negative_seconds=0.0 if i == 0 else 1800.0,
            eligible_negative_burden_seconds=0.0 if i == 0 else 30.0,
            available_seconds=6900.0,
            warning_seconds=30.0,
        )
        first.append(values)
        second.append(dict(values, false_warnings=(i + 1) % 3))
    return first, second


def _core_work(rejected, emit):
    count = ROWS
    duration = (count - 1) * 30.0
    record = dict(
        record_id="core",
        group_id="core",
        patient_id="core",
        fs=200,
        duration_seconds=duration,
        signal_parts=[dict(samples=round(duration * 200))],
    )

    def metadata():
        for i in range(count):
            yield dict(
                row_index=i,
                record_id="core",
                group_id="core",
                patient_id="core",
                timestamp=i * 30.0,
                accepted=not rejected and i >= 21,
            )

    def reader(rec, start, end):
        sample = np.arange(round(start * 200), round(end * 200), dtype=np.float64)
        values = np.sin(sample / 29.0) + sample * 1e-6
        values[sample < 6000] = np.nan
        return np.column_stack((values, values))

    def preprocess(values, *, timings=None):
        if not np.isfinite(values).all():
            raise ValueError("Synthetic core mask crossed accepted history")
        return np.full(5000, values.mean(dtype=np.float64), dtype=np.float32)

    def encode(model, prepared):
        return np.broadcast_to(prepared[:, :1], (len(prepared), 1024)).copy()

    core = IncrementalExtractor(None, reader, encoder=encode, preprocessor=preprocess)
    for i, _ in enumerate(core.iter_record(metadata(), record)):
        if i % 256 == 0:
            emit("progress", completed=i + 1, total=count, synthetic_core=True)
    return core.ledger


def _score_count(head, timing, data, count, counts, *, report=None):
    for name in ("batches_attempted", "batches_completed", "rows_attempted", "rows_completed"):
        counts.setdefault(name, 0)
    counts["applicability"] = "tiled_compute_only"
    for start in range(0, count, 4096):
        indices = np.arange(start, min(start + 4096, count), dtype=np.int64) % len(data.rows)
        counts["batches_attempted"] += 1
        counts["rows_attempted"] += len(indices)
        if report:
            report("progress", scoring=dict(counts))
        logits = head.logits(timing, data, indices)
        if np.shape(logits) != (len(indices),) or not np.isfinite(logits).all():
            raise ValueError("Synthetic all-available scoring unexpectedly abstained")
        counts["batches_completed"] += 1
        counts["rows_completed"] += len(logits)
        if report:
            report("progress", scoring=dict(counts))
    return counts


def _worker(output_name, recipe, expected_sources, messages):
    output = Path(output_name)

    def emit(event, **details):
        messages.put(dict(event=event, monotonic=time.monotonic(), **details))

    phases = {}

    @contextmanager
    def phase(name):
        emit("phase_start", phase=name)
        started = time.monotonic()
        value = dict(status="running")
        try:
            yield value
        except BaseException as error:
            value.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        else:
            value["status"] = "complete"
        finally:
            value["wall_seconds"] = time.monotonic() - started
            value["peak_rss_bytes"] = peak_rss_bytes()
            phases[name] = value
            emit("phase_done", phase=name, measurements=value)

    try:
        verify_sources(expected_sources)
        environment = runtime_metadata(output)
        emit("progress", environment=environment)
        rows = SyntheticRows(ROWS)
        timing = TimingRows(rows)
        with phase("core_writer_reader") as measured:
            measured["core"] = [_core_work(rejected, emit) for rejected in (False, True)]
            _, args, measured["available_writer"] = write_fixture(
                output / "available", rows, rejected=False, report=emit
            )
            _, _, measured["rejected_writer"] = write_fixture(
                output / "rejected", rows, rejected=True, report=emit
            )
        with ExitStack() as stack:
            with phase("verified_reader") as measured:
                with hash_accounting() as reader_accounting:
                    data = stack.enter_context(open_synthetic(args))
                reader_accounting["table_hash_file_calls"] = reader_accounting["table_hash_passes"]
                reader_accounting["table_hash_passes"] //= 3
                measured["accounting"] = reader_accounting
            all_heads, all_calibrators = {}, {}
            for pattern in range(2):
                with phase(f"fixed_heads_pattern_{pattern}") as measured:
                    measured.update(
                        primary_completed=0,
                        null_completed=0,
                        nuisance_completed=0,
                        pair_scoring={},
                        scoring={},
                        fits={},
                    )
                    pairs, stop = synthetic_pairs(rows, pattern)
                    stop_groups = {p.group_id for p in stop}
                    pair_indices = [i for p in pairs + stop for i in (p.positive_index, p.negative_index)]
                    # Fit/stop diagnostics use only declared fit/stop pair indices, never tiled outer groups.
                    for fold in range(3):
                        transforms = compare.fit_transforms(timing, data, pairs)
                        fitter = compare._fit_fixed_logistic
                        with patch.object(
                            compare,
                            "_fit_fixed_logistic",
                            lambda *args: counted_fit(
                                fitter, measured["fits"], "primary", *args, report=emit
                            ),
                        ):
                            heads = compare.fit_primary_heads(timing, data, pairs, transforms)
                        measured["primary_completed"] += 2
                        cells = {
                            (c["fold"], c["role"]): c
                            for c in recipe["role_shape_maxima_from_verified_census"]
                        }
                        for head in heads.values():
                            counted_evaluation(
                                lambda indices: head.logits(timing, data, indices),
                                measured["pair_scoring"],
                                "logits",
                                pair_indices,
                                report=emit,
                            )
                            natural_count = cells[fold, "fit"]["A"] + cells[fold, "stop"]["A"]
                            _score_count(head, timing, data, natural_count, measured["scoring"], report=emit)
                        with patch.object(
                            compare,
                            "_fit_fixed_logistic",
                            lambda *args: counted_fit(fitter, measured["fits"], "null", *args, report=emit),
                        ):
                            for seed, orientation, head in compare.fit_null_heads(
                                timing, data, pairs, transforms, stop_groups=stop_groups
                            ):
                                counted_evaluation(
                                    lambda indices: head.logits(timing, data, indices),
                                    measured["pair_scoring"],
                                    "logits",
                                    pair_indices,
                                    report=emit,
                                )
                                measured["null_completed"] += 1
                                emit(
                                    "progress",
                                    completed=measured["null_completed"],
                                    total=57,
                                    null_seed=seed,
                                    orientation_sha256=hashlib.sha256(tables._json(orientation)).hexdigest(),
                                )
                        with patch.object(
                            compare,
                            "_fit_fixed_logistic",
                            lambda *args: counted_fit(
                                fitter, measured["fits"], "nuisance", *args, report=emit
                            ),
                        ):
                            nuisance = compare.fit_nuisance_head(
                                timing, data, pairs, transforms, stop_groups=stop_groups
                            )
                        counted_evaluation(
                            lambda indices: nuisance.logits(timing, data, indices),
                            measured["pair_scoring"],
                            "logits",
                            pair_indices,
                            report=emit,
                        )
                        measured["nuisance_completed"] += 1
                        all_heads[pattern, fold] = heads
                with phase(f"calibrators_pattern_{pattern}") as measured:
                    measured.update(calibrators_completed=0, scoring={}, scalar_rows=0, fits={})
                    for fold, count in enumerate((37091, 77030, 59018)):
                        for arm, head in all_heads[pattern, fold].items():
                            _score_count(head, timing, data, count, measured["scoring"], report=emit)
                            rng = np.random.default_rng(np.random.SeedSequence([42, 10, pattern, fold]))
                            logits = rng.normal(size=count)
                            labels = np.arange(count) % 2 if pattern == 0 else np.zeros(count, dtype=np.int8)
                            if pattern:
                                labels[count // 2] = 1
                            calibrator = counted_fit(
                                compare.fit_calibrator,
                                measured["fits"],
                                "calibrator",
                                logits,
                                labels,
                                report=emit,
                            )
                            all_calibrators[pattern, fold, arm] = calibrator
                            measured["scalar_rows"] += count
                            measured["calibrators_completed"] += 1
                            emit("progress", completed=measured["calibrators_completed"], total=6)
                with phase(f"policies_streams_pattern_{pattern}") as measured:
                    measured.update(
                        evaluations={},
                        scoring={},
                        serialized_rows=0,
                        adapter_rows=0,
                    )
                    original_evaluate = investigation.evaluate_stream

                    def policy_evaluate(stream, *args, **kwargs):
                        return counted_evaluation(
                            original_evaluate,
                            measured["evaluations"],
                            "policy",
                            stream,
                            *args,
                            report=emit,
                            **kwargs,
                        )

                    config = load_config(ROOT / "configs/precursor-cycle.json")
                    config["alert"].update(
                        policy_version="finite-validity-v1",
                        max_duration_seconds=600.0,
                        rearm_seconds=60.0,
                        max_burden_occupancy=0.05,
                        max_eligible_negative_burden_occupancy=0.05,
                    )
                    cells = {
                        (c["fold"], c["role"]): c for c in recipe["role_shape_maxima_from_verified_census"]
                    }
                    for fold in range(3):
                        for arm, head in all_heads[pattern, fold].items():
                            calibrator = all_calibrators[pattern, fold, arm]
                            adapter = compare.score_rows(head, calibrator, rows, data, range(ROWS))
                            for _ in adapter:
                                measured["adapter_rows"] += 1
                            cell = cells[fold, "policy"]
                            _score_count(head, timing, data, cell["A"], measured["scoring"], report=emit)
                            stream, events = chronology(cell["total_rows"], cell["A"], pattern)
                            with patch.object(investigation, "evaluate_stream", policy_evaluate):
                                policies, _ = counted_evaluation(
                                    freeze_policy_frontiers,
                                    measured["evaluations"],
                                    "frontier",
                                    stream,
                                    events,
                                    config,
                                    deadline=time.monotonic() + 1200,
                                    report=emit,
                                )
                            stop_cell = cells[fold, "stop"]
                            _score_count(head, timing, data, stop_cell["A"], measured["scoring"], report=emit)
                            stop_rows, stop_events = chronology(
                                stop_cell["total_rows"], stop_cell["A"], pattern
                            )
                            counted_evaluation(
                                original_evaluate,
                                measured["evaluations"],
                                "stop",
                                stop_rows,
                                stop_events,
                                config,
                                bootstrap_samples=0,
                                report=emit,
                            )
                            cell = cells[fold, "outer"]
                            _score_count(head, timing, data, cell["A"], measured["scoring"], report=emit)
                            stream, events = chronology(cell["total_rows"], cell["A"], pattern)
                            for policy in policies.values():
                                counted_evaluation(
                                    original_evaluate,
                                    measured["evaluations"],
                                    "outer",
                                    stream,
                                    events,
                                    policy,
                                    bootstrap_samples=0,
                                    report=emit,
                                )
                            investigation._write_predictions(
                                output / f"pattern{pattern}-fold{fold}-{arm}.jsonl.gz", stream
                            )
                            measured["serialized_rows"] += len(stream)
                            emit(
                                "progress",
                                completed=measured["evaluations"]["frontier"]["calls_completed"],
                                total=6,
                            )
                    if (
                        measured["evaluations"]["policy"]["calls_completed"] > 480
                        or measured["evaluations"]["policy"]["rows_completed"] > 26975680
                        or measured["evaluations"]["outer"]["rows_completed"] != 12588880
                    ):
                        raise ValueError("Fixed policy/outer workload accounting differs")
                with phase(f"bootstrap_pattern_{pattern}") as measured:
                    measured.update(
                        comparisons_completed=0,
                        samples_per_comparison=10000,
                        people=121,
                        leave_one_person_out=0,
                        invalid_draws=0,
                    )
                    first, second = bootstrap_people()
                    for _ in range(10):
                        result = compare_person_counts(first, second, seed=42, samples=10000)
                        measured["comparisons_completed"] += 1
                        measured["leave_one_person_out"] += len(result["leave_one_person_out"])
                        measured["invalid_draws"] += result["invalid_draws"]
                        emit("progress", completed=measured["comparisons_completed"], total=10)
        verify_sources(expected_sources)
        emit(
            "complete",
            phases=phases,
            peak_rss_bytes=peak_rss_bytes(),
            environment=environment,
        )
    except BaseException as error:
        emit(
            "failed",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
            phases=phases,
            peak_rss_bytes=peak_rss_bytes(),
        )


def run_benchmark(output, recipe, expected_sources, recipe_sha):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Fresh synthetic benchmark attempt directory required; no retries")
    verify_sources(expected_sources)
    validate_recipe(recipe)
    if recipe_sha != expected_sources[RECIPE_PATH] or recipe != json.loads((ROOT / RECIPE_PATH).read_text()):
        raise ValueError("Synthetic requirements must match the independently frozen source manifest")
    approved_root = (ROOT / RECIPE_PATH).parent / "benchmarks"
    if not output.resolve().is_relative_to(approved_root.resolve()):
        raise ValueError("Fresh benchmark output must be below declared full-cost benchmarks directory")
    before = storage_usage([ROOT / p for p in ("data", "reports", "artifacts")])
    Budget().check(storage={k: v + 70_000_000 for k, v in before.items()})
    output.mkdir(parents=True)
    receipt = base_receipt(expected_sources, recipe_sha)
    receipt.update(
        status="running",
        fixture_recipe=dict(
            seed=42,
            writer_rows=ROWS,
            patterns=2,
            fit_people=119,
            stop_people=2,
            pairs=241,
            unique_fit_rows=482,
            synthetic_heads=132,
            synthetic_calibrators=12,
        ),
        storage_before=before,
    )
    atomic_json(output / "receipt.json", receipt)
    context = mp.get_context("spawn")
    messages = context.Queue()
    process = context.Process(target=_worker, args=(str(output), recipe, expected_sources, messages))
    budget, progress = Budget(), Progress("ecgfounder.synthetic_cost", total=10, device="cpu")
    last_storage = last_log = 0.0
    observed_peak_rss = 0

    def record_failure(error):
        if "error" in receipt:
            receipt.setdefault("secondary_errors", []).append(f"{type(error).__name__}: {error}")
        else:
            receipt.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        if receipt["status"] != "failed":
            receipt["status"] = "partial"

    try:
        process.start()
        while True:
            try:
                message = messages.get(timeout=0.5)
            except queue.Empty:
                message = None
            if message:
                event = message.pop("event")
                if event == "phase_start":
                    budget.stage(message["phase"], now=message["monotonic"])
                    receipt["phase"] = message["phase"]
                    progress.update(phase=message["phase"], force=True)
                elif event == "phase_done":
                    receipt["phases"][message["phase"]] = message["measurements"]
                    if message["measurements"]["status"] == "failed":
                        receipt.setdefault("error", message["measurements"]["error"])
                    budget.check(rss=message["measurements"]["peak_rss_bytes"])
                    atomic_json(output / "receipt.json", receipt)
                elif event == "progress":
                    receipt["last_progress"] = message
                    if "environment" in message:
                        receipt["environment"] = message["environment"]
                elif event in ("complete", "failed"):
                    receipt.update(
                        message, status="complete_software_workload_only" if event == "complete" else "failed"
                    )
                    budget.check()
                    break
            now = time.monotonic()
            rss = 0
            if process.is_alive():
                value = subprocess.run(
                    ["ps", "-o", "rss=", "-p", str(process.pid)], capture_output=True, text=True, timeout=2
                )
                rss = int(value.stdout.strip() or 0) * 1024
            observed_peak_rss = max(observed_peak_rss, rss)
            receipt["observed_peak_rss_bytes"] = observed_peak_rss
            budget.check(now=now, rss=rss)
            if now - last_storage >= 5:
                usage = storage_usage([output])
                budget.check(
                    now=now,
                    output_bytes=max(usage["allocated_bytes"], usage["logical_bytes"]),
                    storage=storage_usage([ROOT / p for p in ("data", "reports", "artifacts")]),
                )
                receipt["output_storage"] = usage
                last_storage = now
            if now - last_log >= 20:
                progress.update(
                    phase=budget.phase,
                    worker_rss_bytes=rss,
                    completed_phases=len(receipt["phases"]),
                    force=True,
                )
                atomic_json(output / "receipt.json", receipt)
                last_log = now
            if not process.is_alive() and message is None:
                raise RuntimeError(
                    f"Synthetic worker exited without terminal receipt, exitcode={process.exitcode}"
                )
    except BaseException as error:
        record_failure(error)
        atomic_json(output / "receipt.json", receipt)
    finally:
        if process.pid is not None and receipt["status"] in ("complete_software_workload_only", "failed"):
            process.join(timeout=5)
        if process.is_alive():
            process.terminate()
        if process.pid is not None:
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join(timeout=1)
        if receipt["status"] == "complete_software_workload_only" and process.exitcode != 0:
            record_failure(RuntimeError(f"Worker ended abnormally: {process.exitcode}"))
        try:
            verify_sources(expected_sources)
        except BaseException as error:
            record_failure(error)
        try:
            terminal_peak = receipt.get("peak_rss_bytes")
            observed_peak_rss = max(
                observed_peak_rss,
                terminal_peak or 0,
                *(p["peak_rss_bytes"] for p in receipt["phases"].values()),
            )
            receipt["final_resources"] = dict(
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                peak_rss_bytes=observed_peak_rss,
                terminal_worker_peak_rss_bytes=terminal_peak,
                output_storage=storage_usage([output]),
                project_storage=storage_usage([ROOT / p for p in ("data", "reports", "artifacts")]),
            )
            final = receipt["final_resources"]
            # Atomic publication briefly retains both receipts; charge its bounded encoded size too.
            publication_bytes = len(json.dumps(receipt, indent=2, allow_nan=False).encode()) + 256
            block_size = os.statvfs(output).f_frsize
            publication_allocated = ((publication_bytes + block_size - 1) // block_size) * block_size
            final["receipt_publication_reserve_bytes"] = max(publication_bytes, publication_allocated)
            budget.check(
                rss=final["peak_rss_bytes"],
                output_bytes=max(
                    final["output_storage"]["allocated_bytes"], final["output_storage"]["logical_bytes"]
                )
                + final["receipt_publication_reserve_bytes"],
                storage={
                    k: v + final["receipt_publication_reserve_bytes"]
                    for k, v in final["project_storage"].items()
                },
            )
            if receipt["status"] == "complete_software_workload_only" and not isinstance(terminal_peak, int):
                raise ValueError("Terminal worker peak RSS evidence missing")
        except BaseException as error:
            record_failure(error)
        receipt.update(elapsed_seconds=time.monotonic() - budget.started, worker_exitcode=process.exitcode)
        atomic_json(output / "receipt.json", receipt)
        progress.finish(status=receipt["status"])
        messages.close()
    if receipt["status"] != "complete_software_workload_only":
        raise RuntimeError(receipt.get("error", "Incomplete synthetic benchmark"))
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.requirements.resolve() != (ROOT / RECIPE_PATH).resolve():
        raise ValueError("Only the frozen reviewed synthetic requirements are accepted")
    run_benchmark(
        args.output,
        json.loads(args.requirements.read_text()),
        json.loads(args.source_manifest.read_text()),
        sha(args.requirements),
    )


if __name__ == "__main__":
    main()
