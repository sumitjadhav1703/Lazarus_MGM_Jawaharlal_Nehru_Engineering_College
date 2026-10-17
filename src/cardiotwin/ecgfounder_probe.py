"""Outcome-blind selection, bounded chunk reuse and engineering-only probe gates."""

import hashlib
import json
import math
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .ecgfounder import (
    InvalidChunk,
    chunk_bounds,
    decision_sample,
    encode_chunks,
    pool_delta,
    preprocess_chunk,
)

OVERHEAD_COMPONENTS = (
    "source_data_hashing", "bootstrap_checkpoint_transfer_loading", "feature_transfer_persistence",
    "six_primary_fits", "null_nuisance_fits", "six_calibrators", "policy_selection", "complete_timelines",
    "person_bootstrap", "artifact_verification_download", "review",
)

PANEL_TAG = "ecgfounder-metadata-supported-panel-20261014-v1"
PANEL_IDENTITY = "ecgfounder-metadata-supported-panel-20261014-b"
PANEL_SPEC_SHA = "792ade5406095eefe2ee0a4cddc9d137b3947ab2a36f5fc22e2ac8ec6f4f9713"
PANEL_DESIGN_SHA = "baf347e0144373baaa7c3de95375783902419f94b7dc177c846247fb68997e42"
OLD_SELECTION_SHA = "01697f20250db697e2e4d9ff5b98a239ba4025aa4e258ac818a157701fa88eca"


def panel_rank(*parts):
    data = json.dumps([PANEL_TAG, 42, *parts], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def select_metadata_supported_requests(rows, records, roles, old_requests):
    """Fixed panel b: stream outcome-blind metadata, retaining one best row per fit group."""
    if (any(type(role["fold"]) is not int for role in roles)
            or sorted(role["fold"] for role in roles) != [0, 1, 2]):
        raise ValueError("Original three fold identities required")
    groups, geometry, group_patients, patient_groups = set(), {}, {}, {}
    for rid, record in records.items():
        if (not isinstance(rid, str) or not rid or record["record_id"] != rid
                or not all(isinstance(record[k], str) and record[k] for k in ("group_id", "patient_id"))
                or record["fs"] != 200):
            raise ValueError("Invalid acquisition identity")
        group, patient = record["group_id"], record["patient_id"]
        if (group_patients.setdefault(group, patient) != patient
                or patient_groups.setdefault(patient, group) != group):
            raise ValueError("Group/patient identities must be one-to-one")
        offset, intervals = 0, []
        for part in record["signal_parts"]:
            count = part["samples"]
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError("Invalid acquisition sample count")
            intervals.append((offset, offset + min(6000, count)))
            offset += count
        duration = record["duration_seconds"]
        if not offset or not math.isfinite(duration) or abs(duration * 200 - offset) > (
                200 * math.ulp(duration) + math.ulp(duration * 200)):
            raise ValueError("Acquisition duration/sample count mismatch")
        geometry[rid] = (offset, intervals)
        groups.add(record["group_id"])
    fit_folds = {}
    for role in roles:
        memberships = [role[name] for name in ("fit", "stop", "calibration", "policy", "outer")]
        flat = [group for members in memberships for group in members]
        if len(flat) != len(set(flat)) or set(flat) != groups:
            raise ValueError("Original roles must be disjoint and cover acquisition groups")
        for group in role["fit"]:
            fit_folds.setdefault(group, []).append(role["fold"])

    # Cadence validation uses one counter per acquisition, including rejected warmup rows.
    seen, best, counts = {}, {}, {}
    for index, row in enumerate(rows):
        rid, group, pid = row["record_id"], row["group_id"], row["patient_id"]
        if rid not in records or group != records[rid]["group_id"] or pid != records[rid]["patient_id"]:
            raise ValueError("Selection row/acquisition identity mismatch")
        accepted, timestamp = row["accepted"], row["timestamp"]
        if not isinstance(accepted, (bool, np.bool_)):
            raise ValueError("Invalid original acceptance flag")
        total, intervals = geometry[rid]
        number = seen.get(rid, 0)
        expected = min(number * 6000, total)
        terminal_number = (total + 5999) // 6000
        if (number > terminal_number or not math.isfinite(timestamp)
                or abs(timestamp * 200 - expected) > (
                    200 * math.ulp(records[rid]["duration_seconds"]) + math.ulp(timestamp * 200))):
            raise ValueError("Original native cadence/coordinate mismatch")
        seen[rid] = number + 1
        if not accepted:
            continue
        if expected < 120000:
            raise ValueError("Accepted row has insufficient native history support")
        if any(expected - 120000 < right and left < expected for left, right in intervals):
            continue
        item = dict(row_index=index, group_id=group, patient_id=pid, record_id=rid,
                    timestamp=float(timestamp), decision_sample=expected)
        for fold in fit_folds.get(group, ()):
            key = (fold, group)
            counts[key] = counts.get(key, 0) + 1
            rank = (panel_rank("row", fold, group, rid, expected), rid.encode("utf-8"), expected)
            if key not in best or rank < best[key][0]:
                best[key] = (rank, item)
    if any(seen.get(rid) != (total + 5999) // 6000 + 1 for rid, (total, _) in geometry.items()):
        raise ValueError("Original native cadence is incomplete")

    boundaries = [r for r in old_requests if r["slot"] != "normal"]
    if len(boundaries) != 12:
        raise ValueError("Exactly twelve original boundary controls required")
    for role in roles:
        fold = role["fold"]
        controls = [r for r in boundaries if r["fold"] == fold]
        if len(controls) != 4 or {r["slot"] for r in controls} != {"initial", "clear", "terminal", "daily"}:
            raise ValueError("Original boundary control identities differ")
        for control in controls:
            if control["role"] != "fit":
                raise ValueError("Original boundary role differs")
            if control["status"] == "structural_not_applicable" and control["slot"] == "daily":
                continue
            rid = control["record_id"]
            if rid not in records:
                raise ValueError("Original boundary acquisition differs")
            record = records[rid]
            expected_time = {"initial": 600., "clear": 630., "terminal": record["duration_seconds"],
                             "daily": record["signal_parts"][0]["samples"] / 200 + 30}[control["slot"]]
            if (control["status"] != "selected" or control["group_id"] not in role["fit"]
                    or any(control[k] != record[k] for k in ("group_id", "patient_id"))
                    or control["timestamp"] != expected_time
                    or control["decision_sample"] != decision_sample(expected_time, record["duration_seconds"])):
                raise ValueError("Original boundary coordinates/identity differ")
    selected, support = [], []
    for role in sorted(roles, key=lambda r: r["fold"]):
        fold = role["fold"]
        qualifying = [group for group in role["fit"] if (fold, group) in best]
        ordered = sorted(qualifying, key=lambda group: (panel_rank("group", fold, group), group.encode("utf-8")))
        if len(ordered) < 12:
            raise ValueError(f"Fold {fold} has fewer than 12 qualifying fit groups")
        for slot_index, group in enumerate(ordered[:12]):
            selected.append(dict(best[fold, group][1], fold=fold, role="fit", slot="normal",
                                 status="selected", slot_index=slot_index))
        selected.extend(r for r in boundaries if r["fold"] == fold)
        support.append(dict(fold=fold, qualifying_groups=len(qualifying),
                            eligible_rows_by_group={g: counts[fold, g] for g in ordered},
                            selected_row_indices=[best[fold, g][1]["row_index"] for g in ordered[:12]]))
    unique = {(r["record_id"], r["decision_sample"]) for r in selected if r["status"] == "selected"}
    return selected, dict(folds=support, unique_histories=len(unique), original_rows=len(rows))


def verify_files(bindings, *, root="."):
    root = Path(root).resolve()
    for name, expected in bindings.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Bound path escapes the project root")
        with path.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Source/input binding changed: {name}")


def read_bound_manifest(path, expected_sha256):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("Bound selection manifest changed")
    return json.loads(data)


def acquisition_calibration_mask(signal_parts, *, first_sample, length):
    """Intersect history coordinates with each actual daily file's first 6000 samples."""
    mask = np.zeros(length, dtype=bool)
    offset = 0
    for part in signal_parts:
        samples = part["samples"]
        start = max(first_sample, offset)
        end = min(first_sample + length, offset + min(6000, samples))
        if start < end:
            mask[start - first_sample:end - first_sample] = True
        offset += samples
    return mask


def relocate_record(record, data_root):
    """Relocate selected acquisition files only; portable content identity stays fixed."""
    root = Path(data_root).resolve()
    rid, old = record["record_id"], Path(record["path"])
    if (record["source"] != "iridia-af" or not re.fullmatch(r"record_[0-9]{3}", rid)
            or old.name != rid or ".." in old.parts):
        raise ValueError("Invalid record identity for relocation")
    directory = root / rid if (root / rid).is_dir() else root
    directory = directory.resolve()
    if not directory.is_relative_to(root):
        raise ValueError("Relocation directory escapes source root")
    parts = []
    for index, part in enumerate(record["signal_parts"]):
        source = Path(part["path"])
        if (source.parent != old or ".." in source.parts or source.name != f"{rid}_ecg_{index:02d}.h5"):
            raise ValueError("Invalid source filename for relocation")
        target = (directory / source.name).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise ValueError("Relocation source missing or escapes source root")
        parts.append({**part, "path": str(target)})
    return {**record, "path": str(directory), "signal_parts": parts}


def select_probe_requests(rows, records, roles):
    """Only identities, accepted flags and acquisition coordinates influence selection."""
    if sorted(r["fold"] for r in roles) != [0, 1, 2]:
        raise ValueError("Original three fold identities required")
    candidates = {}
    for index, row in enumerate(rows):
        if not row["accepted"] or row["timestamp"] < 630:
            continue
        group, rid, pid = row["group_id"], row["record_id"], row["patient_id"]
        record = records[rid]
        if record["group_id"] != group or record["patient_id"] != pid or record["fs"] != 200:
            raise ValueError("Selection row/acquisition identity mismatch")
        timestamp = float(row["timestamp"])
        end = decision_sample(timestamp, record["duration_seconds"])
        item = dict(row_index=index, group_id=group, patient_id=pid, record_id=rid,
                    timestamp=timestamp, decision_sample=end)
        per_group = candidates.setdefault(group, {})
        rank = (abs(timestamp - record["duration_seconds"] / 2), timestamp, index)
        previous = per_group.get(rid)
        if previous is None or rank < previous[0]:
            per_group[rid] = rank, item
    selected = []
    for role in sorted(roles, key=lambda r: r["fold"]):
        fold, fit = role["fold"], set(role["fit"])
        ordered = sorted(fit, key=lambda g: (hashlib.sha256(f"42|{fold}|{g}".encode()).hexdigest(), g))
        normal = []
        for group in ordered:
            if group in candidates:
                rid = min(candidates[group])
                normal.append(dict(candidates[group][rid][1], fold=fold, role="fit", slot="normal",
                                   status="selected", slot_index=len(normal)))
            if len(normal) == 12:
                break
        if len(normal) != 12:
            raise ValueError(f"Fold {fold} has fewer than 12 qualifying fit groups")
        selected.extend(normal)
        eligible = sorted(rid for rid, record in records.items()
                          if record["group_id"] in fit and record["duration_seconds"] >= 630)
        if not eligible:
            raise ValueError("No fit acquisition qualifies for boundary requests")

        def boundary(rid, slot, timestamp, **extra):
            record = records[rid]
            return dict(fold=fold, role="fit", group_id=record["group_id"], patient_id=record["patient_id"],
                        record_id=rid, timestamp=timestamp, decision_sample=decision_sample(
                            timestamp, record["duration_seconds"]), slot=slot, status="selected", **extra)

        selected.extend(boundary(eligible[0], slot, timestamp) for slot, timestamp in (("initial", 600.), ("clear", 630.)))
        terminals = [rid for rid in eligible if decision_sample(records[rid]["duration_seconds"]) % 6000]
        rid = terminals[0] if terminals else eligible[0]
        selected.append(boundary(rid, "terminal", records[rid]["duration_seconds"],
                                 synthetic_terminal_required=not bool(terminals)))
        daily = sorted(rid for rid, record in records.items() if record["group_id"] in fit
                       and len(record["signal_parts"]) >= 2 and record["signal_parts"][0]["samples"] >= 120000
                       and record["signal_parts"][1]["samples"] >= 6000)
        if daily:
            rid = daily[0]
            selected.append(boundary(rid, "daily", records[rid]["signal_parts"][0]["samples"] / 200 + 30))
        else:
            selected.append(dict(fold=fold, role="fit", slot="daily", status="structural_not_applicable",
                                 reason="No fit record has two qualifying daily files", synthetic_daily_required=True))
    return selected


class HistoryEncoder:
    """At most the current history's 60 embeddings; reset at each record boundary."""

    def __init__(self, model):
        self.model, self.record_id, self.cache = model, None, {}

    def encode(self, native, *, timestamp, record_id, first_sample=0, duration=None, validity=None):
        native = np.asarray(native)
        bounds = chunk_bounds(timestamp, duration)
        if native.ndim != 1 or bounds[0][0] < first_sample or bounds[-1][1] > first_sample + len(native):
            raise InvalidChunk("native_support")
        if validity is not None:
            validity = np.asarray(validity)
            if validity.shape != native.shape or validity.dtype != np.bool_:
                raise InvalidChunk("calibration_or_missing")
        if record_id != self.record_id:
            self.cache, self.record_id = {}, record_id
        self.cache = {key: value for key, value in self.cache.items() if key in bounds}
        missing = [key for key in bounds if key not in self.cache]
        prepared, positions = [], []
        timings = dict(resample_seconds=0., filter_seconds=0., normalize_seconds=0.)
        started = time.monotonic()
        for key in missing:
            section = slice(key[0] - first_sample, key[1] - first_sample)
            values = native[section]
            try:
                chunk = preprocess_chunk(values, validity=None if validity is None else validity[section], timings=timings)
            except InvalidChunk as error:
                self.cache[key] = (None, error.reason, None)
            else:
                nuisance = np.array([np.mean(values, dtype=np.float64), np.std(values, dtype=np.float64),
                                     np.mean(np.abs(np.diff(values.astype(np.float64))))])
                prepared.append(chunk)
                positions.append((key, nuisance))
        preprocess_seconds = time.monotonic() - started
        started = time.monotonic()
        if prepared:
            vectors = encode_chunks(self.model, np.asarray(prepared, dtype=np.float32))
            for (key, nuisance), vector in zip(positions, vectors, strict=True):
                self.cache[key] = (vector, None, nuisance)
        infer_seconds = time.monotonic() - started
        started = time.monotonic()
        reasons = Counter(self.cache[key][1] for key in bounds if self.cache[key][1] is not None)
        delta = nuisance = None
        if not reasons:
            delta = pool_delta(np.asarray([self.cache[key][0] for key in bounds], dtype=np.float32))
            scalars = np.asarray([self.cache[key][2] for key in bounds])
            nuisance = np.r_[scalars.mean(0), scalars[30:].mean(0) - scalars[:30].mean(0)].astype(np.float32)
            if not np.isfinite(nuisance).all():
                raise ValueError("Nonfinite nuisance features")
        return dict(available=not bool(reasons), delta=delta, nuisance=nuisance, rejections=dict(reasons),
                    valid_chunks=60 - sum(reasons.values()), new_chunks=len(missing), encoded_chunks=len(prepared),
                    preprocess_seconds=preprocess_seconds, infer_seconds=infer_seconds,
                    pool_seconds=time.monotonic() - started, **timings)


def check_resources(*, elapsed, rss, reserved, output_bytes):
    values = (elapsed, rss, reserved, output_bytes)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise RuntimeError("Invalid resource measurement")
    if elapsed > 1800 or rss > 10_000_000_000 or reserved > 12_000_000_000 or output_bytes > 250_000_000:
        raise RuntimeError("Declared probe resource cap exceeded")


def cost_gate(records, overhead, *, device_name, now_utc=None):
    if not records or any(not math.isfinite(r["elapsed_seconds"]) or r["elapsed_seconds"] <= 0
                          or r["unique_chunks"] <= 0 for r in records):
        raise ValueError("Finite positive cold record timings and chunk counts required")
    aggregate = sum(r["unique_chunks"] for r in records) / sum(r["elapsed_seconds"] for r in records)
    tenth = float(np.quantile([r["unique_chunks"] / r["elapsed_seconds"] for r in records], 0.1))
    rate = min(aggregate, tenth)
    missing = [name for name in OVERHEAD_COMPONENTS if name not in overhead]
    if any(not math.isfinite(value) or value < 0 for value in overhead.values()):
        raise ValueError("Overhead budgets must be finite nonnegative seconds")
    estimate = 1.5 * (1895675 / rate + sum(overhead.values()))
    unmeasured = [name for name in OVERHEAD_COMPONENTS if overhead.get(name, 0) <= 0]
    deadline = datetime(2026, 10, 17, 12, 30, tzinfo=timezone.utc)
    now_utc = datetime.now(timezone.utc) if now_utc is None else now_utc
    deadline_remaining = (deadline - now_utc).total_seconds()
    return dict(full_run_eligible=not unmeasured and "T4" in device_name
                and estimate <= min(41100, deadline_remaining),
                estimated_seconds=estimate, planning_chunks_per_second=rate,
                aggregate_chunks_per_second=aggregate, tenth_percentile_chunks_per_second=tenth,
                missing_overhead_components=missing, unmeasured_overhead_components=unmeasured,
                deadline_remaining_seconds=deadline_remaining, intended_device="T4", measured_device=device_name)
