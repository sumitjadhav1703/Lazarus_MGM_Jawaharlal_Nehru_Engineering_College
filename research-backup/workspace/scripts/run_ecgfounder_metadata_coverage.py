"""Metadata-only known-calibration coverage upper bound; no waveforms or outcomes."""

import argparse
import json
import math
import time
import traceback
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path

import numpy as np

from cardiotwin.data import file_hash
from cardiotwin.progress import Progress

SPEC_SHA256 = "f4930232109a792383b3a1bfa5794d633ce608d33b10d114e43784ec01e0d569"
ADDENDUM_SHA256 = "e53f1a9e2096d19c43118cd08eac47dd4218632c087e4600fd215b985513dea7"
DECLARATION_SHA256 = "35653e952d542e2615e5bacada4191ad3fa8a8092dd77db6c3a2d452c1b8ebeb"
INDEX_KEY = "data/iridia-development-cache-recovered/index.json"
ROLES_KEY = "reports/ml-investigation/next-cycle/precursor-diagnostic-20261012-cadence-fix/roles.json"
EXPECTED_ROWS, EXPECTED_PEOPLE, EXPECTED_RECORDS = 629444, 121, 132
ROLE_NAMES = ("fit", "stop", "calibration", "policy", "outer")


def verify_hashes(bindings):
    for name, expected in bindings.items():
        if file_hash(name) != expected:
            raise ValueError(f"Required hash changed: {name}")


def sample_coordinate(seconds, duration):
    if not math.isfinite(seconds) or not 0 <= seconds <= duration:
        raise ValueError("Invalid in-record timestamp")
    scaled = seconds * 200
    nearest = round(scaled)
    if abs(scaled - nearest) > 200 * math.ulp(duration) + math.ulp(scaled):
        raise ValueError("Timestamp is not on the exact native sample grid")
    return nearest


def calibration_intervals(parts):
    intervals, offset = [], 0
    if not parts:
        raise ValueError("Acquisition has no ordered sample counts")
    for part in parts:
        samples = part["samples"]
        if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
            raise ValueError("Acquisition sample count must be a positive integer")
        intervals.append((offset, offset + min(6000, samples)))
        offset += samples
    return intervals


def intersects_known_mask(start, end, intervals):
    return any(max(start, a) < min(end, b) for a, b in intervals)


def coverage_stats(total, accepted, masked):
    if not 0 <= masked <= accepted <= total:
        raise ValueError("Invalid coverage counts")
    unmasked = accepted - masked
    return dict(total_rows=total, A=accepted, rejected=total - accepted, M=masked, N=unmasked,
                U=unmasked / accepted if accepted else None,
                below_95_percent=20 * unmasked < 19 * accepted if accepted else None,
                status="interpretable" if accepted else "uninterpretable_zero_accepted")


def acquisition_records(index, compact):
    metadata = {}
    for entry in index["records"]:
        record = entry["record"]
        rid = record["record_id"]
        if rid in metadata:
            raise ValueError("Duplicate acquisition record identity")
        metadata[rid] = {key: record[key] for key in (
            "record_id", "patient_id", "fs", "duration_seconds", "signal_parts")}
    records, seen = [], set()
    for identity in compact["records"]:
        rid, pid, group = (identity[key] for key in ("record_id", "patient_id", "group_id"))
        if rid in seen or rid not in metadata or metadata[rid]["patient_id"] != pid:
            raise ValueError("Compact/acquisition identity mismatch")
        if index["group_ids"][pid] != group:
            raise ValueError("Original patient/group identity differs")
        seen.add(rid)
        records.append({**metadata[rid], "group_id": group})
    if seen != set(metadata) or {r["group_id"] for r in records} != set(index["allowed_groups"]):
        raise ValueError("Original acquisition/cohort membership differs")
    return records


class MetadataRows:
    def __init__(self, metadata, records):
        self.metadata, self.records = metadata, records

    def __iter__(self):
        for row in self.metadata:
            code = int(row["record"])
            if not 0 <= code < len(self.records):
                raise ValueError("Unknown compact record code")
            identity = self.records[code]
            yield {**{key: identity[key] for key in ("record_id", "patient_id", "group_id")},
                   "timestamp": float(row["timestamp"]), "duration_seconds": float(row["duration_seconds"]),
                   "accepted": int(row["accepted"])}


def count_coverage(rows, records, roles, *, progress=None):
    groups = {r["group_id"] for r in records}
    if sorted(r["fold"] for r in roles) != [0, 1, 2]:
        raise ValueError("Exactly the original three folds are required")
    membership, cells = {}, {}
    for role in roles:
        fold, assigned = role["fold"], set()
        for name in ROLE_NAMES:
            members = role[name]
            if len(members) != len(set(members)) or assigned.intersection(members) or not set(members) <= groups:
                raise ValueError("Original role membership is duplicated, overlapping or unknown")
            assigned.update(members)
            for group in members:
                membership[fold, group] = name
            cells[fold, name] = [0, 0, 0]
        if assigned != groups:
            raise ValueError("Original role partition is incomplete")
    totals = [0, 0, 0]
    iterator = iter(rows)
    seen = set()
    for record in records:
        rid, duration = record["record_id"], record["duration_seconds"]
        if rid in seen or record["fs"] != 200 or not math.isfinite(duration) or duration <= 0:
            raise ValueError("Invalid acquisition identity/rate/duration")
        seen.add(rid)
        intervals = calibration_intervals(record["signal_parts"])
        samples = sum(part["samples"] for part in record["signal_parts"])
        if sample_coordinate(duration, duration) != samples:
            raise ValueError("Acquisition duration and ordered file samples disagree")
        expected = chain(range(0, samples + 1, 6000), [samples] if samples % 6000 else [])
        for coordinate in expected:
            row = next(iterator, None)
            if row is None:
                raise ValueError("Missing chronological cadence/terminal row")
            if any(row[key] != record[key] for key in ("record_id", "patient_id", "group_id")):
                raise ValueError("Chronological row identity/order differs")
            if row["duration_seconds"] != duration or sample_coordinate(row["timestamp"], duration) != coordinate:
                raise ValueError("Chronological cadence/terminal coordinate differs")
            accepted = row["accepted"]
            if accepted not in (False, True, 0, 1):
                raise ValueError("Cached acceptance must be boolean")
            masked = False
            if accepted:
                if coordinate < 120000:
                    raise ValueError("Accepted row lacks complete in-record history support")
                masked = intersects_known_mask(coordinate - 120000, coordinate, intervals)
            increments = (1, int(accepted), int(masked))
            for i, value in enumerate(increments):
                totals[i] += value
                for fold in range(3):
                    cells[fold, membership[fold, record["group_id"]]][i] += value
            if progress:
                progress(totals[0])
    if next(iterator, None) is not None:
        raise ValueError("Extra chronological rows")
    results = [dict(fold=fold, role=name, **coverage_stats(*cells[fold, name]))
               for fold in range(3) for name in ROLE_NAMES]
    zero = any(cell["A"] == 0 for cell in results)
    ruled_out = any(cell["below_95_percent"] for cell in results)
    return dict(status="failed_zero_accepted" if zero else "passed", unique=coverage_stats(*totals), cells=results,
                records=len(records), people=len({r["patient_id"] for r in records}),
                cadence_identity_checks="passed", checked_rows=totals[0],
                ruling="uninterpretable" if zero else "known_masks_rule_out_unchanged_strict_recipe" if ruled_out
                else "known_calibration_mask_impossibility_not_established",
                interpretation="U is a necessary upper bound, not measured waveform availability or advancement")


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run(args):
    started, started_utc = time.monotonic(), datetime.now(timezone.utc).isoformat()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    state = dict(status="running", checked_rows=0, started_utc=started_utc,
                 new_waveform_reads=0, inference_calls=0, outcome_analyses=0, outcome_fits=0, remote_jobs=0,
                 hdf5_header_reads=0, output_path=str(output.resolve()))
    before = {}
    try:
        cycle, cache, audit = Path(args.cycle), Path(args.cache), Path(args.audit)
        spec, declaration_path = cycle / "metadata-coverage-spec-20261014.md", cycle / "declaration-binding.json"
        addendum = cycle / "metadata-coverage-spec-clarification-20261014.md"
        verify_hashes({str(spec): SPEC_SHA256, str(addendum): ADDENDUM_SHA256,
                       str(declaration_path): DECLARATION_SHA256})
        declaration = json.loads(declaration_path.read_text())
        required = {str(spec): SPEC_SHA256, str(addendum): ADDENDUM_SHA256,
                    str(declaration_path): DECLARATION_SHA256,
                    str(cache / "index.json"): declaration["files_sha256"][INDEX_KEY],
                    str(cache / "compact.json"): declaration["compact_binding"]["compact_sha256"],
                    str(cache / "metadata.npy"): declaration["compact_binding"]["arrays"]["metadata.npy"]["sha256"],
                    str(audit / "roles.json"): declaration["files_sha256"][ROLES_KEY]}
        root = Path(__file__).resolve().parents[1]
        sources = [Path(__file__).resolve(), root / "src/cardiotwin/data.py", root / "src/cardiotwin/progress.py"]
        required.update({str(path): file_hash(path) for path in sources})
        verify_hashes(required)
        before = required.copy()
        state["hashes_before"] = before
        write_json(output / "receipt.json", state)
        index = json.loads((cache / "index.json").read_text())
        compact = json.loads((cache / "compact.json").read_text())
        roles = json.loads((audit / "roles.json").read_text())
        if roles["cache_index_sha256"] != before[str(cache / "index.json")]:
            raise ValueError("Role/index identity mismatch")
        records = acquisition_records(index, compact)
        metadata = np.load(cache / "metadata.npy", mmap_mode="r", allow_pickle=False)
        if (len(metadata) != EXPECTED_ROWS or index["row_count"] != EXPECTED_ROWS
                or len(records) != EXPECTED_RECORDS or len({r["patient_id"] for r in records}) != EXPECTED_PEOPLE):
            raise ValueError("Original full cohort row/person/record counts differ")
        with Progress("ecgfounder.metadata_coverage", total=EXPECTED_ROWS, device="cpu") as progress:
            def update(completed):
                state["checked_rows"] = completed
                if completed % 10000 == 0:
                    progress.update(completed)
            counted = count_coverage(MetadataRows(metadata, records), records, roles["roles"], progress=update)
            progress.update(completed=state["checked_rows"], force=True)
        after = {name: file_hash(name) for name in before}
        state["hashes_after"] = after
        if before != after:
            raise ValueError("Input/source hashes changed after counting")
        result = {**state, **counted, "hashes_after": after, "ended_utc": datetime.now(timezone.utc).isoformat(),
                  "elapsed_seconds": time.monotonic() - started}
        write_json(output / "receipt.json", result)
        return result
    except BaseException as error:
        (output / "original-error.txt").write_text(traceback.format_exc())
        state["hash_errors_after"] = {}
        if "hashes_after" not in state:
            state["hashes_after"] = {}
            for name in before:
                try:
                    state["hashes_after"][name] = file_hash(name)
                except Exception as hash_error:
                    state["hashes_after"][name] = None
                    state["hash_errors_after"][name] = dict(error_type=type(hash_error).__name__, error=str(hash_error))
        state.update(status="invalid", hashes_before=before, error_type=type(error).__name__, error=str(error),
                     ended_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started)
        write_json(output / "receipt.json", state)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cycle", "cache", "audit", "output"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args(argv)
    result = run(args)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
