"""Fixed outcome-blind ECGFounder tables; completion requires the atomic receipt."""

import hashlib
import io
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np

REASONS = (
    "native_support",
    "noncontiguous_native_support",
    "calibration_or_missing",
    "nonfinite_native",
    "native_constant",
    "nonfinite_filtered",
    "filtered_degenerate",
    "nonfinite_normalized",
    "waveform_not_evaluated_original_rejection",
)
ROW_DTYPE = np.dtype(
    [
        ("row_index", "<i8"),
        ("record_index", "<u4"),
        ("timestamp", "<f8"),
        ("decision_sample", "<i8"),
        ("accepted", "?"),
        ("available", "?"),
        ("evaluated", "?"),
        ("valid_chunks", "i1"),
        ("reason_counts", "u1", (len(REASONS),)),
    ]
)
RECEIPT_RESERVE = 250_000
PROJECT_CAP, PROJECT_RESERVE = 25_000_000_000, 250_000_000


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _identity(row, index, records=None):
    rid, group, patient = (row[k] for k in ("record_id", "group_id", "patient_id"))
    timestamp, accepted = row["timestamp"], row["accepted"]
    if (
        type(row["row_index"]) is not int
        or row["row_index"] != index
        or not all(isinstance(v, str) and v for v in (rid, group, patient))
        or not isinstance(accepted, (bool, np.bool_))
        or not math.isfinite(timestamp)
        or timestamp < 0
    ):
        raise ValueError("Invalid original row identity")
    if records is not None and (rid not in records or records[rid][1:] != (group, patient)):
        raise ValueError("Original record/group/patient mapping differs")
    timestamp = float(timestamp)
    scaled = timestamp * 200
    if abs(scaled - round(scaled)) > 200 * math.ulp(max(timestamp, 600.0)) + math.ulp(scaled):
        raise ValueError("Original time is not on the native sample grid")
    return [index, rid, group, patient, timestamp, bool(accepted)]


def identity_digest(rows):
    """SHA256 of canonical UTF-8 identity arrays, each followed by one newline."""
    digest = hashlib.sha256()
    for index, row in enumerate(rows):
        digest.update(_json(_identity(row, index)) + b"\n")
    return digest.hexdigest()


def _sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _snapshot(bindings, root, *, best_effort=False):
    hashes, errors = {}, {}
    for name in sorted(bindings):
        try:
            path = (root / name).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Bound path escapes the project root")
            hashes[name] = _sha256(path)
        except BaseException as error:
            if not best_effort:
                raise
            errors[name] = f"{type(error).__name__}: {error}"
    return dict(files_sha256=hashes, hash_errors=errors)


def _verify(snapshot, bindings):
    if snapshot["hash_errors"] or snapshot["files_sha256"] != bindings:
        raise ValueError("Required source/input bindings changed or could not be verified")


def _freeze_contract(contract):
    count = contract["row_count"]
    if type(count) is not int or count <= 0 or not re.fullmatch("[a-f0-9]{64}", contract["identity_sha256"]):
        raise ValueError("Positive fixed row count and identity SHA256 are required")
    inputs, source = dict(contract["files_sha256"]), dict(contract["source_sha256"])
    if not inputs or not source:
        raise ValueError("Nonempty required input and source bindings are required")
    for name, value in [*inputs.items(), *source.items()]:
        if not isinstance(name, str) or not name or not re.fullmatch("[a-f0-9]{64}", value):
            raise ValueError("Invalid required file binding")
        if name in inputs and name in source and inputs[name] != source[name]:
            raise ValueError("Conflicting required input/source binding")
    records, group_patients, patient_groups = {}, {}, {}
    for number, record in enumerate(contract["records"]):
        rid, group, patient = (record[k] for k in ("record_id", "group_id", "patient_id"))
        if (
            not all(isinstance(v, str) and v for v in (rid, group, patient))
            or rid in records
            or group_patients.setdefault(group, patient) != patient
            or patient_groups.setdefault(patient, group) != group
        ):
            raise ValueError("Invalid record identity mapping")
        records[rid] = (number, group, patient)
    if not records:
        raise ValueError("Original record identity mapping is required")
    frozen = dict(
        row_count=count,
        identity_sha256=contract["identity_sha256"],
        files_sha256=inputs,
        source_sha256=source,
        records=[dict(record_id=rid, group_id=v[1], patient_id=v[2]) for rid, v in records.items()],
    )
    return frozen, records, {**inputs, **source}


def _npy_bytes(shape, dtype):
    header = io.BytesIO()
    np.lib.format.write_array_header_2_0(
        header, dict(descr=np.lib.format.dtype_to_descr(np.dtype(dtype)), fortran_order=False, shape=shape)
    )
    return len(header.getvalue()) + math.prod(shape) * np.dtype(dtype).itemsize


def _storage(storage_check, additional):
    measured = storage_check()
    values = [measured[name] for name in ("allocated_bytes", "logical_bytes")]
    if (
        any(not math.isfinite(v) or v < 0 for v in values)
        or max(values) + additional + PROJECT_RESERVE > PROJECT_CAP
    ):
        raise RuntimeError("Declared project storage cap/reserve exceeded")
    return measured


def _receipt(output, receipt):
    data = _json(receipt) + b"\n"
    if len(data) * 2 > RECEIPT_RESERVE:
        raise RuntimeError("Bounded receipt allowance exceeded")
    path = output / "receipt.json.partial"
    with path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(path, output / "receipt.json")


def _close(arrays, paths):
    errors = []
    for name, array in arrays.items():
        try:
            array.flush()
            with paths[name].open("rb") as handle:
                os.fsync(handle.fileno())
        except BaseException as error:
            errors.append(f"{name}: {type(error).__name__}: {error}")
        finally:
            try:
                array._mmap.close()
            except BaseException as error:
                errors.append(f"{name}: {type(error).__name__}: {error}")
    arrays.clear()
    return errors


def _validate_result(result, expected, index):
    if (
        _identity(result, index) != expected
        or type(result["decision_sample"]) is not int
        or result["decision_sample"] != round(expected[4] * 200)
    ):
        raise ValueError("Extraction row identity/sample differs from original metadata")
    accepted = expected[-1]
    available = result["available"]
    if not isinstance(available, (bool, np.bool_)) or available and not accepted:
        raise ValueError("Unavailable/rejected row cannot be available")
    reasons = result["rejections"]
    if any(
        name not in REASONS or type(count) is not int or not 0 < count <= 60
        for name, count in reasons.items()
    ):
        raise ValueError("Unknown or invalid chunk rejection reason/count")
    if not accepted:
        if result["valid_chunks"] is not None or reasons != {REASONS[-1]: 1}:
            raise ValueError("Original rejection must remain unmeasured")
        valid = -1
    else:
        valid = result["valid_chunks"]
        if (
            type(valid) is not int
            or not 0 <= valid <= 60
            or REASONS[-1] in reasons
            or valid + sum(reasons.values()) != 60
            or bool(available) != (valid == 60)
        ):
            raise ValueError("All60 valid chunks or exact observed rejection counts required")
    for name, width in (("delta", 1024), ("nuisance", 6)):
        values = result[name]
        if available:
            if (
                not isinstance(values, np.ndarray)
                or values.shape != (width,)
                or values.dtype != np.float32
                or not np.isfinite(values).all()
            ):
                raise ValueError("Available vectors must be finite fixed-width float32")
        elif values is not None and (
            not isinstance(values, np.ndarray) or values.shape != (width,) or not np.isnan(values).all()
        ):
            raise ValueError("Unavailable rows cannot contain invented finite features")
    return bool(available), valid, [reasons.get(name, 0) for name in REASONS]


def write_tables(output, results, metadata_rows, contract, *, root, byte_cap, storage_check):
    """Stream fixed tables into a fresh directory; consumers require status=complete.

    Storage callback supplies unique-inode allocated/logical project bytes. Inputs
    and source are opaque hash-bound files; only metadata identity fields are read.
    Table renames are sequential, so filenames alone never establish completeness.
    """
    output, root = Path(output), Path(root).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = dict(
        status="running",
        complete=False,
        full_run_eligible=False,
        rows_written=0,
        current_row=None,
        tables_sha256={},
        reason_names=list(REASONS),
        table_copies=1,
    )
    arrays, paths, bindings = {}, {}, {}
    identity = hashlib.sha256()
    try:
        frozen, records, bindings = _freeze_contract(contract)
        count = frozen["row_count"]
        specs = {
            "delta.npy": ((count, 1024), np.dtype("<f4")),
            "nuisance.npy": ((count, 6), np.dtype("<f4")),
            "rows.npy": ((count,), ROW_DTYPE),
        }
        sizes = {name: _npy_bytes(*spec) for name, spec in specs.items()}
        planned = sum(sizes.values()) + RECEIPT_RESERVE
        receipt.update(
            contract_sha256=hashlib.sha256(_json(frozen)).hexdigest(),
            records=frozen["records"],
            expected_files_sha256=bindings,
            row_count=count,
            planned_bytes=planned,
            table_bytes=sum(sizes.values()),
            table_shapes={name: list(shape) for name, (shape, _) in specs.items()},
            table_dtypes={name: np.lib.format.dtype_to_descr(dtype) for name, (_, dtype) in specs.items()},
        )
        if type(byte_cap) is not int or byte_cap < planned:
            raise RuntimeError("Provisioned byte cap cannot hold tables plus receipt allowance")
        receipt["storage_before"] = _storage(storage_check, planned)
        receipt["hashes_before"] = _snapshot(bindings, root)
        _verify(receipt["hashes_before"], bindings)
        _receipt(output, receipt)
        for name, (shape, dtype) in specs.items():
            paths[name] = output / (name + ".partial")
            arrays[name] = np.lib.format.open_memmap(
                paths[name], mode="w+", dtype=dtype, shape=shape, version=(2, 0)
            )
        expected_rows, values = iter(metadata_rows), iter(results)
        for index in range(count):
            receipt["current_row"] = index
            try:
                expected = _identity(next(expected_rows), index, records)
            except StopIteration as error:
                raise ValueError("Original metadata rows incomplete") from error
            identity.update(_json(expected) + b"\n")
            try:
                result = next(values)
            except StopIteration as error:
                raise ValueError("Extraction rows incomplete") from error
            available, valid, reasons = _validate_result(result, expected, index)
            arrays["delta.npy"][index] = result["delta"] if available else np.nan
            arrays["nuisance.npy"][index] = result["nuisance"] if available else np.nan
            arrays["rows.npy"][index] = (
                index,
                records[expected[1]][0],
                expected[4],
                result["decision_sample"],
                expected[-1],
                available,
                expected[-1],
                valid,
                reasons,
            )
            receipt["rows_written"] += 1
            if receipt["rows_written"] % 4096 == 0:
                _storage(storage_check, 0)
                _receipt(output, receipt)
        sentinel = object()
        if next(expected_rows, sentinel) is not sentinel or next(values, sentinel) is not sentinel:
            raise ValueError("Unexpected extra original/extraction rows")
        receipt["identity_sha256"] = identity.hexdigest()
        if receipt["identity_sha256"] != frozen["identity_sha256"]:
            raise ValueError("Original metadata identity stream digest differs")
        errors = _close(arrays, paths)
        if errors:
            raise OSError("Table flush/close failed: " + "; ".join(errors))
        partial_hashes = {name: _sha256(path) for name, path in paths.items()}
        for name, path in paths.items():
            if path.stat().st_size != sizes[name]:
                raise ValueError("Fixed table size differs from provisioned shape/dtype")
            if (output / name).exists():
                raise FileExistsError("Final table already exists")
            path.rename(output / name)
        receipt["tables_sha256"] = {name: _sha256(output / name) for name in paths}
        if receipt["tables_sha256"] != partial_hashes:
            raise ValueError("Published table hash differs from flushed partial table")
        receipt["hashes_after"] = _snapshot(bindings, root)
        _verify(receipt["hashes_after"], bindings)
        receipt["storage_after"] = _storage(storage_check, 0)
        artifacts = sum(
            p.stat().st_size
            for p in output.iterdir()
            if p.is_file() and p.name not in ("receipt.json", "receipt.json.partial")
        )
        if artifacts + RECEIPT_RESERVE > byte_cap:
            raise RuntimeError("Actual output exceeds provisioned byte cap")
        receipt.update(
            status="complete", complete=True, current_row=None, elapsed_seconds=time.monotonic() - started
        )
        _receipt(output, receipt)
        return receipt
    except BaseException as error:
        receipt.update(
            status="failed",
            complete=False,
            full_run_eligible=False,
            error_type=type(error).__name__,
            original_error=str(error),
            elapsed_seconds=time.monotonic() - started,
            identity_sha256_so_far=identity.hexdigest(),
            close_errors=_close(arrays, paths),
            hashes_after=_snapshot(bindings, root, best_effort=True),
        )
        try:
            _receipt(output, receipt)
        except BaseException:
            # The original processing error must survive an unwritable error receipt.
            pass
        raise
