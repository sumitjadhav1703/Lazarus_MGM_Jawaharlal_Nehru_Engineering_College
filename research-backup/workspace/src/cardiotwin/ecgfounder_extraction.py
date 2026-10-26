"""Outcome-blind incremental ECGFounder core; no table, CLI, fitting or job launcher."""

import math
import resource
import sys
import time
from collections import Counter

import numpy as np

from .ecgfounder import (
    InvalidChunk,
    chunk_bounds,
    decision_sample,
    encode_chunks,
    pool_delta,
    preprocess_chunk,
)


class IncrementalExtractor:
    """Yield original rows with bounded per-record state and strictly historical reads.

    The reader receives (record metadata, start seconds, end seconds). A runner may
    resolve record_id to its bound RecordInfo inside that adapter. Exhaust each
    record iterator before starting the next; a failure or early close cannot resume.
    A CUDA runner must supply resource_check returning measured rss/reserved bytes
    and enforcing its separately declared wall/storage caps.
    """

    def __init__(
        self, model, reader, *, encoder=encode_chunks, preprocessor=preprocess_chunk, resource_check=None
    ):
        self.model, self.reader = model, reader
        self.encoder, self.preprocessor, self.resource_check = encoder, preprocessor, resource_check
        self.cache = {}
        self._records, self._group_patients, self._patient_groups = set(), {}, {}
        self.ledger = dict(
            status="idle",
            record_complete=False,
            current_row=None,
            rows_attempted=0,
            rows_completed=0,
            original_accepted=0,
            available_rows=0,
            original_rejected=0,
            native_reads_attempted=0,
            native_reads_succeeded=0,
            native_reads_completed=0,
            native_samples_attempted=0,
            native_samples_succeeded=0,
            native_samples_completed=0,
            new_chunks=0,
            reused_chunks=0,
            chunks_considered=0,
            valid_new_chunks=0,
            new_invalid_by_reason={},
            encoded_chunks=0,
            encoder_calls_attempted=0,
            encoder_calls_succeeded=0,
            encoder_calls_completed=0,
            encoder_batches_completed=0,
            record_resets=0,
            terminal_resets=0,
            peak_cache_entries=0,
            peak_prepared_chunks=0,
            peak_native_samples=0,
            peak_rss=0,
            peak_reserved=None,
            read_seconds=0.0,
            preprocess_seconds=0.0,
            infer_seconds=0.0,
            pool_seconds=0.0,
            core_seconds=0.0,
            resample_seconds=0.0,
            filter_seconds=0.0,
            normalize_seconds=0.0,
        )
        self._cuda = (
            encoder is encode_chunks and model is not None and next(model.parameters()).device.type == "cuda"
        )
        if self._cuda and resource_check is None:
            raise ValueError("CUDA extraction requires declared resource supervision")

    def _resources(self):
        measured = self.resource_check() if self.resource_check is not None else {}
        if measured is None:
            measured = {}
        if self._cuda and measured.get("reserved") is None:
            raise RuntimeError("CUDA extraction requires measured reserved-memory resource supervision")
        if any(
            not math.isfinite(value) or value < 0
            for name, value in measured.items()
            if name in ("rss", "reserved") and value is not None
        ):
            raise RuntimeError("Invalid extraction resource measurement")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        for name, value, maximum in (
            ("rss", max(rss, measured.get("rss", 0)), 10_000_000_000),
            ("reserved", measured.get("reserved"), 12_000_000_000),
        ):
            if value is None:
                continue
            if not math.isfinite(value) or value < 0 or value > maximum:
                raise RuntimeError("Declared extraction resource cap exceeded")
            prior = self.ledger["peak_" + name]
            self.ledger["peak_" + name] = max(prior or 0, value)

    def _geometry(self, record):
        rid, group, patient = (record[k] for k in ("record_id", "group_id", "patient_id"))
        if (
            not all(isinstance(value, str) and value for value in (rid, group, patient))
            or record["fs"] != 200
            or rid in self._records
        ):
            raise ValueError("Invalid or repeated acquisition identity/fs")
        if (
            self._group_patients.get(group, patient) != patient
            or self._patient_groups.get(patient, group) != group
        ):
            raise ValueError("Group/patient identity must be one-to-one")
        total, masks = 0, []
        for part in record["signal_parts"]:
            count = part["samples"]
            if type(count) is not int or count <= 0:
                raise ValueError("Invalid acquisition sample count")
            masks.append((total, total + min(6000, count)))
            total += count
        duration = record["duration_seconds"]
        if (
            not total
            or not math.isfinite(duration)
            or abs(duration * 200 - total) > 200 * math.ulp(duration) + math.ulp(duration * 200)
        ):
            raise ValueError("Acquisition duration/sample count mismatch")
        self._records.add(rid)
        self._group_patients[group], self._patient_groups[patient] = patient, group
        return total, masks

    def _read(self, record, start, end, decision, masks, timings):
        if not 0 <= start < end <= decision or end - start > 6000:
            raise ValueError("Read exceeds causal native support or bounded slice cap")
        self._resources()
        self.ledger["native_reads_attempted"] += 1
        self.ledger["native_samples_attempted"] += end - start
        started = time.monotonic()
        try:
            values = np.asarray(self.reader(record, start / 200, end / 200))
        finally:
            elapsed = time.monotonic() - started
            timings["read_seconds"] += elapsed
            self.ledger["read_seconds"] += elapsed
        self.ledger["native_reads_succeeded"] += 1
        self.ledger["native_samples_succeeded"] += end - start
        if values.shape != (end - start, 2) or values.dtype.kind not in "fiu":
            raise ValueError("Reader must return exact two-channel native support")
        for left, right in masks:
            a, b = max(start, left), min(end, right)
            if a < b and not np.isnan(values[a - start : b - start, 0]).all():
                raise ValueError("Known acquisition calibration mask was not preserved")
        self.ledger["native_reads_completed"] += 1
        self.ledger["native_samples_completed"] += end - start
        self.ledger["peak_native_samples"] = max(self.ledger["peak_native_samples"], end - start)
        self._resources()
        return values[:, 0]

    def _fill(self, record, bounds, end, masks, timings):
        missing = [key for key in bounds if key not in self.cache]
        self.ledger["new_chunks"] += len(missing)
        self.ledger["reused_chunks"] += 60 - len(missing)
        prepared, positions = [], []
        offset = 0
        while offset < len(missing):
            run = [missing[offset]]
            while (
                len(run) < 3
                and offset + len(run) < len(missing)
                and run[-1][1] == missing[offset + len(run)][0]
            ):
                run.append(missing[offset + len(run)])
            start, stop = run[0][0], run[-1][1]
            native = self._read(record, start, stop, end, masks, timings)
            for key in run:
                values = native[key[0] - start : key[1] - start]
                self.ledger["chunks_considered"] += 1
                stages = ("resample_seconds", "filter_seconds", "normalize_seconds")
                before = {name: timings.get(name, 0.0) for name in stages}
                started = time.monotonic()
                try:
                    try:
                        chunk = self.preprocessor(values, timings=timings)
                    except InvalidChunk as error:
                        self.cache[key] = (None, error.reason, None)
                        reasons = self.ledger["new_invalid_by_reason"]
                        reasons[error.reason] = reasons.get(error.reason, 0) + 1
                    else:
                        if (
                            chunk.shape != (5000,)
                            or chunk.dtype != np.float32
                            or not np.isfinite(chunk).all()
                        ):
                            raise ValueError("Preprocessor must return finite float32 chunk support")
                        scalars = np.array(
                            [
                                values.mean(dtype=np.float64),
                                values.std(dtype=np.float64),
                                np.abs(np.diff(values.astype(np.float64))).mean(),
                            ],
                            dtype=np.float64,
                        )
                        if not np.isfinite(scalars).all():
                            raise ValueError("Nonfinite native nuisance statistics")
                        prepared.append(chunk)
                        positions.append((key, scalars))
                        self.ledger["valid_new_chunks"] += 1
                finally:
                    elapsed = time.monotonic() - started
                    timings["preprocess_seconds"] += elapsed
                    self.ledger["preprocess_seconds"] += elapsed
                    for name in stages:
                        self.ledger[name] += timings.get(name, 0.0) - before[name]
            offset += len(run)
        self.ledger["peak_prepared_chunks"] = max(self.ledger["peak_prepared_chunks"], len(prepared))
        if len(prepared) > 60:
            raise RuntimeError("Prepared chunk cap exceeded")
        if prepared:
            self._resources()
            self.ledger["encoder_calls_attempted"] += 1
            started = time.monotonic()
            try:
                vectors = self.encoder(self.model, np.asarray(prepared, dtype=np.float32))
            finally:
                elapsed = time.monotonic() - started
                timings["infer_seconds"] += elapsed
                self.ledger["infer_seconds"] += elapsed
            self.ledger["encoder_calls_succeeded"] += 1
            if (
                vectors.shape != (len(prepared), 1024)
                or vectors.dtype != np.float32
                or not np.isfinite(vectors).all()
            ):
                raise ValueError("Encoder must return finite float32 width1024 vectors")
            for (key, scalars), vector in zip(positions, vectors, strict=True):
                self.cache[key] = (vector.copy(), None, scalars)
            self.ledger["encoded_chunks"] += len(prepared)
            self.ledger["encoder_calls_completed"] += 1
            self.ledger["encoder_batches_completed"] += (len(prepared) + 7) // 8
            self._resources()
        if len(self.cache) > 60:
            raise RuntimeError("Chunk cache cap exceeded")
        self.ledger["peak_cache_entries"] = max(self.ledger["peak_cache_entries"], len(self.cache))
        return len(missing), 60 - len(missing)

    def iter_record(self, rows, record):
        if self.ledger["status"] not in ("idle", "complete"):
            raise RuntimeError("Extractor is failed or incomplete; automatic resume is prohibited")
        self.ledger.update(status="running", record_complete=False)
        try:
            total, masks = self._geometry(record)
            self.cache.clear()
            self.ledger["record_resets"] += 1
            count = 0
            for number, row in enumerate(rows):
                started = time.monotonic()
                try:
                    self._resources()
                    self.ledger["rows_attempted"] += 1
                    index = row["row_index"]
                    self.ledger["current_row"] = dict(row_index=index, record_id=record["record_id"])
                    if (
                        type(index) is not int
                        or index != self.ledger["rows_completed"]
                        or any(row[k] != record[k] for k in ("record_id", "group_id", "patient_id"))
                    ):
                        raise ValueError("Original row index/acquisition identity mismatch")
                    end, timestamp, accepted = min(number * 6000, total), row["timestamp"], row["accepted"]
                    if (
                        number > (total + 5999) // 6000
                        or not math.isfinite(timestamp)
                        or abs(timestamp * 200 - end)
                        > 200 * math.ulp(record["duration_seconds"]) + math.ulp(timestamp * 200)
                    ):
                        raise ValueError("Original native cadence/coordinate mismatch")
                    if not isinstance(accepted, (bool, np.bool_)):
                        raise ValueError("Invalid original acceptance flag")
                    timings = dict(
                        read_seconds=0.0, preprocess_seconds=0.0, infer_seconds=0.0, pool_seconds=0.0
                    )
                    result = dict(
                        row_index=index,
                        record_id=record["record_id"],
                        group_id=record["group_id"],
                        patient_id=record["patient_id"],
                        timestamp=float(timestamp),
                        decision_sample=end,
                        accepted=bool(accepted),
                        available=False,
                        delta=None,
                        nuisance=None,
                        valid_chunks=None,
                        terminal_reset=False,
                        new_chunks=0,
                        reused_chunks=0,
                        rejections={},
                        original_rejection_reference=None,
                        timings=timings,
                    )
                    if accepted:
                        self.ledger["original_accepted"] += 1
                        decision_sample(timestamp, record["duration_seconds"])
                        bounds = chunk_bounds(timestamp, record["duration_seconds"])
                        if end == total and end % 6000:
                            self.cache.clear()
                            result["terminal_reset"] = True
                            self.ledger["terminal_resets"] += 1
                        self.cache = {key: value for key, value in self.cache.items() if key in bounds}
                        result["new_chunks"], result["reused_chunks"] = self._fill(
                            record, bounds, end, masks, timings
                        )
                        reasons = Counter(
                            self.cache[key][1] for key in bounds if self.cache[key][1] is not None
                        )
                        result.update(rejections=dict(reasons), valid_chunks=60 - sum(reasons.values()))
                        if not reasons:
                            pool_started = time.monotonic()
                            try:
                                vectors = np.asarray([self.cache[key][0] for key in bounds], dtype=np.float32)
                                result["delta"] = pool_delta(vectors)
                                scalars = np.asarray([self.cache[key][2] for key in bounds])
                                result["nuisance"] = np.r_[
                                    scalars.mean(0), scalars[30:].mean(0) - scalars[:30].mean(0)
                                ].astype(np.float32)
                                if not np.isfinite(result["nuisance"]).all():
                                    raise ValueError("Nonfinite pooled nuisance features")
                            finally:
                                elapsed = time.monotonic() - pool_started
                                timings["pool_seconds"] += elapsed
                                self.ledger["pool_seconds"] += elapsed
                            result["available"] = True
                            self.ledger["available_rows"] += 1
                    else:
                        self.ledger["original_rejected"] += 1
                        result["rejections"] = {"waveform_not_evaluated_original_rejection": 1}
                        result["original_rejection_reference"] = index
                    self._resources()
                    self.ledger["rows_completed"] += 1
                    count += 1
                    timings["total_seconds"] = time.monotonic() - started
                finally:
                    self.ledger["core_seconds"] += time.monotonic() - started
                yield result
            if count != (total + 5999) // 6000 + 1:
                raise ValueError("Original record cadence is incomplete")
            self.ledger.update(status="complete", record_complete=True, current_row=None)
        except GeneratorExit:
            self.ledger.update(status="incomplete", record_complete=False)
            raise
        except BaseException as error:
            self.ledger.update(
                status="failed",
                record_complete=False,
                error_type=type(error).__name__,
                original_error=str(error),
            )
            raise
