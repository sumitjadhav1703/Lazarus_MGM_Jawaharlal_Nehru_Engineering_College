from collections.abc import Mapping

import numpy as np
import pytest

from cardiotwin import ecgfounder_extraction as extraction
from cardiotwin.ecgfounder import pool_delta


class BlindRow(Mapping):
    def __init__(self, **values):
        self.values = values

    def __getitem__(self, name):
        assert name in {"row_index", "record_id", "patient_id", "group_id", "timestamp", "accepted"}
        return self.values[name]

    def __iter__(self):
        raise AssertionError("Do not enumerate outcome-bearing row mappings")

    def __len__(self):
        raise AssertionError("Do not inspect outcome-bearing row mappings")


def record(total=138000, rid="r", parts=None):
    return dict(
        record_id=rid,
        patient_id=rid,
        group_id=rid,
        fs=200,
        duration_seconds=total / 200,
        signal_parts=[dict(samples=n) for n in (parts or [total])],
    )


def rows_for(rec, *, accepted=None, first_index=0):
    total = round(rec["duration_seconds"] * 200)
    samples = list(range(0, total + 1, 6000))
    if samples[-1] != total:
        samples.append(total)
    accepted = set(samples if accepted is None else accepted)
    return [
        BlindRow(
            row_index=first_index + i,
            record_id=rec["record_id"],
            patient_id=rec["patient_id"],
            group_id=rec["group_id"],
            timestamp=end / 200,
            accepted=end >= 120000 and end in accepted,
        )
        for i, end in enumerate(samples)
    ]


def wave(start, end):
    sample = np.arange(start, end, dtype=np.float64)
    return np.sin(sample / 29) + sample * 1e-6


def fake_preprocess(values, *, timings=None):
    # Real chunk validity gates, with cheap deterministic numeric work for scheduler tests.
    from cardiotwin.ecgfounder import InvalidChunk

    if values.shape != (2000,):
        raise InvalidChunk("native_support")
    if not np.isfinite(values).all():
        raise InvalidChunk("nonfinite_native")
    if values.max() == values.min():
        raise InvalidChunk("native_constant")
    return np.full(5000, np.mean(values, dtype=np.float64), dtype=np.float32)


def fake_encode(model, prepared):
    return np.broadcast_to(prepared[:, :1], (len(prepared), 1024)).copy()


def setup(rec, *, fault=None, resource_check=None):
    reads, forwards = [], []

    def reader(metadata, start, end):
        a, b = round(start * 200), round(end * 200)
        reads.append((metadata["record_id"], a, b))
        if fault == "read":
            raise OSError("original synthetic read error")
        values = wave(a, b)
        offset = 0
        for part in metadata["signal_parts"]:
            left, right = max(a, offset), min(b, offset + min(6000, part["samples"]))
            if left < right and fault != "mask":
                values[left - a : right - a] = np.nan
            offset += part["samples"]
        if fault == "nonfinite":
            values[(np.arange(a, b) >= 126000) & (np.arange(a, b) < 128000)] = np.nan
        if fault == "constant":
            values[(np.arange(a, b) >= 126000) & (np.arange(a, b) < 128000)] = 7.0
        result = np.column_stack([values, values])
        return result[:-1] if fault == "shape" else result

    def encoder(model, prepared):
        forwards.append(len(prepared))
        if fault == "encoder":
            raise RuntimeError("original synthetic encoder error")
        result = fake_encode(model, prepared)
        if fault == "vectors":
            result[0, 0] = np.nan
        return result

    core = extraction.IncrementalExtractor(
        None, reader, encoder=encoder, preprocessor=fake_preprocess, resource_check=resource_check
    )
    return core, reads, forwards


def direct(end):
    chunks = [fake_preprocess(wave(a, a + 2000)) for a in range(end - 120000, end, 2000)]
    vectors = fake_encode(None, np.asarray(chunks))
    scalars = np.array(
        [
            [v.mean(dtype=np.float64), v.std(dtype=np.float64), np.abs(np.diff(v.astype(np.float64))).mean()]
            for v in [wave(a, a + 2000) for a in range(end - 120000, end, 2000)]
        ]
    )
    nuisance = np.r_[scalars.mean(0), scalars[30:].mean(0) - scalars[:30].mean(0)].astype(np.float32)
    return pool_delta(vectors), nuisance


def test_incremental_three_chunks_match_direct_sixty_and_never_read_future():
    rec = record(138000)
    core, reads, forwards = setup(rec)
    results = list(core.iter_record(rows_for(rec, accepted={126000, 132000, 138000}), rec))
    assert len(results) == 24 and [r["row_index"] for r in results] == list(range(24))
    assert forwards == [60, 3, 3]
    assert len(reads) == 22 and reads[-2:] == [("r", 126000, 132000), ("r", 132000, 138000)]
    assert all(0 <= a < b <= 138000 and b - a <= 6000 for _, a, b in reads)
    for result in results[-3:]:
        delta, nuisance = direct(result["decision_sample"])
        np.testing.assert_allclose(result["delta"], delta, atol=1e-5, rtol=1e-4)
        np.testing.assert_array_equal(result["nuisance"], nuisance)
    assert core.ledger["status"] == "complete" and core.ledger["rows_completed"] == 24
    assert core.ledger["new_chunks"] == 66 and core.ledger["reused_chunks"] == 114
    assert core.ledger["peak_cache_entries"] == 60
    assert core.ledger["native_samples_completed"] == 132000


def test_rejected_rows_keep_order_without_reads_and_later_history_fills_gap():
    rec = record(144000)
    core, reads, forwards = setup(rec)
    stream = core.iter_record(rows_for(rec, accepted={126000, 138000, 144000}), rec)
    results = list(stream)
    rejected = results[22]  # t660 between accepted t630 and690
    assert not rejected["accepted"] and not rejected["available"] and rejected["delta"] is None
    assert rejected["rejections"] == {"waveform_not_evaluated_original_rejection": 1}
    assert rejected["original_rejection_reference"] == 22
    assert forwards == [60, 6, 3]
    assert reads[-3:] == [("r", 126000, 132000), ("r", 132000, 138000), ("r", 138000, 144000)]
    np.testing.assert_allclose(results[-2]["delta"], direct(138000)[0])


def test_offgrid_terminal_direct_reset_and_new_record_never_share_cache():
    rec = record(132001)
    core, reads, forwards = setup(rec)
    results = list(core.iter_record(rows_for(rec, accepted={126000, 132000, 132001}), rec))
    assert forwards == [60, 3, 60] and results[-1]["terminal_reset"]
    assert reads[-20][1] == 12001 and reads[-1][2] == 132001
    np.testing.assert_array_equal(results[-1]["delta"], direct(132001)[0])
    second = record(126000, rid="s")
    more = list(core.iter_record(rows_for(second, accepted={126000}, first_index=len(results)), second))
    assert forwards[-1] == 60 and more[-1]["new_chunks"] == 60
    assert reads[-20][0] == "s" and core.ledger["record_resets"] == 2


@pytest.mark.parametrize("fault,reason", [("nonfinite", "nonfinite_native"), ("constant", "native_constant")])
def test_invalid_chunks_cached_with_original_reason_until_eviction(fault, reason):
    rec = record(138000)
    core, _, forwards = setup(rec, fault=fault)
    results = list(core.iter_record(rows_for(rec, accepted={126000, 132000, 138000}), rec))
    assert results[-2]["rejections"] == {reason: 1} and results[-2]["valid_chunks"] == 59
    assert results[-1]["rejections"] == {reason: 1} and not results[-1]["available"]
    assert forwards == [60, 2, 3]
    assert core.ledger["new_invalid_by_reason"] == {reason: 1}


def test_actual_cumulative_file_masks_partial_chunks_and_exact_clearance():
    rec = record(252003, parts=[126003, 126000])
    core, reads, _ = setup(rec)
    results = list(core.iter_record(rows_for(rec), rec))
    assert not results[20]["available"]  # initial t600 calibration
    assert results[21]["available"]
    assert not results[-2]["available"]  # t1260 history starts132000, last3 mask samples remain
    assert results[-1]["available"]  # exact off-grid terminal starts132003
    assert all(
        b <= result["decision_sample"]
        for result in results
        if result["accepted"]
        for _, a, b in reads
        if a >= result["decision_sample"] - 6000 and b <= result["decision_sample"]
    )


@pytest.mark.parametrize("fault", ["read", "shape", "mask", "encoder", "vectors"])
def test_engineering_errors_stop_and_preserve_attempt_success_complete_counts(fault):
    rec = record(126000)
    core, reads, _ = setup(rec, fault=fault)
    with pytest.raises((ValueError, OSError, RuntimeError)):
        list(core.iter_record(rows_for(rec), rec))
    assert core.ledger["status"] == "failed" and core.ledger["current_row"]["row_index"] == 20
    assert core.ledger["native_reads_attempted"] >= 1
    assert core.ledger["native_reads_succeeded"] == (0 if fault == "read" else len(reads))
    assert core.ledger["native_reads_completed"] == (0 if fault in {"read", "shape", "mask"} else len(reads))
    assert core.ledger["rows_completed"] == 20
    assert core.ledger["original_error"]
    with pytest.raises(RuntimeError, match="failed|incomplete"):
        list(core.iter_record(rows_for(rec), rec))


@pytest.mark.parametrize(
    "fault", ["fs", "parts", "duration", "identity", "cadence", "index", "coordinate", "warmup", "accepted"]
)
def test_malformed_metadata_fails_closed_without_silent_exclusion(fault):
    rec = record(126000)
    rows = rows_for(rec)
    if fault == "fs":
        rec["fs"] = 250
    elif fault == "parts":
        rec["signal_parts"][0]["samples"] = 0
    elif fault == "duration":
        rec["duration_seconds"] += 0.005
    elif fault == "identity":
        rows[0].values["patient_id"] = "other"
    elif fault == "cadence":
        del rows[1]
    elif fault == "index":
        rows[1].values["row_index"] = 0
    elif fault == "coordinate":
        rows[1].values["timestamp"] += 0.001
    elif fault == "warmup":
        rows[0].values["accepted"] = True
    else:
        rows[0].values["accepted"] = 2
    core, _, _ = setup(rec)
    with pytest.raises(ValueError):
        list(core.iter_record(rows, rec))


def test_stream_is_lazy_bounded_and_interrupt_retains_partial_state():
    rec = record(132000)
    core, reads, forwards = setup(rec)
    stream = core.iter_record(rows_for(rec), rec)
    first = next(stream)
    assert first["row_index"] == 0 and not reads and not forwards
    for _ in range(20):
        next(stream)
    assert len(core.cache) <= 60
    stream.close()
    assert core.ledger["status"] == "incomplete" and core.ledger["rows_completed"] == 21
    assert core.ledger["record_complete"] is False


def test_resource_supervision_fails_closed_and_no_caps_are_swallowed():
    rec = record(126000)
    core, reads, _ = setup(rec, resource_check=lambda: {"rss": 10_000_000_001, "reserved": 0})
    with pytest.raises(RuntimeError, match="resource"):
        list(core.iter_record(rows_for(rec), rec))
    assert not reads and core.ledger["status"] == "failed"


@pytest.mark.parametrize("rss", [float("nan"), -1.0, float("inf")])
def test_malformed_resource_measurements_cannot_be_hidden_by_process_rss(rss):
    rec = record(126000)
    core, reads, _ = setup(rec, resource_check=lambda: {"rss": rss, "reserved": 0})
    with pytest.raises(RuntimeError, match="resource"):
        list(core.iter_record(rows_for(rec), rec))
    assert not reads


def test_real_release_preprocessing_incremental_parity_with_fake_encoder():
    from cardiotwin.ecgfounder import preprocess_chunk, preprocess_history

    rec = record(132000)
    core, reads, _ = setup(rec)
    core.preprocessor = preprocess_chunk
    result = list(core.iter_record(rows_for(rec, accepted={126000, 132000}), rec))[-1]
    prepared = preprocess_history(wave(12000, 132000), timestamp=660.0, first_sample=12000)
    expected = pool_delta(fake_encode(None, prepared))
    np.testing.assert_allclose(result["delta"], expected, atol=1e-5, rtol=1e-4)
    assert reads[-1] == ("r", 126000, 132000)
    assert core.ledger["encoder_batches_completed"] == 9
    assert result["timings"]["filter_seconds"] >= 0


def test_consumer_does_not_read_an_accepted_future_row_before_requesting_it():
    rec = record(132000)
    core, reads, forwards = setup(rec)
    stream = core.iter_record(rows_for(rec, accepted={126000, 132000}), rec)
    for _ in range(22):
        result = next(stream)
    assert result["decision_sample"] == 126000 and len(reads) == 20 and forwards == [60]
    assert all(end <= result["decision_sample"] for _, _, end in reads)
    next_result = next(stream)
    assert next_result["decision_sample"] == 132000 and reads[-1] == ("r", 126000, 132000)
    with pytest.raises(StopIteration):
        next(stream)


def test_cuda_requires_actual_reserved_memory_supervision_without_running_encoder():
    from types import SimpleNamespace

    model = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device=SimpleNamespace(type="cuda"))]))
    with pytest.raises(ValueError, match="supervision"):
        extraction.IncrementalExtractor(model, lambda *a: pytest.fail("read"))
    core = extraction.IncrementalExtractor(model, lambda *a: pytest.fail("read"), resource_check=lambda: {})
    with pytest.raises(RuntimeError, match="reserved|resource"):
        list(core.iter_record(rows_for(record(126000)), record(126000)))


@pytest.mark.parametrize("fault", ["missing_terminal", "duplicate_terminal"])
def test_incomplete_or_duplicate_terminal_cannot_mark_record_complete(fault):
    rec = record(126001)
    rows = rows_for(rec, accepted=set())
    if fault == "missing_terminal":
        rows.pop()
    else:
        extra = BlindRow(**rows[-1].values)
        extra.values["row_index"] = len(rows)
        rows.append(extra)
    core, reads, _ = setup(rec)
    with pytest.raises(ValueError, match="cadence|coordinate"):
        list(core.iter_record(rows, rec))
    assert not reads and not core.ledger["record_complete"]
