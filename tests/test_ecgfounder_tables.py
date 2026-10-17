import hashlib
import json
from collections.abc import Mapping

import numpy as np
import pytest

from cardiotwin import ecgfounder_tables as tables


class BlindIdentity(Mapping):
    def __init__(self, **values):
        self.values = values

    def __getitem__(self, key):
        assert key in {"row_index", "record_id", "patient_id", "group_id", "timestamp", "accepted"}
        return self.values[key]

    def __iter__(self):
        raise AssertionError("Do not enumerate outcome-bearing metadata")

    def __len__(self):
        raise AssertionError("Do not inspect outcomes")


def fixture(tmp_path):
    metadata = [
        BlindIdentity(row_index=i, record_id="r", patient_id="p", group_id="g", timestamp=t, accepted=i > 0)
        for i, t in enumerate((0.0, 630.0, 660.0))
    ]
    results = []
    for i, row in enumerate(metadata):
        result = {
            key: row[key]
            for key in ("row_index", "record_id", "patient_id", "group_id", "timestamp", "accepted")
        }
        result.update(
            decision_sample=round(row["timestamp"] * 200),
            available=i == 1,
            valid_chunks=None if i == 0 else 60 if i == 1 else 59,
            rejections={"waveform_not_evaluated_original_rejection": 1}
            if i == 0
            else {}
            if i == 1
            else {"nonfinite_native": 1},
            delta=np.arange(1024, dtype=np.float32) if i == 1 else None,
            nuisance=np.arange(6, dtype=np.float32) if i == 1 else None,
        )
        results.append(result)
    paths = []
    for name in ("metadata.bin", "source.py"):
        path = tmp_path / name
        path.write_text("synthetic bound " + name)
        paths.append(path)
    contract = dict(
        row_count=3,
        identity_sha256=tables.identity_digest(metadata),
        records=[dict(record_id="r", patient_id="p", group_id="g")],
        files_sha256={"metadata.bin": hashlib.sha256(paths[0].read_bytes()).hexdigest()},
        source_sha256={"source.py": hashlib.sha256(paths[1].read_bytes()).hexdigest()},
    )
    return metadata, results, contract


def write(tmp_path, metadata, results, contract, **kwargs):
    return tables.write_tables(
        tmp_path / "output",
        iter(results),
        iter(metadata),
        contract,
        root=tmp_path,
        byte_cap=1_000_000,
        storage_check=lambda: dict(allocated_bytes=0, logical_bytes=0),
        **kwargs,
    )


def test_fixed_tables_preserve_every_row_nan_semantics_hashes_and_fresh_output(tmp_path):
    metadata, results, contract = fixture(tmp_path)
    receipt = write(tmp_path, metadata, results, contract)
    assert (
        receipt["status"] == "complete" and receipt["rows_written"] == 3 and not receipt["full_run_eligible"]
    )
    delta = np.load(tmp_path / "output/delta.npy", mmap_mode="r", allow_pickle=False)
    nuisance = np.load(tmp_path / "output/nuisance.npy", mmap_mode="r", allow_pickle=False)
    status = np.load(tmp_path / "output/rows.npy", mmap_mode="r", allow_pickle=False)
    assert delta.shape == (3, 1024) and nuisance.shape == (3, 6) and delta.dtype == np.float32
    np.testing.assert_array_equal(delta[1], results[1]["delta"])
    assert np.isnan(delta[[0, 2]]).all() and np.isnan(nuisance[[0, 2]]).all()
    assert status["row_index"].tolist() == [0, 1, 2]
    assert status["accepted"].tolist() == [False, True, True]
    assert status["available"].tolist() == [False, True, False]
    assert status["evaluated"].tolist() == [False, True, True]
    assert status["valid_chunks"].tolist() == [-1, 60, 59]
    assert status["reason_counts"][0].sum() == 1 and status["reason_counts"][2].sum() == 1
    assert receipt["identity_sha256"] == contract["identity_sha256"]
    for name, digest in receipt["tables_sha256"].items():
        assert hashlib.sha256((tmp_path / "output" / name).read_bytes()).hexdigest() == digest
    assert not list((tmp_path / "output").glob("*.partial"))
    assert receipt["planned_bytes"] <= 1_000_000
    assert receipt["table_bytes"] == sum(
        (tmp_path / "output" / n).stat().st_size for n in receipt["tables_sha256"]
    )
    with pytest.raises(FileExistsError):
        write(tmp_path, metadata, results, contract)


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "extra",
        "duplicate",
        "time",
        "accepted",
        "record",
        "group",
        "patient",
        "sample",
        "width",
        "dtype",
        "nonfinite",
        "nuisance",
        "valid",
        "reason",
        "rejected_features",
        "rejected_available",
    ],
)
def test_bad_rows_cannot_publish_complete_table(tmp_path, fault):
    metadata, results, contract = fixture(tmp_path)
    if fault == "missing":
        results.pop()
    elif fault == "extra":
        results.append(dict(results[-1]))
    elif fault == "duplicate":
        results[1]["row_index"] = 0
    elif fault == "time":
        results[1]["timestamp"] = 631.0
    elif fault == "accepted":
        results[1]["accepted"] = False
    elif fault == "record":
        results[1]["record_id"] = "other"
    elif fault == "group":
        results[1]["group_id"] = "other"
    elif fault == "patient":
        results[1]["patient_id"] = "other"
    elif fault == "sample":
        results[1]["decision_sample"] += 1
    elif fault == "width":
        results[1]["delta"] = np.zeros(1000, np.float32)
    elif fault == "dtype":
        results[1]["delta"] = results[1]["delta"].astype(np.float64)
    elif fault == "nonfinite":
        results[1]["delta"][0] = np.nan
    elif fault == "nuisance":
        results[1]["nuisance"] = np.zeros(5, np.float32)
    elif fault == "valid":
        results[1]["valid_chunks"] = 59
    elif fault == "reason":
        results[2]["rejections"] = {"unknown": 1}
    elif fault == "rejected_features":
        results[0]["delta"] = np.zeros(1024, np.float32)
    else:
        results[0]["available"] = True
    with pytest.raises(ValueError):
        write(tmp_path, metadata, results, contract)
    receipt = json.loads((tmp_path / "output/receipt.json").read_text())
    assert receipt["status"] == "failed" and not receipt["full_run_eligible"] and receipt["original_error"]
    assert not receipt.get("complete", False)


@pytest.mark.parametrize(
    "fault",
    [
        "identity_hash",
        "input_before",
        "input_during",
        "source_deleted",
        "missing_metadata",
        "extra_metadata",
        "mapping",
    ],
)
def test_bound_metadata_and_source_failures_preserve_after_hashes(tmp_path, fault):
    metadata, results, contract = fixture(tmp_path)
    if fault == "identity_hash":
        contract["identity_sha256"] = "0" * 64
    elif fault == "input_before":
        (tmp_path / "metadata.bin").write_text("changed")
    elif fault == "missing_metadata":
        metadata.pop()
    elif fault == "extra_metadata":
        metadata.append(metadata[-1])
    elif fault == "mapping":
        contract["records"][0]["patient_id"] = "other"

    def stream():
        for i, row in enumerate(results):
            if i == 1 and fault == "input_during":
                (tmp_path / "metadata.bin").write_text("changed during extraction")
            if i == 1 and fault == "source_deleted":
                (tmp_path / "source.py").unlink()
            yield row

    with pytest.raises((ValueError, OSError)):
        write(tmp_path, metadata, stream(), contract)
    receipt = json.loads((tmp_path / "output/receipt.json").read_text())
    assert receipt["status"] == "failed" and "hashes_after" in receipt
    if fault == "source_deleted":
        assert "source.py" in receipt["hashes_after"]["hash_errors"]


def test_iterator_original_error_survives_after_hash_errors(tmp_path):
    metadata, results, contract = fixture(tmp_path)

    def stream():
        yield results[0]
        (tmp_path / "source.py").unlink()
        raise OSError("original iterator failure")

    with pytest.raises(OSError, match="original iterator failure"):
        write(tmp_path, metadata, stream(), contract)
    receipt = json.loads((tmp_path / "output/receipt.json").read_text())
    assert receipt["rows_written"] == 1 and receipt["original_error"] == "original iterator failure"
    assert "source.py" in receipt["hashes_after"]["hash_errors"]
    assert list((tmp_path / "output").glob("*.partial"))


@pytest.mark.parametrize("fault", ["byte_cap", "storage", "empty_sources", "escape"])
def test_reservation_and_required_bindings_fail_before_table_allocation(tmp_path, fault):
    metadata, results, contract = fixture(tmp_path)
    byte_cap = 1 if fault == "byte_cap" else 1_000_000
    if fault == "empty_sources":
        contract["source_sha256"] = {}
    if fault == "escape":
        contract["files_sha256"] = {"../escape": "0" * 64}

    def storage():
        return dict(allocated_bytes=24_900_000_000 if fault == "storage" else 0, logical_bytes=0)

    with pytest.raises((ValueError, RuntimeError)):
        tables.write_tables(
            tmp_path / "output",
            iter(results),
            iter(metadata),
            contract,
            root=tmp_path,
            byte_cap=byte_cap,
            storage_check=storage,
        )
    assert not list((tmp_path / "output").glob("*.npy*"))


def test_exact_provisioned_bytes_do_not_double_charge_receipt_reserve(tmp_path):
    metadata, results, contract = fixture(tmp_path)
    first = write(tmp_path, metadata, results, contract)
    second = tables.write_tables(
        tmp_path / "second",
        iter(results),
        iter(metadata),
        contract,
        root=tmp_path,
        byte_cap=first["planned_bytes"],
        storage_check=lambda: dict(allocated_bytes=0, logical_bytes=0),
    )
    assert second["status"] == "complete"
    assert second["planned_bytes"] == first["table_bytes"] + tables.RECEIPT_RESERVE


def test_published_file_corruption_is_caught_before_complete_receipt(tmp_path, monkeypatch):
    metadata, results, contract = fixture(tmp_path)
    original = tables._sha256

    def corrupt(path):
        if path.name == "delta.npy":
            with path.open("r+b") as handle:
                handle.seek(-1, 2)
                old = handle.read(1)
                handle.seek(-1, 2)
                handle.write(bytes([old[0] ^ 1]))
        return original(path)

    monkeypatch.setattr(tables, "_sha256", corrupt)
    with pytest.raises(ValueError, match="hash"):
        write(tmp_path, metadata, results, contract)
    receipt = json.loads((tmp_path / "output/receipt.json").read_text())
    assert not receipt["complete"] and receipt["status"] == "failed"


def test_interrupted_sequential_publish_leaves_authoritative_incomplete_receipt(tmp_path, monkeypatch):
    from pathlib import Path

    metadata, results, contract = fixture(tmp_path)
    original = Path.rename

    def interrupt(path, target):
        if path.name == "nuisance.npy.partial":
            raise KeyboardInterrupt("synthetic publish interruption")
        return original(path, target)

    monkeypatch.setattr(Path, "rename", interrupt)
    with pytest.raises(KeyboardInterrupt, match="synthetic publish"):
        write(tmp_path, metadata, results, contract)
    receipt = json.loads((tmp_path / "output/receipt.json").read_text())
    assert (tmp_path / "output/delta.npy").is_file() and (tmp_path / "output/nuisance.npy.partial").is_file()
    assert receipt["status"] == "failed" and receipt["complete"] is False
    assert receipt["original_error"] == "synthetic publish interruption"


def test_core_iterator_integrates_without_outcomes_or_full_row_list(tmp_path):
    from cardiotwin.ecgfounder_extraction import IncrementalExtractor

    metadata, _, contract = fixture(tmp_path)
    metadata = [
        BlindIdentity(
            row_index=i, record_id="r", patient_id="p", group_id="g", timestamp=i * 30.0, accepted=i >= 21
        )
        for i in range(23)
    ]
    rec = dict(
        record_id="r",
        patient_id="p",
        group_id="g",
        fs=200,
        duration_seconds=660.0,
        signal_parts=[dict(samples=132000)],
    )
    reads = []

    def reader(record, start, end):
        reads.append((round(start * 200), round(end * 200)))
        values = np.sin(np.arange(round(start * 200), round(end * 200)) / 20.0)
        return np.column_stack([values, values])

    def prepare(values, *, timings):
        return np.full(5000, values.mean(), np.float32)

    def encode(model, chunks):
        return np.broadcast_to(chunks[:, :1], (len(chunks), 1024)).copy()

    core = IncrementalExtractor(None, reader, encoder=encode, preprocessor=prepare)
    contract.update(row_count=len(metadata), identity_sha256=tables.identity_digest(metadata))
    receipt = write(tmp_path, metadata, core.iter_record(iter(metadata), rec), contract)
    assert receipt["rows_written"] == 23 and core.ledger["status"] == "complete"
    assert len(reads) == 21 and reads[-1] == (126000, 132000)
    status = np.load(tmp_path / "output/rows.npy", allow_pickle=False)
    assert status["available"].sum() == 2 and (status["valid_chunks"][:21] == -1).all()


def test_source_hash_interrupt_preserves_its_original_type_and_after_evidence(tmp_path, monkeypatch):
    metadata, results, contract = fixture(tmp_path)
    original = tables._sha256

    def interrupt(path):
        if path.name == "source.py":
            raise KeyboardInterrupt("original source hash interruption")
        return original(path)

    monkeypatch.setattr(tables, "_sha256", interrupt)
    with pytest.raises(KeyboardInterrupt, match="original source hash interruption"):
        write(tmp_path, metadata, results, contract)
    receipt = json.loads((tmp_path / "output/receipt.json").read_text())
    assert receipt["error_type"] == "KeyboardInterrupt"
    assert "source.py" in receipt["hashes_after"]["hash_errors"]
    assert not list((tmp_path / "output").glob("*.npy*"))
