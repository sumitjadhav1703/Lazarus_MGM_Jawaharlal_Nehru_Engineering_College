import hashlib
import importlib
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cardiotwin import ecgfounder_tables as tables
from cardiotwin.precursor import LINEAR_SCHEMA


def compare():
    assert (Path(__file__).parents[1] / "src/cardiotwin/ecgfounder_compare.py").exists(), (
        "Comparison unit absent"
    )
    return importlib.import_module("cardiotwin.ecgfounder_compare")


class Identity(Mapping):
    def __init__(self, **values):
        self.values = values

    def __getitem__(self, key):
        assert key in {"row_index", "record_id", "patient_id", "group_id", "timestamp", "accepted"}
        return self.values[key]

    def __iter__(self):
        raise AssertionError("No outcome-bearing row enumeration")

    def __len__(self):
        raise AssertionError("No outcome-bearing row inspection")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path, monkeypatch, *, bad_schema=False, bad_mapping=False):
    module = compare()
    records = [dict(record_id="r", group_id="g", patient_id="p")]
    names = list(LINEAR_SCHEMA) + ["unrelated_field"]
    compact = dict(
        features=names[:-2] if bad_schema else names,
        records=[dict(record_id="r", group_id="g", patient_id="wrong")] if bad_mapping else records,
    )
    (tmp_path / "compact.json").write_text(json.dumps(compact))
    (tmp_path / "checkpoint.pth").write_bytes(b"tiny synthetic checkpoint, never loaded")
    (tmp_path / "source.py").write_text("synthetic declared source")
    monkeypatch.setattr(module, "CHECKPOINT_SHA256", sha(tmp_path / "checkpoint.pth"))
    monkeypatch.setattr(module, "CHECKPOINT_BYTES", (tmp_path / "checkpoint.pth").stat().st_size)
    metadata = [
        Identity(
            row_index=i,
            record_id="r",
            group_id="g",
            patient_id="p",
            timestamp=0.0 if i == 0 else 600.0 + i * 30,
            accepted=i != 0,
        )
        for i in range(5)
    ]
    results = []
    for i, row in enumerate(metadata):
        item = {
            key: row[key]
            for key in ("row_index", "record_id", "group_id", "patient_id", "timestamp", "accepted")
        }
        available = i in (1, 2, 3)
        vector = np.full(1024, 7.0, np.float32)
        vector[0] = {1: 2.0, 2: 4.0, 3: 8.0}.get(i, 0.0)
        item.update(
            decision_sample=round(row["timestamp"] * 200),
            available=available,
            valid_chunks=60 if available else None if i == 0 else 59,
            rejections={} if available else {tables.REASONS[-1] if i == 0 else "nonfinite_native": 1},
            delta=vector if available else None,
            nuisance=np.arange(6, dtype=np.float32) if available else None,
        )
        results.append(item)
    contract = dict(
        row_count=5,
        identity_sha256=tables.identity_digest(metadata),
        records=records,
        files_sha256={name: sha(tmp_path / name) for name in ("compact.json", "checkpoint.pth")},
        source_sha256={"source.py": sha(tmp_path / "source.py")},
    )
    output = tmp_path / "output"
    tables.write_tables(
        output,
        iter(results),
        iter(metadata),
        contract,
        root=tmp_path,
        byte_cap=1_000_000,
        storage_check=lambda: dict(allocated_bytes=0, logical_bytes=0),
    )
    args = dict(
        output=output,
        metadata_rows=iter(metadata),
        contract=contract,
        root=tmp_path,
        receipt_sha256=sha(output / "receipt.json"),
        compact_path="compact.json",
        checkpoint_path="checkpoint.pth",
        feature_schema=names,
    )
    return module, metadata, args


def reseal(output, *, table=None, update=None):
    path = output / "receipt.json"
    receipt = json.loads(path.read_text())
    if table:
        receipt["tables_sha256"][table] = sha(output / table)
    if update:
        receipt.update(update)
    path.write_text(json.dumps(receipt))
    return sha(path)


def test_verified_reader_preserves_readonly_rows_abstentions_and_repeated_batch_order(tmp_path, monkeypatch):
    module, _, args = fixture(tmp_path, monkeypatch)
    with module.VerifiedTables.open(**args) as data:
        available, delta, nuisance = data.batch([3, 0, 1, 3, 4])
        assert available.tolist() == [True, False, True, True, False]
        assert delta.shape == (5, 1024) and nuisance.shape == (5, 6)
        assert delta[[0, 2, 3], 0].tolist() == [8.0, 2.0, 8.0]
        assert np.isnan(delta[[1, 4]]).all()
        assert not data.rows.flags.writeable and not data.delta.flags.writeable
        assert len(data.rows) == 5


@pytest.mark.parametrize(
    "fault",
    [
        "receipt_hash",
        "incomplete",
        "table_bytes",
        "row_index",
        "sample",
        "available_finite",
        "unavailable_nan",
        "valid_chunks",
        "evaluated",
        "record_index",
        "dtype",
        "shape",
        "reason",
        "source",
        "checkpoint_pin",
    ],
)
def test_corruption_or_incomplete_provenance_blocks_reader_before_joins(tmp_path, monkeypatch, fault):
    module, _, args = fixture(tmp_path, monkeypatch)
    output = args["output"]
    if fault == "receipt_hash":
        args["receipt_sha256"] = "0" * 64
    elif fault == "incomplete":
        args["receipt_sha256"] = reseal(output, update=dict(status="failed", complete=False))
    elif fault == "source":
        (tmp_path / "source.py").write_text("changed source")
    elif fault == "checkpoint_pin":
        monkeypatch.setattr(module, "CHECKPOINT_SHA256", "0" * 64)
    elif fault in {"row_index", "sample", "valid_chunks", "evaluated", "record_index", "reason"}:
        rows = np.load(output / "rows.npy", allow_pickle=False)
        if fault == "row_index":
            rows["row_index"][2] = 1
        elif fault == "sample":
            rows["decision_sample"][1] += 1
        elif fault == "valid_chunks":
            rows["valid_chunks"][1] = 59
        elif fault == "evaluated":
            rows["evaluated"][0] = True
        elif fault == "record_index":
            rows["record_index"][1] = 3
        else:
            rows["reason_counts"][4, 0] = 60
        np.save(output / "rows.npy", rows)
        args["receipt_sha256"] = reseal(output, table="rows.npy")
    else:
        delta = np.load(output / "delta.npy", allow_pickle=False)
        if fault == "available_finite":
            delta[1, 0] = np.nan
        elif fault == "unavailable_nan":
            delta[0, 0] = 0.0
        elif fault == "dtype":
            delta = delta.astype(np.float64)
        elif fault == "shape":
            delta = delta[:, :-1]
        else:
            delta[1, 0] += 1
        np.save(output / "delta.npy", delta)
        if fault != "table_bytes":
            args["receipt_sha256"] = reseal(output, table="delta.npy")
    with pytest.raises((ValueError, OSError)):
        module.VerifiedTables.open(**args)


@pytest.mark.parametrize(
    "fault", ["schema", "mapping", "metadata_missing", "metadata_extra", "metadata_identity"]
)
def test_original_schema_mapping_and_entire_identity_stream_are_checked(tmp_path, monkeypatch, fault):
    module, metadata, args = fixture(
        tmp_path, monkeypatch, bad_schema=fault == "schema", bad_mapping=fault == "mapping"
    )
    if fault == "metadata_missing":
        metadata.pop()
    elif fault == "metadata_extra":
        metadata.append(metadata[-1])
    elif fault == "metadata_identity":
        metadata[2].values["group_id"] = "other"
    args["metadata_rows"] = iter(metadata)
    with pytest.raises(ValueError):
        module.VerifiedTables.open(**args)


def test_mutation_during_identity_scan_invalidates_final_source_and_table_hashes(tmp_path, monkeypatch):
    module, metadata, args = fixture(tmp_path, monkeypatch)

    def changing_rows():
        for i, row in enumerate(metadata):
            if i == 3:
                (tmp_path / "source.py").write_text("mutated during validation")
            yield row

    args["metadata_rows"] = changing_rows()
    with pytest.raises(ValueError):
        module.VerifiedTables.open(**args)


@pytest.mark.parametrize("indices", [[-1], [5], [1.0], [True], list(range(4097))])
def test_join_batches_reject_invalid_indices_and_unbounded_requests(tmp_path, monkeypatch, indices):
    module, _, args = fixture(tmp_path, monkeypatch)
    with module.VerifiedTables.open(**args) as data, pytest.raises(ValueError):
        data.batch(indices)


def pairs():
    return [
        SimpleNamespace(group_id="g", patient_id="p", record_id="r", positive_index=1, negative_index=n)
        for n in (2, 3)
    ]


def timing_rows():
    rows = []
    for i, value in enumerate((999.0, 2.0, 4.0, 8.0, -999.0)):
        row = {name: value for name in LINEAR_SCHEMA}
        row[LINEAR_SCHEMA[1]] = None
        row[LINEAR_SCHEMA[2]] = None if i == 2 else i * 10.0
        rows.append(row)
    return rows


def test_pair_row_masses_accumulate_duplicate_contributions_and_equal_person_mass():
    module = compare()
    values = pairs() + [
        SimpleNamespace(group_id="h", patient_id="q", record_id="s", positive_index=5, negative_index=6)
    ]
    indices, masses, weights = module.pair_row_masses(values)
    assert indices.tolist() == [1, 2, 3, 5, 6]
    np.testing.assert_allclose(weights, [0.75, 0.75, 1.5])
    np.testing.assert_allclose(masses, [0.75, 0.375, 0.375, 0.75, 0.75])


def test_fit_only_shared46_timing_and1070_candidate_preserve_empty_fields_and_weighted_scales(
    tmp_path, monkeypatch
):
    module, _, args = fixture(tmp_path, monkeypatch)
    timing = timing_rows()
    with module.VerifiedTables.open(**args) as data:
        fitted = module.fit_transforms(timing, data, pairs())
        assert fitted.fit_indices.tolist() == [1, 2, 3]
        np.testing.assert_array_equal(fitted.row_masses, [1.0, 0.5, 0.5])
        assert fitted.timing_scaler.mean_[0] == 4.0
        assert fitted.timing_scaler.var_[0] == 6.0
        assert fitted.timing_imputer.statistics_[1] == 0.0
        assert fitted.timing_scaler.mean_[2] == 17.5
        assert fitted.timing_scaler.mean_[25] == 0.25
        assert fitted.delta_scaler.mean_[0] == 4.0 and fitted.delta_scaler.var_[0] == 6.0
        assert fitted.delta_scaler.scale_[1] == 1.0
        features = [timing[i] for i in (3, 1)]
        baseline = fitted.transform(features)
        candidate = fitted.transform(features, data.batch([3, 1])[1])
        assert baseline.shape == (2, 46) and candidate.shape == (2, 1070)
        np.testing.assert_array_equal(candidate[:, :46], baseline)
        np.testing.assert_allclose(candidate[:, 46], [4 / np.sqrt(6), -2 / np.sqrt(6)])
        before = fitted.identity_sha256()
        timing[4][LINEAR_SCHEMA[0]] = 1e100
        fitted.transform([timing[4]])
        assert fitted.identity_sha256() == before
        assert len(fitted.manifest()["timing_schema"]) == 23


@pytest.mark.parametrize(
    "fault", ["unavailable", "identity", "negative", "same_row", "no_pairs", "missing_field", "infinity"]
)
def test_unavailable_or_invalid_fit_inputs_cannot_fit_preprocessing(tmp_path, monkeypatch, fault):
    module, _, args = fixture(tmp_path, monkeypatch)
    fit_pairs, timing = pairs(), timing_rows()
    if fault == "unavailable":
        fit_pairs[0].negative_index = 4
    elif fault == "identity":
        fit_pairs[0].patient_id = "other"
    elif fault == "negative":
        fit_pairs[0].negative_index = -1
    elif fault == "same_row":
        fit_pairs[0].negative_index = 1
    elif fault == "no_pairs":
        fit_pairs = []
    elif fault == "missing_field":
        del timing[1][LINEAR_SCHEMA[0]]
    else:
        timing[1][LINEAR_SCHEMA[0]] = np.inf
    with module.VerifiedTables.open(**args) as data, pytest.raises(ValueError):
        module.fit_transforms(timing, data, fit_pairs)


def test_delta_missingness_cannot_be_imputed_or_append_indicators(tmp_path, monkeypatch):
    module, _, args = fixture(tmp_path, monkeypatch)
    with module.VerifiedTables.open(**args) as data:
        fitted = module.fit_transforms(timing_rows(), data, pairs())
        with pytest.raises(ValueError):
            fitted.transform([timing_rows()[4]], data.batch([4])[1])
        with pytest.raises(ValueError):
            fitted.transform([timing_rows()[1]], np.zeros((1, 1025)))


def test_delta_scaling_keeps_float64_accuracy_in_shared_logistic_design(tmp_path, monkeypatch):
    module, _, args = fixture(tmp_path, monkeypatch)
    with module.VerifiedTables.open(**args) as data:
        fitted = module.fit_transforms(timing_rows(), data, pairs())
        design = fitted.transform([timing_rows()[3]], data.batch([3])[1])
        np.testing.assert_allclose(design[0, 46], 4 / np.sqrt(6), rtol=1e-12, atol=1e-12)


def test_failed_open_closes_already_opened_maps_and_preserves_validation_error(tmp_path, monkeypatch):
    module, metadata, args = fixture(tmp_path, monkeypatch)
    original = np.load
    opened = []

    def track(*a, **k):
        value = original(*a, **k)
        opened.append(value)
        return value

    monkeypatch.setattr(module.np, "load", track)
    metadata[3].values["patient_id"] = "wrong"
    args["metadata_rows"] = iter(metadata)
    with pytest.raises(ValueError, match="mapping|identity"):
        module.VerifiedTables.open(**args)
    assert opened and all(value._mmap.closed for value in opened)


def test_archive_disguised_as_fixed_table_fails_without_replacing_original_validation(tmp_path, monkeypatch):
    module, _, args = fixture(tmp_path, monkeypatch)
    with (args["output"] / "rows.npy").open("wb") as handle:
        np.savez(handle, injected=np.array([1]))
    args["receipt_sha256"] = reseal(args["output"], table="rows.npy")
    with pytest.raises(ValueError, match="table|memmap"):
        module.VerifiedTables.open(**args)


def test_table_change_during_identity_scan_is_rechecked_before_handle_exposure(tmp_path, monkeypatch):
    module, metadata, args = fixture(tmp_path, monkeypatch)

    def changing_rows():
        yield from metadata
        with (args["output"] / "delta.npy").open("r+b") as handle:
            handle.seek(-1, 2)
            value = handle.read(1)
            handle.seek(-1, 2)
            handle.write(bytes([value[0] ^ 1]))

    args["metadata_rows"] = changing_rows()
    with pytest.raises(ValueError, match="changed"):
        module.VerifiedTables.open(**args)
