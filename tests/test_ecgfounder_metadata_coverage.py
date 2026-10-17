import hashlib
import importlib.util
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def module():
    path = Path(__file__).resolve().parents[1] / "scripts/run_ecgfounder_metadata_coverage.py"
    spec = importlib.util.spec_from_file_location("metadata_coverage_runner", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class Guard(Mapping):
    def __init__(self, values, allowed):
        self.values, self.allowed = values, set(allowed)

    def __getitem__(self, key):
        assert key in self.allowed, f"Forbidden metadata field: {key}"
        return self.values[key]

    def __iter__(self):
        raise AssertionError("Do not enumerate complete metadata mappings")

    def __len__(self):
        raise AssertionError("Do not inspect complete metadata mappings")


def fixture(duration=630.):
    records, rows = [], []
    for index in range(5):
        identity = dict(record_id=f"r{index}", patient_id=f"p{index}", group_id=f"g{index}")
        records.append(dict(identity, fs=200., duration_seconds=duration,
                            signal_parts=[dict(samples=round(duration * 200))]))
        times = list(np.arange(0, duration + 1e-9, 30))
        if times[-1] < duration:
            times.append(duration)
        rows.extend(dict(identity, timestamp=float(t), duration_seconds=duration, accepted=t >= 600) for t in times)
    names = ("fit", "stop", "calibration", "policy", "outer")
    roles = [dict(fold=f, **{name: [f"g{(i+f)%5}"] for i, name in enumerate(names)}) for f in range(3)]
    return records, rows, roles


def test_guarded_census_counts_unique_rows_once_and_all_fifteen_role_cells():
    runner = module()
    records, rows, roles = fixture()
    protected = [Guard(row, {"record_id", "patient_id", "group_id", "timestamp", "duration_seconds", "accepted"})
                 for row in rows]
    result = runner.count_coverage(protected, records, roles)
    assert result["unique"]["total_rows"] == 110
    assert result["unique"]["A"] == 10 and result["unique"]["M"] == 5 and result["unique"]["N"] == 5
    assert result["unique"]["rejected"] == 100
    assert len(result["cells"]) == 15 and sum(c["total_rows"] for c in result["cells"]) == 330
    assert all(c["A"] == 2 and c["M"] == 1 and c["N"] == 1 for c in result["cells"])
    assert result["ruling"] == "known_masks_rule_out_unchanged_strict_recipe"


def test_exact_half_open_masks_use_actual_file_order_partial_masks_and_endpoints():
    runner = module()
    masks = runner.calibration_intervals([dict(samples=16812345), dict(samples=3000), dict(samples=180000)])
    assert masks == [(0, 6000), (16812345, 16815345), (16815345, 16821345)]
    assert not runner.intersects_known_mask(6000, 126000, masks)
    assert not runner.intersects_known_mask(16692345, 16812345, masks)
    assert runner.intersects_known_mask(16812344, 16812346, masks)
    assert not runner.intersects_known_mask(16821345, 16941345, masks)
    assert runner.calibration_intervals([dict(samples=3000), dict(samples=16812345)])[:2] == [(0, 3000), (3000, 9000)]


@pytest.mark.parametrize("accepted,masked,below", [(20, 1, False), (20, 2, True), (100, 5, False), (101, 6, True)])
def test_integer_95_percent_gate_keeps_original_accepted_denominator(accepted, masked, below):
    result = module().coverage_stats(1000, accepted, masked)
    assert result["N"] == accepted - masked and result["A"] == accepted
    assert result["rejected"] == 1000 - accepted
    assert result["below_95_percent"] is below


def test_zero_accepted_is_uninterpretable_and_never_passes():
    runner = module()
    assert runner.coverage_stats(10, 0, 0)["U"] is None
    records, rows, roles = fixture()
    for row in rows:
        row["accepted"] = False
    result = runner.count_coverage(rows, records, roles)
    assert result["status"] == "failed_zero_accepted" and result["ruling"] == "uninterpretable"


def test_terminal_off_cadence_row_and_float_roundoff_are_counted_exactly():
    runner = module()
    records, rows, roles = fixture(630.005)
    rows[-1]["timestamp"] = np.nextafter(630.005, 0)
    result = runner.count_coverage(rows, records, roles)
    assert result["unique"]["total_rows"] == 115
    assert result["unique"]["A"] == 15 and result["unique"]["M"] == 5 and result["unique"]["N"] == 10


@pytest.mark.parametrize("fault", ["missing", "duplicate", "extra", "reordered", "off_grid", "early_accepted"])
def test_cadence_and_history_support_disagreements_invalidate_without_dropping_rows(fault):
    runner = module()
    records, rows, roles = fixture(630.005)
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows.insert(1, rows[0])
    elif fault == "extra":
        rows.append(rows[-1])
    elif fault == "reordered":
        rows[0], rows[1] = rows[1], rows[0]
    elif fault == "off_grid":
        rows[-1]["timestamp"] = 630.001
    elif fault == "early_accepted":
        rows[0]["accepted"] = True
    with pytest.raises(ValueError):
        runner.count_coverage(rows, records, roles)


@pytest.mark.parametrize("fault", ["overlap", "missing", "unknown", "duplicate", "fold"])
def test_original_role_partition_must_be_disjoint_complete_and_exact(fault):
    runner = module()
    records, rows, roles = fixture()
    if fault == "overlap":
        roles[0]["fit"] += roles[0]["stop"]
    elif fault == "missing":
        roles[0]["outer"] = []
    elif fault == "unknown":
        roles[0]["fit"] = ["outside"]
    elif fault == "duplicate":
        roles[0]["fit"] *= 2
    elif fault == "fold":
        roles[1]["fold"] = 0
    with pytest.raises(ValueError, match="role|fold"):
        runner.count_coverage(rows, records, roles)


@pytest.mark.parametrize("fault", ["fs", "duration", "part_count", "group", "row_duration"])
def test_acquisition_and_row_identity_reconciliation_fails_closed(fault):
    runner = module()
    records, rows, roles = fixture()
    if fault == "fs":
        records[0]["fs"] = 250
    elif fault == "duration":
        records[0]["duration_seconds"] += 0.005
    elif fault == "part_count":
        records[0]["signal_parts"][0]["samples"] = -1
    elif fault == "group":
        rows[0]["group_id"] = "other"
    elif fault == "row_duration":
        rows[0]["duration_seconds"] += 30
    with pytest.raises(ValueError):
        runner.count_coverage(rows, records, roles)


def test_interpretation_reads_only_permitted_metadata_keys():
    runner = module()
    records, _, _ = fixture()
    index_records = [Guard({"record": Guard(record, {"record_id", "patient_id", "fs", "duration_seconds", "signal_parts"})},
                           {"record"}) for record in records]
    identities = [{key: record[key] for key in ("record_id", "patient_id", "group_id")} for record in records]
    index = Guard(dict(records=index_records, allowed_groups=[f"g{i}" for i in range(5)],
                       group_ids={f"p{i}": f"g{i}" for i in range(5)}, row_count=110),
                  {"records", "allowed_groups", "group_ids", "row_count"})
    compact = Guard(dict(records=identities), {"records"})
    assert len(runner.acquisition_records(index, compact)) == 5


def disk_fixture(tmp_path, monkeypatch):
    runner = module()
    records, rows, roles = fixture()
    cycle, cache, audit = [tmp_path / name for name in ("cycle", "cache", "audit")]
    for path in (cycle, cache, audit):
        path.mkdir()
    dtype = [("record", "i4"), ("timestamp", "f8"), ("duration_seconds", "f8"), ("accepted", "i1")]
    metadata = np.array([(int(row["record_id"][1:]), row["timestamp"], row["duration_seconds"], row["accepted"])
                         for row in rows], dtype=dtype)
    np.save(cache / "metadata.npy", metadata)
    index = dict(records=[dict(record={key: record[key] for key in (
        "record_id", "patient_id", "fs", "duration_seconds", "signal_parts")}) for record in records],
        allowed_groups=[f"g{i}" for i in range(5)], group_ids={f"p{i}": f"g{i}" for i in range(5)}, row_count=len(rows))
    (cache / "index.json").write_text(json.dumps(index))
    identities = [{key: record[key] for key in ("record_id", "patient_id", "group_id")} for record in records]
    (cache / "compact.json").write_text(json.dumps({"records": identities}))
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    (audit / "roles.json").write_text(json.dumps({"roles": roles, "cache_index_sha256": digest(cache / "index.json")}))
    declaration = dict(files_sha256={runner.INDEX_KEY: digest(cache / "index.json"), runner.ROLES_KEY: digest(audit / "roles.json")},
                       compact_binding=dict(compact_sha256=digest(cache / "compact.json"),
                                            arrays={"metadata.npy": {"sha256": digest(cache / "metadata.npy")}}))
    (cycle / "declaration-binding.json").write_text(json.dumps(declaration))
    (cycle / "metadata-coverage-spec-20261014.md").write_text("synthetic fixed specification")
    (cycle / "metadata-coverage-spec-clarification-20261014.md").write_text("synthetic fixed clarification")
    monkeypatch.setattr(runner, "SPEC_SHA256", digest(cycle / "metadata-coverage-spec-20261014.md"))
    monkeypatch.setattr(runner, "ADDENDUM_SHA256", digest(cycle / "metadata-coverage-spec-clarification-20261014.md"), raising=False)
    monkeypatch.setattr(runner, "DECLARATION_SHA256", digest(cycle / "declaration-binding.json"))
    monkeypatch.setattr(runner, "EXPECTED_ROWS", 110)
    monkeypatch.setattr(runner, "EXPECTED_PEOPLE", 5)
    monkeypatch.setattr(runner, "EXPECTED_RECORDS", 5)
    return runner, SimpleNamespace(cycle=cycle, cache=cache, audit=audit, output=tmp_path / "output")


def test_metadata_only_disk_run_requires_no_features_pairs_raw_or_weights(tmp_path, monkeypatch):
    runner, args = disk_fixture(tmp_path, monkeypatch)
    result = runner.run(args)
    assert result["status"] == "passed" and result["unique"]["total_rows"] == 110
    assert result["hashes_before"] == result["hashes_after"]
    assert result["new_waveform_reads"] == result["inference_calls"] == result["outcome_fits"] == result["remote_jobs"] == 0
    assert not (args.cache / "features.npy").exists() and not (args.audit / "pairs.jsonl").exists()


def test_source_or_metadata_change_after_counting_invalidates_receipt(tmp_path, monkeypatch):
    runner, args = disk_fixture(tmp_path, monkeypatch)
    original = runner.count_coverage

    def count_and_mutate(*a, **k):
        result = original(*a, **k)
        (args.cache / "compact.json").write_text("changed after census")
        return result

    monkeypatch.setattr(runner, "count_coverage", count_and_mutate)
    with pytest.raises(ValueError, match="changed|hash"):
        runner.run(args)
    receipt = json.loads((args.output / "receipt.json").read_text())
    assert receipt["status"] == "invalid" and receipt["new_waveform_reads"] == 0
    assert set(receipt["hashes_after"]) == set(receipt["hashes_before"])
    assert receipt["hashes_after"][str(args.cache / "compact.json")] != receipt["hashes_before"][str(args.cache / "compact.json")]
    assert receipt["checked_rows"] == 110


@pytest.mark.parametrize("interrupted", [False, True])
@pytest.mark.parametrize("deleted", [False, True])
def test_partial_count_failure_preserves_original_cause_and_best_effort_after_hashes(
        tmp_path, monkeypatch, interrupted, deleted):
    runner, args = disk_fixture(tmp_path, monkeypatch)
    original = runner.count_coverage
    error_type = KeyboardInterrupt if interrupted else ValueError
    original_error = "original partial-count interruption" if interrupted else "original partial-count cadence failure"
    missing = args.cache / "compact.json"

    def fail_during_count(rows, records, roles, *, progress):
        def fail_after_three(completed):
            progress(completed)
            if completed == 3:
                if deleted:
                    missing.unlink()
                raise error_type(original_error)
        return original(rows, records, roles, progress=fail_after_three)

    monkeypatch.setattr(runner, "count_coverage", fail_during_count)
    with pytest.raises(error_type, match=original_error):
        runner.run(args)
    receipt = json.loads((args.output / "receipt.json").read_text())
    assert receipt["status"] == "invalid" and receipt["checked_rows"] == 3
    assert receipt["error_type"] == error_type.__name__ and receipt["error"] == original_error
    assert original_error in (args.output / "original-error.txt").read_text()
    assert set(receipt["hashes_after"]) == set(receipt["hashes_before"])
    if deleted:
        assert receipt["hashes_after"][str(missing)] is None
        assert receipt["hash_errors_after"][str(missing)]["error_type"] == "FileNotFoundError"
        assert set(receipt["hash_errors_after"]) == {str(missing)}
        for path, digest in receipt["hashes_before"].items():
            if path != str(missing):
                assert receipt["hashes_after"][path] == digest
    else:
        assert receipt["hashes_before"] == receipt["hashes_after"]
        assert receipt["hash_errors_after"] == {}


def test_wrong_metadata_identity_fails_before_counting(tmp_path, monkeypatch):
    runner, args = disk_fixture(tmp_path, monkeypatch)
    (args.cache / "metadata.npy").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        runner.run(args)
    receipt = json.loads((args.output / "receipt.json").read_text())
    assert receipt["checked_rows"] == 0 and receipt["status"] == "invalid"


@pytest.mark.parametrize("when", ["before", "after"])
def test_clarification_hash_is_required_before_and_after_counting(tmp_path, monkeypatch, when):
    runner, args = disk_fixture(tmp_path, monkeypatch)
    path = args.cycle / "metadata-coverage-spec-clarification-20261014.md"
    if when == "before":
        path.write_text("changed before counting")
    else:
        original = runner.count_coverage

        def count_and_mutate(*a, **k):
            result = original(*a, **k)
            path.write_text("changed after counting")
            return result

        monkeypatch.setattr(runner, "count_coverage", count_and_mutate)
    with pytest.raises(ValueError, match="hash|changed"):
        runner.run(args)
    receipt = json.loads((args.output / "receipt.json").read_text())
    assert receipt["status"] == "invalid"
    assert receipt["checked_rows"] == (0 if when == "before" else 110)


def test_source_hash_change_is_detected(tmp_path):
    runner = module()
    source = tmp_path / "source.py"
    source.write_text("fixed source")
    bindings = {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}
    runner.verify_hashes(bindings)
    source.write_text("modified source")
    with pytest.raises(ValueError, match="hash|changed"):
        runner.verify_hashes(bindings)


def test_cli_help_needs_no_metadata(capsys):
    with pytest.raises(SystemExit) as exited:
        module().main(["--help"])
    assert exited.value.code == 0
    assert "--cycle" in capsys.readouterr().out
