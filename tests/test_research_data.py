import csv

import numpy as np
import pytest
import wfdb

from cardiotwin.data import list_records, read_intervals, read_signal, signal_available


def iridia_fixture(root):
    h5py = pytest.importorskip("h5py")
    folder = root / "records" / "record_000"
    folder.mkdir(parents=True)
    # Published schema: time-major, two-lead ECG and zero-based file/sample indices.
    for day in range(2):
        with h5py.File(folder / f"record_000_ecg_{day:02d}.h5", "w") as handle:
            handle.create_dataset("ecg", data=np.full((8000, 2), day, dtype=np.float32))
    (folder / "record_000_ecg_labels.csv").write_text(
        "start_datetime,start_file_index,start_qrs_index,end_datetime,end_file_index,end_qrs_index,af_duration,nsr_duration\n"
        "2012-01-01,0,7000,2012-01-02,1,7000,40,35\n"
    )
    with (root / "metadata.csv").open("w", newline="") as out:
        writer = csv.DictWriter(
            out,
            fieldnames=[
                "patient_id",
                "patient_sex",
                "patient_age",
                "record_id",
                "record_start_time",
                "record_files",
                "record_samples",
            ],
        )
        writer.writeheader()
        writer.writerow(
            dict(
                patient_id="patient_000",
                patient_sex="female",
                patient_age=72,
                record_id="record_000",
                record_start_time="2012-01-01T12:00:00",
                record_files=2,
                record_samples=16000,
            )
        )
    return list_records(root)[0]


def test_iridia_continuous_slice_and_calibration_exclusion(tmp_path):
    record = iridia_fixture(tmp_path)
    assert record.source == "iridia-af"
    assert record.patient_id == "iridia-af:patient_000"
    assert record.duration_seconds == 80
    assert signal_available(record)
    signal = read_signal(record, 39, 41)
    np.testing.assert_array_equal(signal[:200], 0)
    assert np.isnan(signal[200:]).all()  # Daily device calibration masked for every model/runtime.
    intervals = read_intervals(record)
    assert [(i.start, i.end, i.rhythm) for i in intervals] == [
        (0, 30, "unknown"),
        (30, 35, "other"),
        (35, 40, "AF"),
        (40, 70, "unknown"),
        (70, 75, "AF"),
        (75, 80, "other"),
    ]
    assert record.clinical["age"] == 72
    assert record.clinical["sex_male"] == 0
    assert record.clinical["hypertension"] is None


def test_iridia_reads_only_requested_samples(tmp_path):
    record = iridia_fixture(tmp_path)
    h5py = pytest.importorskip("h5py")
    before = read_signal(record, 30, 35)
    with h5py.File(record.signal_parts[0]["path"], "r+") as handle:
        handle["ecg"][7000:] = 999
    np.testing.assert_array_equal(before, read_signal(record, 30, 35))


def test_research_read_session_reuses_and_closes_hdf5_handles(tmp_path, monkeypatch):
    from cardiotwin.data import cached_signal_reader

    record = iridia_fixture(tmp_path)
    h5py = pytest.importorskip("h5py")
    original = h5py.File
    opened = []

    def open_file(*args, **kwargs):
        handle = original(*args, **kwargs)
        opened.append(handle)
        return handle

    monkeypatch.setattr(h5py, "File", open_file)
    expected = read_signal(record, 39, 41)
    opened.clear()
    with cached_signal_reader() as reader:
        first = reader(record, 39, 41)
        first[:] = 999
        np.testing.assert_array_equal(reader(record, 39, 41), expected)
        assert len(opened) == 2
    assert all(not handle.id.valid for handle in opened)
    # The normal runtime reader still releases files so acquisition can edit them.
    with original(record.signal_parts[0]["path"], "r+") as handle:
        assert handle["ecg"].shape == (8000, 2)


def test_source_specific_rhythm_mapping_and_unknown_default(tmp_path):
    from cardiotwin.data import RecordInfo

    wfdb.wrann(
        "00",
        "atr",
        sample=np.array([1, 500]),
        symbol=['"', '"'],
        aux_note=["(N", "(AFIB"],
        write_dir=str(tmp_path),
    )
    record = RecordInfo("00", "record-group", str(tmp_path / "00"), 100, 10, True, {}, [])
    assert read_intervals(record)[0].rhythm == "unknown"
    record.source = "ltaf"
    assert read_intervals(record)[1].rhythm == "other"
    assert read_intervals(record)[2].rhythm == "AF"


def test_iridia_bad_annotation_indices_fail_closed(tmp_path):
    record = iridia_fixture(tmp_path)
    annotation = tmp_path / "records/record_000/record_000_ecg_labels.csv"
    annotation.write_text("start_file_index,start_qrs_index,end_file_index,end_qrs_index\n0,7000,2,100\n")
    with pytest.raises(ValueError, match="annotation"):
        read_intervals(record)


def test_source_exclusions_are_explicit_and_fingerprinted(tmp_path):
    import json

    from cardiotwin.data import dataset_fingerprint

    record = iridia_fixture(tmp_path)
    before = dataset_fingerprint(tmp_path)
    assert record.source_exclusions == []
    reason = "independently verified unlocalizable acquisition discontinuity"
    (tmp_path / "record-exclusions.json").write_text(json.dumps({record.record_id: [reason]}))
    assert list_records(tmp_path)[0].source_exclusions == [reason]
    assert dataset_fingerprint(tmp_path) != before


def test_acquisition_rejects_checksum_and_publishes_atomically(tmp_path):
    import hashlib
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "prepare_data", Path(__file__).parents[1] / "scripts/prepare_research_data.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    upstream = tmp_path / "upstream.csv"
    upstream.write_bytes(b"published metadata")
    destination = tmp_path / "metadata.csv"
    checksum = "md5:" + hashlib.md5(upstream.read_bytes()).hexdigest()
    module.fetch(upstream.as_uri(), destination, checksum)
    assert destination.read_bytes() == upstream.read_bytes()
    destination.unlink()
    with pytest.raises(ValueError, match="checksum"):
        module.fetch(upstream.as_uri(), destination, "md5:" + "0" * 32)
    assert not destination.exists()


def test_parallel_archive_resumes_only_journaled_bytes(tmp_path, monkeypatch):
    import hashlib
    import importlib.util
    import io
    import json
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "prepare_ranges", Path(__file__).parents[1] / "scripts/prepare_research_data.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    content = b"public archive content for four ranges"
    destination = tmp_path / "cohort.zip"
    partial = destination.with_name(destination.name + ".partial")
    partial.write_bytes(content[:5])
    # A resumed parallel file is fully allocated; only next=5 is actually complete.
    with partial.open("ab") as handle:
        handle.truncate(len(content))
    checksum = "md5:" + hashlib.md5(content).hexdigest()
    journal = destination.with_name(destination.name + ".ranges.json")
    journal.write_text(
        json.dumps(
            {
                "url": "https://example.test/cohort",
                "size": len(content),
                "checksum": checksum,
                "contiguous_prefix": 5,
                "ranges": [{"start": 5, "end": len(content), "next": 5}],
            }
        )
    )
    requests = []

    def response(request, timeout=120):
        start, end = [int(value) for value in request.get_header("Range").removeprefix("bytes=").split("-")]
        requests.append((start, end))
        result = io.BytesIO(content[start : end + 1])
        result.status = 206
        result.headers = {"Content-Range": f"bytes {start}-{end}/{len(content)}"}
        return result

    monkeypatch.setattr(module, "open_retry", response)
    module.download_ranges("https://example.test/cohort", destination, len(content), checksum)
    assert requests == [(5, len(content) - 1)]
    assert destination.read_bytes() == content
    assert not journal.exists()


def test_structural_audit_is_annotation_only_and_requires_complete_integrity(tmp_path):
    import importlib.util
    import json
    from pathlib import Path

    from cardiotwin.data import file_hash

    record = iridia_fixture(tmp_path)
    metadata = tmp_path / "metadata.csv"
    rows = list(csv.DictReader(metadata.open()))
    rows[0].update(record_timedelta="100", record_seconds="80")
    with metadata.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    spec = importlib.util.spec_from_file_location(
        "structural_audit", Path(__file__).parents[1] / "scripts/prepare_research_data.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = {
        "target": {
            "history_seconds": 10,
            "gap_seconds": 2,
            "horizon_seconds": 20,
            "cadence_seconds": 2.5,
            "min_episode_seconds": 3,
            "washout_seconds": 10,
        }
    }
    (tmp_path / "source.json").write_text(
        json.dumps({"status": "preflight_complete", "archive_md5_verified": False})
    )
    with pytest.raises(ValueError, match="integrity"):
        module.structural_annotation_audit(tmp_path, config)
    (tmp_path / "source.json").write_text(json.dumps({"status": "complete", "archive_md5_verified": True}))
    (tmp_path / "file-checksums.json").write_text(
        json.dumps(
            {
                str(Path(part["path"]).relative_to(tmp_path)): file_hash(part["path"])
                for part in record.signal_parts
            }
        )
    )
    report = module.structural_annotation_audit(tmp_path, config)
    assert report["monitoring_hours"] == 80 / 3600
    assert report["calibration_unknown_hours"] == 60 / 3600
    assert report["independent_explicit_patient_groups"] == 1
    assert report["continuity"]["global_end_duration_mismatch_is_not_exclusion"]
    assert report["record_counts"][0]["source_exclusions"] == []


@pytest.mark.parametrize("source", ["iridia-af", "npz", "wfdb"])
def test_terminal_fractional_sample_boundary_preserves_full_history(tmp_path, source):
    from cardiotwin.data import RecordInfo

    samples, fs = 120013, 200
    duration = samples / fs
    # Both terminal product and duration-minus-history suffer binary rounding.
    assert duration * fs > samples
    values = np.arange(samples, dtype=float)
    signal = np.column_stack([values, values])
    path = tmp_path / "fractional"
    parts = []
    if source == "iridia-af":
        h5py = pytest.importorskip("h5py")
        path = path.with_suffix(".h5")
        with h5py.File(path, "w") as handle:
            handle.create_dataset("ecg", data=signal)
        parts = [{"path": str(path), "samples": samples, "key": "ecg"}]
    elif source == "npz":
        path = path.with_suffix(".npz")
        np.savez(path, signal=signal)
    else:
        wfdb.wrsamp(
            "fractional",
            fs=fs,
            units=["mV", "mV"],
            sig_name=["I", "II"],
            p_signal=signal / samples,
            fmt=["16", "16"],
            write_dir=str(tmp_path),
        )
    record = RecordInfo(
        "fractional",
        "person",
        str(path),
        fs,
        duration,
        False,
        {},
        [],
        source=source,
        signal_parts=parts,
    )
    terminal = read_signal(record, duration - 600, duration)
    assert len(terminal) == 600 * fs
    if source != "wfdb":
        np.testing.assert_array_equal(terminal[-1], signal[-1])
    # A real fractional boundary must retain ceil semantics and never read its future sample.
    sliced = read_signal(record, 30 + 0.25 / fs, 31 + 0.25 / fs)
    assert len(sliced) == fs
    if source != "wfdb":
        np.testing.assert_array_equal(sliced[:, 0], values[6001:6201])
    with pytest.raises(ValueError, match="duration"):
        read_signal(record, duration - 1, duration + 1 / fs)


def test_terminal_rounding_does_not_hide_true_iridia_coverage_gap(tmp_path):
    record = iridia_fixture(tmp_path)
    record.duration_seconds += 1 / record.fs
    with pytest.raises(ValueError, match="coverage incomplete"):
        read_signal(record, 75, record.duration_seconds)
