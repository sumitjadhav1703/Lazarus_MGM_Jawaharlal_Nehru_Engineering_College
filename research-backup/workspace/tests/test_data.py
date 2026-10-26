import numpy as np
import pytest

from cardiotwin.audit import audit_dataset
from cardiotwin.data import RecordInfo, make_demo, normalize_clinical, read_intervals, read_signal
from cardiotwin.labels import RhythmInterval, eligible_onsets, label_at

CONFIG = {
    "seed": 42,
    "target": {
        "history_seconds": 600,
        "gap_seconds": 120,
        "horizon_seconds": 600,
        "cadence_seconds": 30,
        "min_episode_seconds": 30,
        "washout_seconds": 600,
    },
    "quality": {
        "max_missing_fraction": 0.05,
        "max_flatline_fraction": 0.2,
        "min_peaks": 10,
        "max_lead_disagreement": 0.5,
    },
    "audit": {"min_event_subjects": 12, "min_test_event_subjects": 3},
}


def test_forecast_inclusive_boundaries_and_exclusions():
    intervals = [
        RhythmInterval(0, 1800, "other"),
        RhythmInterval(1800, 1900, "AF"),
        RhythmInterval(1900, 4000, "other"),
    ]
    assert eligible_onsets(intervals, 4000, CONFIG) == [1800]
    assert label_at(1200, intervals, 4000, CONFIG) == (1, None)  # upper bound
    assert label_at(1680, intervals, 4000, CONFIG) == (1, None)  # lower bound
    assert label_at(1700, intervals, 4000, CONFIG) == (None, "gap")
    assert label_at(1800, intervals, 4000, CONFIG) == (None, "current_AF")
    assert label_at(1900, intervals, 4000, CONFIG) == (None, "washout")
    assert label_at(100, intervals, 4000, CONFIG) == (None, "history")
    assert label_at(3450, intervals, 4000, CONFIG) == (None, "future")
    assert label_at(1100, intervals, 4000, CONFIG) == (0, None)
    assert label_at(3400, intervals, 4000, CONFIG) == (0, None)


def test_uncovered_unknown_and_short_episode_are_not_negatives():
    unknown = [
        RhythmInterval(0, 1700, "other"),
        RhythmInterval(1700, 1800, "unknown"),
        RhythmInterval(1800, 1900, "AF"),
        RhythmInterval(1900, 4000, "other"),
    ]
    assert label_at(1200, unknown, 4000, CONFIG) == (None, "unknown")
    assert eligible_onsets(unknown, 4000, CONFIG) == []
    assert label_at(1000, [], 4000, CONFIG) == (None, "unknown")
    short = [
        RhythmInterval(0, 1800, "other"),
        RhythmInterval(1800, 1810, "AF"),
        RhythmInterval(1810, 4000, "other"),
    ]
    assert label_at(1200, short, 4000, CONFIG) == (None, "ineligible_event")
    assert eligible_onsets(short, 4000, CONFIG) == []


def test_recurrence_requires_clean_history_and_washout():
    intervals = [
        RhythmInterval(0, 1800, "other"),
        RhythmInterval(1800, 1900, "AF"),
        RhythmInterval(1900, 2200, "other"),
        RhythmInterval(2200, 2300, "AF"),
        RhythmInterval(2300, 4000, "other"),
    ]
    assert eligible_onsets(intervals, 4000, CONFIG) == [1800]


def test_metadata_rejects_future_and_same_day_echo():
    row = {
        "Date_Holter": "2021-03-13",
        "Age_at_Holter": "65",
        "Sex": "M",
        "BMI": "21.2",
        "Echo_Date": "2021-03-13",
        "Echo_LAD": "40",
        "Echo_LVEF": "39",
        "AF_Type": "PAF",
        "Date_of_First_Diagnosis_of_AF_AFL": "2025-06-02",
        "Date_1st_AF_Ablation": "2022-05-01",
        "Previous_AF_Ablation": "True",
        "PPM_Date": "2021-04-01",
        "PPM_on_Holter": "False",
        "Antiarrhythmic_Drug_BB": "bisoprolol",
        "Comments": "future diagnosis",
    }
    clinical, audit = normalize_clinical(row)
    assert clinical["age"] == 65
    assert clinical["beta_blocker"] == 1
    assert clinical["echo_lad"] is None
    assert clinical["prior_af_ablation"] is None
    assert not any("AF_Type" == k or "Comments" == k for k in clinical)
    assert any(a["source"] == "AF_Type" and not a["available"] for a in audit)
    assert any(a["source"] == "Date_of_First_Diagnosis_of_AF_AFL" and not a["available"] for a in audit)
    row["Echo_Date"] = "2021-03-12"
    row["Date_1st_AF_Ablation"] = "2020-05-01"
    clinical, _ = normalize_clinical(row)
    assert clinical["echo_lad"] == 40
    assert clinical["prior_af_ablation"] == 1


def test_wfdb_chunk_and_n_ambiguity(tmp_path):
    import wfdb

    signal = np.column_stack([np.sin(np.arange(1000) / 5), np.cos(np.arange(1000) / 5)])
    wfdb.wrsamp(
        "001",
        fs=100,
        units=["mV", "mV"],
        sig_name=["ECG1", "ECG2"],
        p_signal=signal,
        fmt=["16", "16"],
        write_dir=str(tmp_path),
    )
    wfdb.wrann(
        "001",
        "atr",
        sample=np.array([0, 500]),
        symbol=['"', '"'],
        aux_note=["(N", "(AFIB"],
        write_dir=str(tmp_path),
    )
    record = RecordInfo("001", "patient-a", str(tmp_path / "001"), 100, 10, True, {}, [])
    assert read_signal(record, 1, 2).shape == (100, 2)
    assert [(i.start, i.end, i.rhythm) for i in read_intervals(record)] == [(0, 5, "unknown"), (5, 10, "AF")]
    record.annotated = False
    assert read_intervals(record) == [RhythmInterval(0, 10, "unknown")]
    with pytest.raises(ValueError):
        read_signal(record, -1, 2)


def test_demo_grouping_duplicates_and_quality_gate(tmp_path):
    data_dir, output = tmp_path / "data", tmp_path / "audit"
    make_demo(data_dir, CONFIG)
    report = audit_dataset(data_dir, output, CONFIG)
    assert report["synthetic"] is True
    assert report["feasible"] is True
    assert report["eligible_events"] >= 12
    import json

    manifest = json.loads((data_dir / "manifest.json").read_text())
    original = manifest["records"][0]
    duplicate = dict(original, record_id="copy", patient_id="other-subject")
    manifest["records"].append(duplicate)
    (data_dir / "manifest.json").write_text(json.dumps(manifest))
    repeat = audit_dataset(data_dir, output, CONFIG)
    splits = json.loads((output / "splits.json").read_text())
    assert splits[duplicate["patient_id"]] == splits[original["patient_id"]]
    assert repeat["duplicate_records"] >= 1
    record_path = data_dir / original["path"]
    with np.load(record_path) as contents:
        signal = contents["signal"].copy()
        fs = float(contents["fs"])
    signal[:] = np.nan
    np.savez_compressed(record_path, signal=signal, fs=fs)
    rejected = audit_dataset(data_dir, output, CONFIG)
    assert rejected["eligible_events"] < repeat["eligible_events"]
    assert rejected["exclusions"].get("signal_quality", 0) > 0
    assert audit_dataset(data_dir, output, CONFIG)["eligible_events"] == rejected["eligible_events"]


def test_event_at_upper_boundary_does_not_require_unknown_later_followup():
    intervals = [
        RhythmInterval(0, 1800, "other"),
        RhythmInterval(1800, 1900, "AF"),
        RhythmInterval(1900, 4000, "unknown"),
    ]
    # t=1200 has known labels through 1800 and a confirmed >=30s AF episode.
    assert eligible_onsets(intervals, 4000, CONFIG) == [1800]
    assert label_at(1200, intervals, 4000, CONFIG) == (1, None)


def test_download_checksums_repair_corrupt_files_and_remove_failed_partial(tmp_path, monkeypatch):
    import hashlib

    from cardiotwin import data

    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    files = {
        "AdditionalData.csv": b"Data_ID,Subject_ID\n001,alpha\n",
        "RECORDS.txt": b"001\n",
        "LICENSE.txt": b"Public source license",
        "001.hea": b"header",
        "001.atr": b"annotation",
        "001.dat": b"ECG",
    }
    for name, contents in files.items():
        (source / name).write_bytes(contents)
    (source / "SHA256SUMS.txt").write_text(
        "\n".join(f"{hashlib.sha256(contents).hexdigest()} {name}" for name, contents in files.items())
    )
    monkeypatch.setattr(data, "SOURCE_URL", source.as_uri() + "/")
    data.download_shdb(output, ["001"])
    (output / "001.dat").write_bytes(b"corrupted")
    (output / "AdditionalData.csv").write_bytes(b"corrupted metadata")
    data.download_shdb(output, ["001"])
    assert (output / "001.dat").read_bytes() == b"ECG"
    assert (output / "AdditionalData.csv").read_bytes() == files["AdditionalData.csv"]
    (source / "001.dat").write_bytes(b"bad upstream")
    (output / "001.dat").unlink()
    with pytest.raises(ValueError, match="SHA256"):
        data.download_shdb(output, ["001"])
    assert not (output / "001.dat.partial").exists()
    assert not (output / "001.dat").exists()


def test_audit_reuses_frozen_patient_splits_after_quality_change(tmp_path):
    import json

    # Keep this timing fixture small and explicit; the long demo is exercised separately.
    config = {
        **CONFIG,
        "target": {
            "history_seconds": 20,
            "gap_seconds": 2,
            "horizon_seconds": 10,
            "cadence_seconds": 2,
            "min_episode_seconds": 3,
            "washout_seconds": 20,
        },
        "audit": {"min_event_subjects": 4, "min_test_event_subjects": 1},
    }
    fs = 100
    signal = np.random.default_rng(1).normal(0, 0.009, (10000, 2))
    for center in range(50, 9950, 80):
        offsets = np.arange(-8, 9)
        pulse = np.exp(-0.5 * (offsets / 1.6) ** 2)
        signal[center + offsets] += pulse[:, None]
    records = []
    for i in range(8):
        path = tmp_path / f"{i}.npz"
        np.savez(path, signal=signal * (1 + i / 10), fs=fs)
        records.append(
            {
                "record_id": str(i),
                "patient_id": f"p{i}",
                "path": path.name,
                "fs": fs,
                "duration_seconds": 100,
                "annotated": True,
                "clinical": {},
                "metadata_audit": [],
                "intervals": [
                    {"start": 0, "end": 50, "rhythm": "other"},
                    {"start": 50, "end": 60, "rhythm": "AF"},
                    {"start": 60, "end": 100, "rhythm": "other"},
                ],
            }
        )
    records[1]["start_time"] = "2021-01-01T00:00:00"
    np.savez(tmp_path / "overlap.npz", signal=signal * 4, fs=fs)
    records.append(
        {**records[1], "record_id": "overlap", "path": "overlap.npz", "start_time": "2021-01-01T00:00:50"}
    )
    (tmp_path / "manifest.json").write_text(json.dumps({"synthetic": True, "records": records}))
    out = tmp_path / "audit"
    report = audit_dataset(tmp_path, out, config)
    assert report["eligible_events"] == 8
    assert report["overlapping_records"] == 1
    assert "overlap" not in report["included_record_ids"]
    original = json.loads((out / "splits.json").read_text())
    np.savez(tmp_path / "0.npz", signal=np.full_like(signal, np.nan), fs=fs)
    assert audit_dataset(tmp_path, out, config)["eligible_events"] == 7
    assert json.loads((out / "splits.json").read_text()) == original


@pytest.mark.parametrize(
    "fs,duration", [(0, 100), (-100, 100), (float("nan"), 100), (100, 0), (100, float("inf"))]
)
def test_record_header_rejects_invalid_timebase(fs, duration):
    with pytest.raises(ValueError):
        RecordInfo("record", "patient", "record-path", fs, duration, True, {}, [])
