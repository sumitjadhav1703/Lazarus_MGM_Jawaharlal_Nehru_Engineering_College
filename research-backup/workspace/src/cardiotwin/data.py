"""SHDB-AF ingestion and explicitly synthetic development records."""

import csv
import hashlib
import json
import math
import re
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import wfdb

from .labels import RhythmInterval

SOURCE_URL = "https://physionet.org/files/shdb-af/1.0.1/"
CLINICAL_FIELDS = (
    "age",
    "sex_male",
    "bmi",
    "hypertension",
    "diabetes",
    "heart_failure",
    "prior_stroke",
    "vascular_disease",
    "prior_af_ablation",
    "prior_afl",
    "pacemaker",
    "beta_blocker",
    "antiarrhythmic",
    "anticoagulation",
    "echo_lad",
    "echo_lvef",
)


@dataclass
class RecordInfo:
    record_id: str
    patient_id: str
    path: str
    fs: float
    duration_seconds: float
    annotated: bool
    clinical: dict
    metadata_audit: list
    signal_hash: str | None = None
    start_time: str | None = None
    source: str = "shdb-af"
    signal_parts: list[dict] = dataclass_field(default_factory=list)
    source_exclusions: list[str] = dataclass_field(default_factory=list)

    def __post_init__(self):
        if not (
            math.isfinite(self.fs)
            and self.fs > 0
            and math.isfinite(self.duration_seconds)
            and self.duration_seconds > 0
        ):
            raise ValueError("Record sampling rate and duration must be finite and positive")
        if not self.record_id or not self.patient_id or not self.path:
            raise ValueError("Record, patient and path identifiers must be present")


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _boolean(value):
    if str(value).lower() in {"true", "1", "1.0"}:
        return 1.0
    if str(value).lower() in {"false", "0", "0.0"}:
        return 0.0
    return None


def _date(value):
    try:
        return datetime.fromisoformat(str(value)).date()
    except (ValueError, TypeError):
        return None


def normalize_clinical(row):
    """Allowlist baseline fields; every inclusion/exclusion carries an availability reason."""
    clinical = dict.fromkeys(CLINICAL_FIELDS)
    audit = []
    holter_date = _date(row.get("Date_Holter"))
    sources = {
        "age": "Age_at_Holter",
        "bmi": "BMI",
        "hypertension": "HTN",
        "diabetes": "DM",
        "heart_failure": "CHF",
        "prior_stroke": "Stroke",
        "vascular_disease": "Vascular_Diseases",
        "prior_afl": "Previously_Documented_AFL",
        "pacemaker": "PPM_on_Holter",
        "prior_af_ablation": "Previous_AF_Ablation",
        "sex_male": "Sex",
        "beta_blocker": "Antiarrhythmic_Drug_BB",
        "antiarrhythmic": "Antiarrhythmic_Drug_nonBB",
        "anticoagulation": "Anticoagulation",
        "echo_lad": "Echo_LAD",
        "echo_lvef": "Echo_LVEF",
    }
    for field, source in sources.items():
        raw = row.get(source)
        reason = "baseline_at_holter"
        if field in {"age", "bmi", "echo_lad", "echo_lvef"}:
            value = _number(raw)
        elif field == "sex_male":
            value = {"M": 1.0, "F": 0.0}.get(str(raw).upper())
        elif field in {"beta_blocker", "antiarrhythmic", "anticoagulation"}:
            value = 1.0 if raw and str(raw).strip() else None
            reason = "listed_medication_at_holter" if value else "missing_not_assumed_absent"
        else:
            value = _boolean(raw)
        if field.startswith("echo_"):
            echo_date = _date(row.get("Echo_Date"))
            if not (echo_date and holter_date and echo_date < holter_date):
                value, reason = None, "echo_not_verified_strictly_before_holter"
        if field == "prior_af_ablation" and value == 1:
            procedure_date = _date(row.get("Date_1st_AF_Ablation"))
            if not (procedure_date and holter_date and procedure_date < holter_date):
                value, reason = None, "ablation_not_verified_before_holter"
        if field == "pacemaker" and value == 1:
            ppm_date = _date(row.get("PPM_Date"))
            if not (ppm_date and holter_date and ppm_date < holter_date):
                value, reason = None, "pacemaker_not_verified_before_holter"
        clinical[field] = value
        audit.append(
            {
                "field": field,
                "source": source,
                "available": value is not None,
                "reason": reason if value is not None or reason != "baseline_at_holter" else "missing",
            }
        )
    for source in [
        "AF_Type",
        "Date_of_First_Diagnosis_of_AF_AFL",
        "AF_Duration_Months",
        "PPM_after_Holter",
        "Date_Redo_AF_Ablation",
        "Redo_Detail",
        "Comments",
        "Indication_Holter",
        "PPM_Indication",
        "Ablation1_PVI",
        "Ablation1_CTI",
        "Ablation1_Others",
    ]:
        audit.append(
            {
                "field": None,
                "source": source,
                "available": False,
                "reason": "outcome_post_recording_or_unverified_timing",
            }
        )
    return clinical, audit


def _patient_id(identifier):
    return "p-" + hashlib.sha256(("SHDB-AF:" + str(identifier)).encode()).hexdigest()[:16]


def _start_time(row):
    date = _date(row.get("Date_Holter"))
    if date is None:
        return None
    try:
        # Source dates are shifted relative dates with no timezone; preserve their local clock.
        start = datetime.strptime(row.get("Holter_start_time", ""), "%I:%M %p").time()  # noqa: DTZ007
        return datetime.combine(date, start).isoformat()
    except ValueError:
        return None


def list_records(data_dir):
    data_dir = Path(data_dir)
    manifest = data_dir / "manifest.json"
    if manifest.exists():
        records = []
        for item in json.loads(manifest.read_text())["records"]:
            path = Path(item["path"])
            item = dict(item, path=str(path if path.is_absolute() else data_dir / path))
            item.pop("intervals", None)
            if item.get("signal_parts"):
                item["signal_parts"] = [
                    dict(
                        part,
                        path=str(
                            Path(part["path"])
                            if Path(part["path"]).is_absolute()
                            else data_dir / part["path"]
                        ),
                    )
                    for part in item["signal_parts"]
                ]
            records.append(RecordInfo(**item))
        return records
    if (data_dir / "metadata.csv").exists():
        return _list_iridia(data_dir)
    if (data_dir / "source.json").exists():
        source = json.loads((data_dir / "source.json").read_text()).get("dataset")
        if source == "ltaf":
            return _list_ltaf(data_dir)
    csv_path = data_dir / "AdditionalData.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path}: SHDB-AF metadata or synthetic manifest required")
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = {str(row["Data_ID"]).zfill(3): row for row in csv.DictReader(handle)}
    records = []
    for header in sorted(data_dir.rglob("*.hea")):
        record_id = header.stem
        if record_id not in rows:
            raise ValueError(f"No Data_ID metadata for record {record_id}")
        row = rows[record_id]
        if not row.get("Subject_ID"):
            raise ValueError(f"Missing Subject_ID for record {record_id}")
        info = wfdb.rdheader(str(header.with_suffix("")))
        clinical, audit = normalize_clinical(row)
        annotated = _boolean(row.get("Annotated")) == 1 and header.with_suffix(".atr").exists()
        records.append(
            RecordInfo(
                record_id,
                _patient_id(row["Subject_ID"]),
                str(header.with_suffix("")),
                float(info.fs),
                info.sig_len / info.fs,
                annotated,
                clinical,
                audit,
                start_time=_start_time(row),
            )
        )
    if not records:
        raise ValueError(f"No record headers found in {data_dir}")
    return records


def _list_iridia(data_dir):
    import h5py

    records = []
    exclusions_path = data_dir / "record-exclusions.json"
    exclusions = json.loads(exclusions_path.read_text()) if exclusions_path.exists() else {}
    with (data_dir / "metadata.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    seen = set()
    for row in rows:
        identifier = row["record_id"]
        if not re.fullmatch(r"patient_\d{3}", row["patient_id"]):
            raise ValueError("Invalid IRIDIA independent patient identifier")
        if not re.fullmatch(r"record_\d{3}", identifier) or identifier in seen:
            raise ValueError("Invalid or duplicate IRIDIA record identifier")
        seen.add(identifier)
        folder = data_dir / "records" / identifier
        paths = sorted(folder.glob(f"{identifier}_ecg_*.h5"))
        if not paths:
            continue  # Acquisition may be incomplete; never fabricate missing ECG.
        if len(paths) != int(row["record_files"]):
            raise ValueError(f"Incomplete IRIDIA daily files for {identifier}")
        parts = []
        for index, path in enumerate(paths):
            if path.name != f"{identifier}_ecg_{index:02d}.h5":
                raise ValueError(f"Noncontinuous IRIDIA daily files for {identifier}")
            with h5py.File(path, "r") as handle:
                if len(handle.keys()) != 1:
                    raise ValueError(f"Ambiguous IRIDIA HDF5 signal key: {path}")
                key = next(iter(handle))
                shape = handle[key].shape
                if len(shape) != 2 or shape[1] != 2 or shape[0] <= 0:
                    raise ValueError(f"Expected time-major two-lead IRIDIA ECG: {path}")
                parts.append({"path": str(path), "samples": int(shape[0]), "key": key})
        samples = sum(part["samples"] for part in parts)
        if samples != int(row["record_samples"]):
            raise ValueError(f"IRIDIA metadata/sample count mismatch for {identifier}")
        clinical = dict.fromkeys(CLINICAL_FIELDS)
        clinical.update(
            age=_number(row["patient_age"]), sex_male={"male": 1.0, "female": 0.0}.get(row["patient_sex"])
        )
        audit = [
            {
                "field": name,
                "source": "IRIDIA_baseline_metadata",
                "available": value is not None,
                "reason": "record_baseline_age_sex" if value is not None else "not_provided",
            }
            for name, value in clinical.items()
        ]
        records.append(
            RecordInfo(
                identifier,
                f"iridia-af:{row['patient_id']}",
                str(folder),
                200.0,
                samples / 200,
                (folder / f"{identifier}_ecg_labels.csv").exists(),
                clinical,
                audit,
                start_time=row["record_start_time"],
                source="iridia-af",
                signal_parts=parts,
                source_exclusions=exclusions.get(identifier, []),
            )
        )
    if not records:
        raise ValueError("No complete IRIDIA ECG records acquired")
    return records


def _list_ltaf(data_dir):
    records = []
    for path in sorted(data_dir.glob("*.hea")):
        header = wfdb.rdheader(str(path.with_suffix("")))
        # Public release provides recording IDs, not a verified cross-record subject map.
        records.append(
            RecordInfo(
                f"ltaf-{path.stem}",
                f"ltaf-record:{path.stem}",
                str(path.with_suffix("")),
                float(header.fs),
                header.sig_len / header.fs,
                path.with_suffix(".atr").exists(),
                dict.fromkeys(CLINICAL_FIELDS),
                [],
                source="ltaf",
            )
        )
    if not records:
        raise ValueError("No LTAF headers acquired")
    return records


def signal_paths(record):
    if record.signal_parts:
        return [Path(part["path"]) for part in record.signal_parts]
    path = Path(record.path)
    return [path if path.suffix == ".npz" else path.with_suffix(".dat")]


def signal_available(record):
    return all(path.is_file() for path in signal_paths(record))


def record_signal_hash(record):
    paths = signal_paths(record)
    if len(paths) == 1:
        return file_hash(paths[0])
    digest = hashlib.sha256()
    for path in paths:
        digest.update(bytes.fromhex(file_hash(path)))
    return digest.hexdigest()


@lru_cache(maxsize=2)
def _npz_signal(path, mtime_ns, size):
    # ponytail: two complete synthetic records cached; use memmapped .npy if demo records grow.
    with np.load(path, allow_pickle=False) as contents:
        return contents["signal"]


@contextmanager
def cached_signal_reader():
    """Research-only immutable read session; two HDF5 chunk caches, at most 8 MiB."""
    datasets = OrderedDict()

    def dataset(part):
        import h5py

        path = Path(part["path"])
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size, part["key"])
        if key not in datasets:
            if len(datasets) == 2:
                datasets.popitem(last=False)[1][0].close()
            handle = h5py.File(path, "r", rdcc_nbytes=4 * 1024**2)
            try:
                datasets[key] = (handle, handle[part["key"]])
            except Exception:
                handle.close()
                raise
        datasets.move_to_end(key)
        return datasets[key][1]

    try:
        yield lambda record, start, end: read_signal(record, start, end, _hdf5_datasets=dataset)
    finally:
        for handle, _ in datasets.values():
            handle.close()


def read_signal(record, start_seconds, end_seconds, *, _hdf5_datasets=None):
    if (
        not math.isfinite(start_seconds)
        or not math.isfinite(end_seconds)
        or start_seconds < 0
        or end_seconds <= start_seconds
        or end_seconds > record.duration_seconds
    ):
        raise ValueError("Signal range must satisfy 0 <= start < end <= duration")
    indices = []
    for seconds in (start_seconds, end_seconds):
        scaled = seconds * record.fs
        nearest = round(scaled)
        # Subtracting history from duration can round before multiplication too.
        tolerance = record.fs * math.ulp(record.duration_seconds) + math.ulp(scaled)
        indices.append(nearest if abs(scaled - nearest) <= tolerance else math.ceil(scaled))
    start, end = indices
    if record.source == "iridia-af":
        import h5py

        pieces, offset = [], 0
        for part in record.signal_parts:
            length = part["samples"]
            if start < offset + length and end > offset:
                local_start, local_end = max(0, start - offset), min(length, end - offset)
                if _hdf5_datasets is not None:
                    piece = np.asarray(_hdf5_datasets(part)[local_start:local_end], dtype=float)
                else:
                    with h5py.File(part["path"], "r") as handle:
                        piece = np.asarray(handle[part["key"]][local_start:local_end], dtype=float)
                # Device calibration is acquisition metadata, never an annotation-derived mask.
                if local_start < 6000:
                    piece[: min(local_end, 6000) - local_start] = np.nan
                pieces.append(piece)
            offset += length
        result = np.concatenate(pieces, axis=0)
        if len(result) != end - start:
            raise ValueError("IRIDIA signal coverage incomplete")
        return result
    path = Path(record.path)
    if path.suffix == ".npz":
        stat = path.stat()
        return _npz_signal(str(path), stat.st_mtime_ns, stat.st_size)[start:end].copy()
    return wfdb.rdrecord(str(path), sampfrom=start, sampto=end, physical=True).p_signal


@lru_cache(maxsize=2)
def _rhythm_marks(path, mtime_ns, size):
    annotations = wfdb.rdann(path, "atr")
    return [
        (int(sample), note.strip().rstrip("\x00"))
        for sample, note in zip(annotations.sample, annotations.aux_note)
        if note.strip().startswith("(")
    ]


def read_intervals(record, n_is_other=False):
    if not record.annotated:
        return [RhythmInterval(0, record.duration_seconds, "unknown")]
    path = Path(record.path)
    if record.source == "iridia-af":
        return _iridia_intervals(record)
    if path.suffix == ".npz":
        manifest = json.loads((path.parent / "manifest.json").read_text())
        item = next(i for i in manifest["records"] if i["record_id"] == record.record_id)
        return [RhythmInterval(**i) for i in item["intervals"]]
    annotation_stat = path.with_suffix(".atr").stat()
    marks = _rhythm_marks(str(path), annotation_stat.st_mtime_ns, annotation_stat.st_size)
    changes = []
    other_codes = {"(AFL", "(AT", "(PAT", "(NOD"}
    for sample, note in marks:
        rhythm = "AF" if note == "(AFIB" else "other" if note in other_codes else "unknown"
        if note == "(N" and (n_is_other or record.source == "ltaf"):
            rhythm = "other"
        timestamp = float(sample / record.fs)
        if timestamp < record.duration_seconds:
            changes.append((timestamp, rhythm))
    boundaries = {0.0: "unknown", **dict(changes)}
    starts = sorted(boundaries)
    result = []
    for start, end in zip(starts, starts[1:] + [record.duration_seconds]):
        if end <= start:
            continue
        rhythm = boundaries[start]
        if result and result[-1].end == start and result[-1].rhythm == rhythm:
            previous = result.pop()
            start = previous.start
        result.append(RhythmInterval(start, end, rhythm))
    return result


def _iridia_intervals(record):
    """Reviewed AF ranges use zero-based daily ECG sample coordinates, not vendor RR."""
    offsets = np.r_[0, np.cumsum([part["samples"] for part in record.signal_parts])]
    ranges = []
    path = Path(record.path) / f"{record.record_id}_ecg_labels.csv"
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                times = []
                for side in ("start", "end"):
                    day, sample = int(row[f"{side}_file_index"]), int(row[f"{side}_qrs_index"])
                    if (
                        not 0 <= day < len(record.signal_parts)
                        or not 0 <= sample <= record.signal_parts[day]["samples"]
                    ):
                        raise ValueError("invalid file/sample coordinate")
                    times.append((offsets[day] + sample) / record.fs)
                if times[1] <= times[0]:
                    raise ValueError("nonpositive AF duration")
                ranges.append(tuple(times))
            except (ValueError, KeyError, IndexError) as error:
                raise ValueError(f"Invalid IRIDIA annotation in {path}: {error}") from error
    ranges.sort()
    if any(a[1] > b[0] for a, b in zip(ranges, ranges[1:])):
        raise ValueError("Overlapping IRIDIA annotation ranges")
    # Author's clean_front operation removes 6000 samples from each daily ECG file.
    calibration = [
        (float(offset / record.fs), float(min(offset + 6000, offsets[index + 1]) / record.fs))
        for index, offset in enumerate(offsets[:-1])
    ]
    boundaries = sorted({0.0, record.duration_seconds, *[float(t) for r in ranges + calibration for t in r]})
    result = []
    for start, end in zip(boundaries, boundaries[1:]):
        rhythm = (
            "unknown"
            if any(a <= start < b for a, b in calibration)
            else "AF"
            if any(a <= start < b for a, b in ranges)
            else "other"
        )
        if result and result[-1].rhythm == rhythm:
            start = result.pop().start
        result.append(RhythmInterval(start, end, rhythm))
    return result


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download_shdb(data_dir, record_ids=None, metadata_only=False, source_url=None):
    """Download selected official flat files atomically and verify source SHA256 hashes."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    source_url = (source_url or SOURCE_URL).rstrip("/") + "/"

    def fetch(name, expected=None):
        destination = data_dir / name
        if destination.exists() and (expected is None or file_hash(destination) == expected):
            return
        temporary = destination.with_suffix(destination.suffix + ".partial")
        try:
            with (
                urllib.request.urlopen(source_url + name, timeout=120) as response,
                temporary.open("wb") as out,
            ):
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            if expected and file_hash(temporary) != expected:
                raise ValueError(f"SHA256 mismatch for {name}")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    for name in ["AdditionalData.csv", "RECORDS.txt", "SHA256SUMS.txt", "LICENSE.txt"]:
        fetch(name)
    checksums = {}
    for line in (data_dir / "SHA256SUMS.txt").read_text().splitlines():
        checksum, name = line.split(maxsplit=1)
        checksums[name.lstrip("*")] = checksum
    for name in ["AdditionalData.csv", "RECORDS.txt", "LICENSE.txt"]:
        if name in checksums:
            fetch(name, checksums[name])
    available = (data_dir / "RECORDS.txt").read_text().splitlines()
    selected = available if record_ids is None else [str(record).zfill(3) for record in record_ids]
    if any(not re.fullmatch(r"\d{3}", record) or record not in available for record in selected):
        raise ValueError("Requested record IDs must belong to SHDB-AF RECORDS")
    extensions = ["hea", "atr"] if metadata_only else ["hea", "atr", "dat"]
    downloaded = [
        f"{record}.{extension}"
        for record in selected
        for extension in extensions
        if f"{record}.{extension}" in checksums
    ]
    if metadata_only:
        # Four small metadata requests in parallel; large signal downloads stay sequential.
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda name: fetch(name, checksums[name]), downloaded))
    else:
        for name in downloaded:
            fetch(name, checksums[name])
    return {"source": source_url, "records": selected, "metadata_only": metadata_only, "files": downloaded}


def make_demo(output_dir, config):
    """24 independent synthetic profiles and manufactured ECGs; no clinical validity claimed."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.get("seed", 42))
    fs, duration = 100.0, 4200.0
    manifest = {"synthetic": True, "source": "manufactured development data, not patients", "records": []}
    for index in range(24):
        record_id = f"demo-{index:03d}"
        onset = float(rng.integers(1800, 2700)) if index < 18 else None
        signal = rng.normal(0, 0.009, (int(fs * duration), 2)).astype(np.float32)
        base_rr = float(rng.uniform(0.7, 0.95))
        beat = 0.5
        while beat < duration - 0.1:
            # Deliberately manufactured precursor for exercising forecast plumbing, not AF physiology.
            precursor = onset is not None and onset - 600 <= beat < onset + 300
            rr = base_rr * (0.83 if precursor else 1) + rng.normal(0, 0.13 if precursor else 0.025)
            center = int(beat * fs)
            offsets = np.arange(-8, 9)
            pulse = np.exp(-0.5 * (offsets / 1.6) ** 2)
            if center - 8 >= 0 and center + 8 < len(signal):
                signal[center + offsets, 0] += pulse
                signal[center + offsets, 1] += pulse * 0.83
            beat += max(0.35, rr)
        if index == 0:
            signal[30000:36000] = np.nan  # abstention example away from event history
        path = output_dir / f"{record_id}.npz"
        np.savez_compressed(path, signal=signal, fs=fs)
        clinical = dict.fromkeys(CLINICAL_FIELDS)
        clinical.update(
            age=float(rng.integers(40, 85)),
            sex_male=float(index % 2),
            bmi=float(rng.uniform(20, 31)),
            hypertension=float(index % 3 == 0),
            diabetes=float(index % 4 == 0),
            beta_blocker=float(index % 5 == 0),
        )
        intervals = (
            [
                {"start": 0, "end": onset, "rhythm": "other"},
                {"start": onset, "end": onset + 300, "rhythm": "AF"},
                {"start": onset + 300, "end": duration, "rhythm": "other"},
            ]
            if onset is not None
            else [{"start": 0, "end": duration, "rhythm": "other"}]
        )
        manifest["records"].append(
            {
                "record_id": record_id,
                "patient_id": f"synthetic-{index:03d}",
                "path": path.name,
                "fs": fs,
                "duration_seconds": duration,
                "annotated": True,
                "clinical": clinical,
                "metadata_audit": [
                    {
                        "field": f,
                        "source": "synthetic",
                        "available": v is not None,
                        "reason": "manufactured_baseline",
                    }
                    for f, v in clinical.items()
                ],
                "signal_hash": file_hash(path),
                "start_time": None,
                "intervals": intervals,
            }
        )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    return {"synthetic": True, "records": 24, "patients": 24, "output": str(output_dir)}


def dataset_fingerprint(data_dir):
    """Hash signal, annotation and clinical inputs; independent of absolute folder names."""
    data_dir = Path(data_dir)
    digest = hashlib.sha256()
    files = {
        p
        for p in data_dir.rglob("*")
        if p.is_file() and p.suffix in {".npz", ".hea", ".dat", ".atr", ".h5", ".csv"}
    }
    files.update(p for p in [data_dir / "manifest.json", data_dir / "AdditionalData.csv"] if p.exists())
    files.update(p for p in [data_dir / "source.json"] if p.exists())
    files.update(p for p in [data_dir / "record-exclusions.json"] if p.exists())
    for path in sorted(files):
        digest.update(str(path.relative_to(data_dir)).encode())
        digest.update(bytes.fromhex(file_hash(path)))
    return digest.hexdigest()
