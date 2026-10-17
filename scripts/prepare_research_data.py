"""Acquire versioned public research cohorts; verified archives and no outcome-based selection."""

import argparse
import binascii
import csv
import hashlib
import io
import json
import os
import shutil
import struct
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

IRIDIA_API = "https://zenodo.org/api/records/8405941"
LTAF_URL = "https://physionet.org/files/ltafdb/1.0.0/"


def open_retry(request, timeout=120):
    for attempt in range(5):
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt == 4:
                raise
            delay = min(60, max(5, int(error.headers.get("Retry-After", 10))))
            time.sleep(delay)


def digest(path, algorithm="sha256"):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, algorithm).hexdigest()


def fetch(url, destination, checksum=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    algorithm, expected = checksum.split(":", 1) if checksum else ("sha256", None)
    if destination.exists() and (expected is None or digest(destination, algorithm) == expected):
        return
    partial = destination.with_name(destination.name + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
    with open_retry(request) as response:
        if offset and response.status != 206:
            raise ValueError("Server did not honor resume Range; retained partial for inspection")
        with partial.open("ab" if offset else "wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
    if expected is not None and digest(partial, algorithm) != expected:
        raise ValueError(f"Published {algorithm} checksum mismatch: {destination.name}")
    partial.replace(destination)


def download_ranges(url, destination, size, checksum, workers=4):
    """Resume a known contiguous prefix, then journal disjoint durable byte-range writes."""
    destination = Path(destination)
    partial = destination.with_name(destination.name + ".partial")
    journal = destination.with_name(destination.name + ".ranges.json")
    algorithm, expected = checksum.split(":", 1)
    if destination.exists() and digest(destination, algorithm) == expected:
        return
    if journal.exists():
        state = json.loads(journal.read_text())
        if (state["url"], state["size"], state["checksum"]) != (url, size, checksum):
            raise ValueError("Range resume identity mismatch")
        if not partial.exists() or partial.stat().st_size != size:
            raise ValueError("Journal exists but partial archive coverage is inconsistent")
    else:
        prefix = partial.stat().st_size if partial.exists() else 0
        if prefix > size:
            raise ValueError("Partial archive exceeds published size")
        step = (size - prefix + workers - 1) // workers
        state = {
            "url": url,
            "size": size,
            "checksum": checksum,
            "contiguous_prefix": prefix,
            "ranges": [
                {"start": start, "end": min(size, start + step), "next": start}
                for start in range(prefix, size, max(1, step))
            ],
        }
        # Journal exists BEFORE preallocation, so file size is never mistaken for progress.
        journal.write_text(json.dumps(state, indent=2))
        with partial.open("ab") as handle:
            handle.truncate(size)
            os.fsync(handle.fileno())
    lock = Lock()

    def persist():
        temporary = journal.with_name(journal.name + ".partial")
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(journal)

    def worker(index):
        span = state["ranges"][index]
        current = span["next"]
        for attempt in range(5):
            if current >= span["end"]:
                return
            request = urllib.request.Request(url, headers={"Range": f"bytes={current}-{span['end'] - 1}"})
            try:
                with open_retry(request) as response, partial.open("r+b", buffering=0) as out:
                    expected_range = f"bytes {current}-{span['end'] - 1}/{size}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                        raise ValueError("Server range response does not match requested archive bytes")
                    checkpoint = current
                    while current < span["end"]:
                        chunk = response.read(min(1024 * 1024, span["end"] - current))
                        if not chunk:
                            raise OSError("Archive range ended prematurely")
                        written = os.pwrite(out.fileno(), chunk, current)
                        if written != len(chunk):
                            raise OSError("Short archive write")
                        current += written
                        if current - checkpoint >= 64 * 1024 * 1024 or current == span["end"]:
                            os.fsync(out.fileno())
                            with lock:
                                span["next"] = current
                                persist()
                            checkpoint = current
                            print(
                                f"range {index}: {current - span['start']}/{span['end'] - span['start']} bytes",
                                flush=True,
                            )
                return
            except (OSError, urllib.error.URLError):
                if attempt == 4:
                    raise
                # Only fsynced journal progress is trusted when retrying.
                current = span["next"]
                time.sleep(min(30, 2**attempt))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(worker, range(len(state["ranges"]))))
    if digest(partial, algorithm) != expected:
        raise ValueError("Parallel archive published checksum mismatch; no data published")
    partial.replace(destination)
    journal.unlink()


class RemoteRangeFile(io.RawIOBase):
    """Seek only the ZIP directory over HTTP before committing storage or reading labels."""

    def __init__(self, url, size):
        self.url, self.size, self.position = url, size, 0

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = (
            offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        )
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.size, self.position + size)
        if end <= self.position:
            return b""
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.position}-{end - 1}"})
        with open_retry(request) as response:
            if response.status != 206:
                raise ValueError("Archive server must support byte ranges for storage preflight")
            result = response.read()
        if len(result) != end - self.position:
            raise ValueError("Incomplete archive byte range")
        self.position = end
        return result


def remote_member_bytes(archive, info):
    # ponytail: one compressed daily member buffered; stream inflate if member sizes grow.
    stream = RemoteRangeFile(archive["links"]["self"], archive["size"])
    stream.seek(info.header_offset)
    raw = stream.read(30 + info.compress_size + 4096)
    header = struct.unpack("<4s5H3I2H", raw[:30])
    if header[0] != b"PK\x03\x04":
        raise ValueError("Invalid ZIP local member header")
    start = 30 + header[-2] + header[-1]
    compressed = raw[start : start + info.compress_size]
    if info.compress_type not in {zipfile.ZIP_DEFLATED, zipfile.ZIP_STORED}:
        raise ValueError("Unsupported published ZIP member compression")
    content = zlib.decompress(compressed, -15) if info.compress_type == zipfile.ZIP_DEFLATED else compressed
    if len(content) != info.file_size or binascii.crc32(content) != info.CRC:
        raise ValueError(f"ZIP member CRC mismatch: {info.filename}")
    return content


def prepare_iridia(
    destination, metadata_only=False, prepare_only=False, max_storage_gb=25, parallel_ranges=False
):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    existing_path = destination / "source.json"
    existing_source = json.loads(existing_path.read_text()) if existing_path.exists() else {}
    with urllib.request.urlopen(IRIDIA_API, timeout=60) as response:
        release = json.load(response)
    (destination / "release-metadata.json").write_text(json.dumps(release, indent=2))
    if release["metadata"]["license"]["id"] != "cc-by-4.0":
        raise ValueError("Unexpected IRIDIA release license")
    files = {item["key"]: item for item in release["files"]}
    metadata = files["iridia-af-metadata-v1.0.1.csv"]
    fetch(metadata["links"]["self"], destination / "metadata.csv", metadata["checksum"])
    if existing_source.get("status") == "complete" and existing_source.get("archive_md5_verified"):
        checksums_path = destination / "file-checksums.json"
        if not checksums_path.exists():
            raise ValueError("Completed IRIDIA acquisition has no extracted integrity manifest")
        checksums = json.loads(checksums_path.read_text())
        if any(
            not (destination / name).is_file() or digest(destination / name) != checksum
            for name, checksum in checksums.items()
        ):
            raise ValueError("Extracted IRIDIA integrity mismatch; completed acquisition not trusted")
        return existing_source
    source = {
        "dataset": "iridia-af",
        "version": "1.0.1",
        "license": "CC-BY-4.0",
        "doi": release["doi"],
        "metadata_checksum": metadata["checksum"],
        "acquired_at": datetime.now(UTC).isoformat(),
        "selection": "all published records, declared before annotations",
        "timebase": "continuous concatenated ECG samples; compare wall-clock discrepancy separately",
        "calibration": "first 6000 samples of each daily file unknown, following author's clean_front",
        "vendor_rr": "not model inputs or reviewed beat ground truth",
    }
    if metadata_only:
        source["status"] = "metadata_only"
        (destination / "source.json").write_text(json.dumps(source, indent=2))
        return source
    archive = files["iridia-af-records-v1.0.1.zip"]
    archive_path = destination / archive["key"]
    if archive_path.exists():
        if digest(archive_path, "md5") != archive["checksum"].split(":")[1]:
            raise ValueError("IRIDIA archive published MD5 mismatch")
        zip_source = archive_path
    else:
        zip_source = RemoteRangeFile(archive["links"]["self"], archive["size"])
    with zipfile.ZipFile(zip_source) as handle:
        inventory = [
            {
                "name": info.filename,
                "size": info.file_size,
                "compressed_size": info.compress_size,
                "crc32": f"{info.CRC:08x}",
            }
            for info in handle.infolist()
        ]
    uncompressed = sum(item["size"] for item in inventory)
    peak = archive["size"] + uncompressed
    (destination / "archive-inventory.json").write_text(json.dumps(inventory, indent=2))
    if peak > max_storage_gb * 1e9:
        raise ValueError(
            f"Full cohort exceeds storage budget: {peak / 1e9:.3f} GB; no label-based subset selected"
        )
    if shutil.disk_usage(destination).free < peak:
        raise ValueError("Insufficient free disk space for archive plus extraction")
    source.update(
        archive_checksum=archive["checksum"],
        archive_bytes=archive["size"],
        uncompressed_bytes=uncompressed,
        peak_acquisition_bytes=peak,
    )
    (destination / "source.json").write_text(json.dumps(dict(source, status="preflight_complete"), indent=2))
    if not prepare_only:
        if parallel_ranges:
            download_ranges(archive["links"]["self"], archive_path, archive["size"], archive["checksum"])
        else:
            fetch(archive["links"]["self"], archive_path, archive["checksum"])
    if not archive_path.exists():
        partial = archive_path.with_name(archive_path.name + ".partial")
        if partial.exists() and partial.stat().st_size == archive["size"]:
            if digest(partial, "md5") != archive["checksum"].split(":")[1]:
                raise ValueError("Downloaded IRIDIA archive published MD5 mismatch")
            partial.replace(archive_path)
        else:
            raise FileNotFoundError("IRIDIA archive is not fully downloaded yet; resume acquisition")
    root = destination / "records"
    root.mkdir(exist_ok=True)
    hashes = {}
    with zipfile.ZipFile(archive_path) as handle:
        for info in handle.infolist():
            if info.is_dir():
                continue
            relative = Path(info.filename)
            if len(relative.parts) != 3 or relative.parts[1].startswith(".") or ".." in relative.parts:
                raise ValueError(f"Unexpected archive member path: {relative}")
            output = root / relative.parts[1] / relative.parts[2]
            output.parent.mkdir(exist_ok=True)
            temporary = output.with_name(output.name + ".partial")
            with handle.open(info) as incoming, temporary.open("wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            temporary.replace(output)  # ZipFile validates member CRC while streaming.
            hashes[str(output.relative_to(destination))] = digest(output)
    (destination / "file-checksums.json").write_text(json.dumps(hashes, indent=2))
    source.update(status="complete", archive_md5_verified=True, extracted_files=len(hashes))
    (destination / "source.json").write_text(json.dumps(source, indent=2))
    # Verified extracted members and their SHA256 persist; discard redundant 10.53 GB ZIP.
    archive_path.unlink()
    return source


def prepare_ltaf(destination, signals=False):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("RECORDS", "SHA256SUMS.txt", "README.md"):
        fetch(LTAF_URL + name, destination / name)
    fetch("https://physionet.org/content/ltafdb/1.0.0/", destination / "LICENSE-and-description.html")
    checksums = {}
    for line in (destination / "SHA256SUMS.txt").read_text().splitlines():
        checksum, name = line.split(maxsplit=1)
        checksums[name.lstrip("*")] = checksum
    records = (destination / "RECORDS").read_text().splitlines()
    files = [
        f"{record}.{ext}"
        for record in records
        for ext in (("hea", "atr", "dat") if signals else ("hea", "atr"))
    ]
    for name in ["RECORDS", "README.md"]:
        if name in checksums:
            fetch(LTAF_URL + name, destination / name, "sha256:" + checksums[name])

    def download_file(name):
        if name not in checksums:
            raise ValueError(f"Missing published LTAF checksum: {name}")
        fetch(LTAF_URL + name, destination / name, "sha256:" + checksums[name])

    if signals:
        for name in files:
            download_file(name)
    else:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(download_file, files))
    source = {
        "dataset": "ltaf",
        "version": "1.0.0",
        "source": LTAF_URL,
        "license": "Open Data Commons Attribution License 1.0",
        "records": records,
        "files": files,
        "signals_downloaded": signals,
        "reviewed_annotations": "atr",
        "subject_identity": "record-level only; cross-record patient map not published",
        "acquired_at": datetime.now(UTC).isoformat(),
        "selection": "all official records",
        "normal_rhythm_mapping": "Reviewed (N is normal rhythm; unmapped notes remain unknown",
    }
    (destination / "source.json").write_text(json.dumps(source, indent=2))
    return source


def prepare_iridia_annotations(destination):
    """Reviewed metadata only; complete archive MD5 must subsequently be checked."""
    destination = Path(destination)
    prepare_iridia(destination, metadata_only=True)
    release = json.loads((destination / "release-metadata.json").read_text())
    archive = next(item for item in release["files"] if item["key"].endswith(".zip"))
    with zipfile.ZipFile(RemoteRangeFile(archive["links"]["self"], archive["size"])) as handle:
        annotations = [info for info in handle.infolist() if info.filename.endswith("_ecg_labels.csv")]

    def acquire(info):
        relative = Path(info.filename)
        output = destination / "records" / relative.parts[1] / relative.parts[2]
        if (
            output.exists()
            and output.stat().st_size == info.file_size
            and binascii.crc32(output.read_bytes()) == info.CRC
        ):
            with output.open(newline="") as incoming:
                events = list(csv.DictReader(incoming))
            return {
                "record_id": relative.parts[1],
                "annotated_episodes": len(events),
                "sha256": digest(output),
                "archive_crc32_verified": True,
            }
        content = remote_member_bytes(archive, info)
        relative = Path(info.filename)
        output = destination / "records" / relative.parts[1] / relative.parts[2]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(content)
        with output.open(newline="") as incoming:
            events = list(csv.DictReader(incoming))
        return {
            "record_id": relative.parts[1],
            "annotated_episodes": len(events),
            "sha256": digest(output),
            "archive_crc32_verified": True,
        }

    with ThreadPoolExecutor(max_workers=4) as pool:
        summaries = list(pool.map(acquire, annotations))
    report = {
        "records": len(summaries),
        "annotated_episodes": sum(item["annotated_episodes"] for item in summaries),
        "record_annotations": summaries,
        "archive_md5_verified": False,
        "status": "annotation_metadata_only_not_signal_feasibility",
        "source": IRIDIA_API,
        "selection": "all published annotations",
    }
    (destination / "annotation-metadata-audit.json").write_text(json.dumps(report, indent=2))
    return report


def continuity_diagnostics(destination):
    """Compare reviewed timestamp offsets; trimmed ends do not imply internal missing data."""
    from cardiotwin.data import list_records, read_intervals

    destination = Path(destination)
    with (destination / "metadata.csv").open(newline="") as incoming:
        metadata = {row["record_id"]: row for row in csv.DictReader(incoming)}
    result = []
    for record in list_records(destination):
        offsets, total = [], 0
        for part in record.signal_parts:
            offsets.append(total / record.fs)
            total += part["samples"]
        row = metadata[record.record_id]
        start_clock = datetime.fromisoformat(row["record_start_time"])
        differences = []
        annotation = Path(record.path) / f"{record.record_id}_ecg_labels.csv"
        with annotation.open(newline="") as incoming:
            for event in csv.DictReader(incoming):
                for side in ("start", "end"):
                    signal_time = (
                        offsets[int(event[f"{side}_file_index"])]
                        + int(event[f"{side}_qrs_index"]) / record.fs
                    )
                    wall_time = (
                        datetime.fromisoformat(event[f"{side}_datetime"]) - start_clock
                    ).total_seconds()
                    differences.append(wall_time - signal_time)
        intervals = read_intervals(record)
        result.append(
            {
                "record_id": record.record_id,
                "patient_id": record.patient_id,
                "daily_samples": [part["samples"] for part in record.signal_parts],
                "sample_duration_seconds": record.duration_seconds,
                "initial_wall_duration_seconds": float(row["record_timedelta"]),
                "reported_corrected_seconds": float(row["record_seconds"]),
                "end_trim_or_clock_difference_seconds": float(row["record_timedelta"])
                - record.duration_seconds,
                "annotation_wall_minus_sample_seconds": differences,
                "annotation_offset_range_seconds": max(differences) - min(differences)
                if differences
                else None,
                "annotation_offset_max_jump_seconds": max(
                    (abs(b - a) for a, b in zip(differences, differences[1:])), default=0
                ),
                "annotated_af_intervals": sum(interval.rhythm == "AF" for interval in intervals),
                "source_exclusions": [],
                "interpretation": "reviewed sample coordinates authoritative; datetime labels snapped to QRS; end trimming documented",
            }
        )
    report = {
        "records": result,
        "source_paper": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10584890/",
        "global_end_duration_mismatch_is_not_exclusion": True,
    }
    (destination / "continuity-diagnostics.json").write_text(json.dumps(report, indent=2))
    return report


def prepare_engineering_probe(destination, identifier="record_000"):
    """Predeclared first numeric record for I/O timing only, never a training subset."""
    destination = Path(destination)
    release = json.loads((destination / "release-metadata.json").read_text())
    archive = next(item for item in release["files"] if item["key"].endswith(".zip"))
    with zipfile.ZipFile(RemoteRangeFile(archive["links"]["self"], archive["size"])) as handle:
        members = [
            info
            for info in handle.infolist()
            if f"/{identifier}/" in info.filename
            and "_ecg_" in info.filename
            and info.filename.endswith(".h5")
        ]
    paths = []
    for info in members:
        content = remote_member_bytes(archive, info)
        output = destination / "records" / identifier / Path(info.filename).name
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".partial")
        temporary.write_bytes(content)
        temporary.replace(output)
        paths.append({"path": str(output), "sha256": digest(output), "member_crc_verified": True})
    report = {
        "record": identifier,
        "purpose": "engineering throughput only; predeclared before performance",
        "archive_md5_verified": False,
        "training_subset": False,
        "files": paths,
    }
    (destination / "engineering-probe.json").write_text(json.dumps(report, indent=2))
    return report


def structural_annotation_audit(destination, config):
    """Complete cohort feasibility on annotation/sample axes; never read ECG values or predictions."""
    from cardiotwin.data import list_records, read_intervals
    from cardiotwin.labels import eligible_onsets, label_at

    destination = Path(destination)
    source = json.loads((destination / "source.json").read_text())
    if source.get("status") != "complete" or source.get("archive_md5_verified") is not True:
        raise ValueError("Full published IRIDIA archive integrity is required for cohort audit")
    records = list_records(destination)
    continuity = continuity_diagnostics(destination)
    checksums = json.loads((destination / "file-checksums.json").read_text())
    identical = defaultdict(list)
    groups = defaultdict(
        lambda: {
            "records": [],
            "eligible_events": 0,
            "eligible_cadence_events": 0,
            "monitoring_seconds": 0.0,
            "eligible_labeled_seconds": 0.0,
        }
    )
    outcomes, exposures = Counter(), Counter()
    record_reports = []
    cadence = config["target"]["cadence_seconds"]
    for record in records:
        signature = tuple(
            checksums[str(Path(part["path"]).relative_to(destination))] for part in record.signal_parts
        )
        identical[signature].append({"record_id": record.record_id, "patient_id": record.patient_id})
        intervals = read_intervals(record)
        onsets = eligible_onsets(intervals, record.duration_seconds, config)
        times = [index * cadence for index in range(int(record.duration_seconds // cadence) + 1)]
        if not times or times[-1] < record.duration_seconds:
            times.append(record.duration_seconds)
        counts, seconds, opportunities = Counter(), Counter(), set()
        for index, t in enumerate(times):
            label, reason = label_at(t, intervals, record.duration_seconds, config)
            category = "positive" if label == 1 else "negative" if label == 0 else reason
            dt = times[index + 1] - t if index + 1 < len(times) else 0
            counts[category] += 1
            seconds[category] += dt
            if label == 1:
                opportunities.update(
                    onset
                    for onset in onsets
                    if config["target"]["gap_seconds"] <= onset - t <= config["target"]["horizon_seconds"]
                )
        eligible_seconds = seconds["positive"] + seconds["negative"]
        group = groups[record.patient_id]
        group["records"].append(record.record_id)
        group["eligible_events"] += len(onsets)
        group["eligible_cadence_events"] += len(opportunities)
        group["monitoring_seconds"] += record.duration_seconds
        group["eligible_labeled_seconds"] += eligible_seconds
        outcomes.update(counts)
        exposures.update(seconds)
        record_reports.append(
            {
                "record_id": record.record_id,
                "group_id": record.patient_id,
                "monitoring_seconds": record.duration_seconds,
                "eligible_onsets": onsets,
                "eligible_cadence_events": len(opportunities),
                "cadence_window_counts": dict(counts),
                "label_exposure_seconds": dict(seconds),
                "source_exclusions": record.source_exclusions,
            }
        )
    duplicate_sets = [rows for rows in identical.values() if len(rows) > 1]
    offset_values = [v for row in continuity["records"] for v in row["annotation_wall_minus_sample_seconds"]]
    report = {
        "dataset": "iridia-af",
        "source": source,
        "target": config["target"],
        "status": "annotation_structural_feasibility_only; signal_quality_and_forecast_performance_not_evaluated",
        "records": len(records),
        "independent_explicit_patient_groups": len(groups),
        "total_samples": sum(part["samples"] for record in records for part in record.signal_parts),
        "monitoring_hours": sum(record.duration_seconds for record in records) / 3600,
        "daily_files": sum(len(record.signal_parts) for record in records),
        "calibration_unknown_hours": sum(
            i.end - i.start for record in records for i in read_intervals(record) if i.rhythm == "unknown"
        )
        / 3600,
        "eligible_annotation_events": sum(group["eligible_events"] for group in groups.values()),
        "eligible_cadence_events": sum(group["eligible_cadence_events"] for group in groups.values()),
        "event_groups": sum(group["eligible_cadence_events"] > 0 for group in groups.values()),
        "eligible_labeled_hours": (exposures["positive"] + exposures["negative"]) / 3600,
        "cadence_window_counts": dict(outcomes),
        "label_exposure_hours": {key: value / 3600 for key, value in exposures.items()},
        "group_counts": dict(sorted(groups.items())),
        "record_counts": record_reports,
        "binary_identical_signal_file_sets": duplicate_sets,
        "continuity": {
            "maximum_annotation_offset_range_seconds": max(
                r["annotation_offset_range_seconds"] or 0 for r in continuity["records"]
            ),
            "maximum_annotation_offset_jump_seconds": max(
                r["annotation_offset_max_jump_seconds"] for r in continuity["records"]
            ),
            "minimum_wall_minus_sample_offset_seconds": min(offset_values),
            "maximum_wall_minus_sample_offset_seconds": max(offset_values),
            "global_end_duration_mismatch_is_not_exclusion": True,
            "interpretation": "Reviewed sample indices authoritative. Constant wall-clock offsets affect absolute timestamp reliability; no global label shift or exclusion applied.",
        },
        "calibration_rule": "first6000samples/day masked NaN in shared signal reader and unknown in offline labels; continuous sample-axis timeline preserved",
        "limitations": [
            "annotation eligibility does not prove usable ECG or model skill",
            "binary identity checks do not establish absence of near-duplicate physiology",
            "calibration, baseline warm-up and signal availability are evaluated separately; no unavailable event removed from sensitivity denominator",
        ],
    }
    path = destination / "structural-annotation-audit.json"
    path.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["iridia", "ltaf"])
    parser.add_argument("--output")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--annotations-only", action="store_true")
    parser.add_argument("--continuity-audit", action="store_true")
    parser.add_argument("--engineering-probe", action="store_true")
    parser.add_argument("--structural-audit", action="store_true")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--signals", action="store_true", help="LTAF signals: only after finalist freeze")
    parser.add_argument("--max-storage-gb", type=float, default=25)
    parser.add_argument("--parallel-ranges", action="store_true")
    args = parser.parse_args()
    if args.dataset == "iridia":
        if args.structural_audit:
            result = structural_annotation_audit(
                args.output or "data/iridia-af", json.loads(Path(args.config).read_text())
            )
        elif args.engineering_probe:
            result = prepare_engineering_probe(args.output or "data/iridia-af")
        elif args.continuity_audit:
            result = continuity_diagnostics(args.output or "data/iridia-af")
        elif args.annotations_only:
            result = prepare_iridia_annotations(args.output or "data/iridia-af")
        else:
            result = prepare_iridia(
                args.output or "data/iridia-af",
                args.metadata_only,
                args.prepare_only,
                args.max_storage_gb,
                args.parallel_ranges,
            )
    else:
        result = prepare_ltaf(args.output or "data/ltaf", args.signals)
    print(json.dumps(result, indent=2))
