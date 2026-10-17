"""Bounded, patient-disjoint research. Held-out signals are never a tuning input.

Cache creation and final-test evaluation are separate explicit operations. This
module reuses the production extractor, probability contracts and alert policy.
"""

import gzip
import hashlib
import json
import multiprocessing
import resource
import shutil
import sys
import time
import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict
from itertools import chain
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import FunctionTransformer, RobustScaler

from .evaluation import evaluate_stream, policy_constraints
from .features import ENGINEERED_ECG_FIELDS, FEATURE_VERSION, _peaks
from .modeling import Predictor, build_pipeline, feature_schema, matrix, raw_score
from .train import _calibrate, _fit_model, patient_macro_log_loss, training_weights

CACHE_VERSION = "continuous-investigation-v1"
MOTIF_FIELDS = ("ecg_short_long_fraction", "ecg_rr_alternation", "ecg_short_pair_clustering")


def _write_predictions(path, rows):
    """Compress research timelines without changing the public evaluation format."""
    path = Path(path)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_compact_prediction(row), allow_nan=False) + "\n")
    return str(path)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def _fresh(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("Research outputs require a fresh empty directory")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _group(row):
    return row.get("group_id", row["patient_id"])


def _usable(rows):
    return [r for r in rows if r["accepted"] and r["label"] in (0, 1)]


def _subset(rows, groups):
    groups = set(groups)
    return [r for r in rows if _group(r) in groups]


def _events(events, rows):
    keys = {(r["patient_id"], r["record_id"]) for r in rows}
    return [e for e in events if (e["patient_id"], e["record_id"]) in keys]


def _record_contract(record):
    """Content identity without machine-specific absolute roots."""
    from .data import file_hash, record_signal_hash, signal_available

    fields = (
        "record_id",
        "patient_id",
        "fs",
        "duration_seconds",
        "annotated",
        "clinical",
        "start_time",
        "source",
        "source_exclusions",
    )
    metadata = {field: getattr(record, field, None) for field in fields}
    metadata["parts"] = [
        {k: v for k, v in part.items() if k != "path"} for part in getattr(record, "signal_parts", [])
    ]
    if hasattr(record, "path"):
        metadata["signal_sha256"] = record_signal_hash(record) if signal_available(record) else None
        annotation = Path(record.path).with_suffix(".atr")
        metadata["annotation_sha256"] = file_hash(annotation) if annotation.is_file() else None
        if getattr(record, "source", None) == "iridia-af":
            annotations = Path(record.path) / f"{record.record_id}_ecg_labels.csv"
            metadata["annotation_sha256"] = file_hash(annotations) if annotations.is_file() else None
        if Path(record.path).suffix == ".npz":
            manifest = Path(record.path).parent / "manifest.json"
            metadata["annotation_sha256"] = file_hash(manifest)
    return metadata


def _extraction_contract(config):
    return {key: config[key] for key in ("feature_version", "target", "quality", "baseline")}


def require_complete_acquisition(records):
    """A metadata/first-file probe cannot silently become the primary cohort."""
    from .data import signal_available

    checked = set()
    for record in records:
        if getattr(record, "source", None) != "iridia-af":
            continue
        directory = next(
            (
                parent
                for parent in [Path(record.path), *Path(record.path).parents]
                if (parent / "source.json").is_file()
            ),
            None,
        )
        if directory is None:
            raise ValueError("Primary IRIDIA cohort requires verified complete acquisition provenance")
        if directory not in checked:
            source = json.loads((directory / "source.json").read_text())
            if source.get("status") != "complete" or source.get("archive_md5_verified") is not True:
                raise ValueError("Incomplete/unverified IRIDIA acquisition cannot enter primary training")
            if source.get("archive_checksum") != "md5:81d58d21c9835a601cd9eb6c68bf9feb":
                raise ValueError("IRIDIA acquisition checksum differs from the declared public release")
            checked.add(directory)
        if not signal_available(record):
            raise ValueError("An included primary recording has missing acquired signal parts")


def make_primary_split(records, events, group_ids=None, seed=42, predeclared_development_groups=()):
    """Metadata-only allocation; approximately 20% of each event-status stratum.

    Membership is patient-level and includes repeated recordings. Record-level
    names must never be passed as independent groups without provenance evidence.
    """
    require_complete_acquisition(records)
    group_ids = group_ids or {r.patient_id: r.patient_id for r in records}
    if set(group_ids) != {r.patient_id for r in records}:
        raise ValueError("Every included patient needs one audited independent group")
    groups = sorted(set(group_ids.values()))
    if len(groups) < 12:
        raise ValueError("At least 12 independent groups required for primary and nested development splits")
    forced_development = set(predeclared_development_groups)
    if not forced_development <= set(groups):
        raise ValueError("Predeclared development groups must belong to the audited cohort")
    positive = {group_ids[e["patient_id"]] for e in events}
    rng = np.random.default_rng(seed)
    test = set()
    for status in (False, True):
        total = [g for g in groups if (g in positive) == status]
        stratum = [g for g in total if g not in forced_development]
        if not stratum:
            continue
        rng.shuffle(stratum)
        count = min(len(stratum), max(1, round(len(total) * 0.2)))
        if status and len(total) - count < 9:
            raise ValueError(
                "At least nine development event groups are required after reserving primary test"
            )
        test.update(stratum[:count])
    split = {
        "seed": seed,
        "group_ids": group_ids,
        "development_groups": sorted(set(groups) - test),
        "test_groups": sorted(test),
        "test_fraction": len(test) / len(groups),
        "event_groups": sorted(positive),
        "record_ids": sorted(r.record_id for r in records),
        "independent_unit": "audited_group_id",
        "predeclared_development_groups": sorted(forced_development),
        "record_contracts": {r.record_id: _record_contract(r) for r in records},
    }
    split["sha256"] = _digest(split)
    return split


def rr_motifs(signal, fs):
    """Descriptive normalized interval patterns, not diagnosed premature beats.

    Adjacency retains invalid intervals; no pair is manufactured across a gap.
    """
    lead = np.asarray(signal)[:, 0]
    peaks = _peaks(lead, fs)
    rr = np.diff(peaks)
    output = dict.fromkeys(MOTIF_FIELDS)
    if len(rr) < 10:
        return output
    samples = np.minimum(np.rint(peaks * fs).astype(int), len(lead) - 1)
    missing = np.r_[0, np.cumsum(~np.isfinite(lead))]
    valid = (rr >= 0.25) & (rr <= 2.5)
    valid &= missing[samples[1:] + 1] == missing[samples[:-1]]
    if valid.mean() < 0.8 or valid.sum() < 10:
        return output
    median = np.median(rr[valid])
    normalized = rr / median
    adjacent = valid[:-1] & valid[1:]
    if not adjacent.any():
        return output
    short, long = normalized < 0.8, normalized > 1.2
    motif = short[:-1] & long[1:]
    output[MOTIF_FIELDS[0]] = float(motif[adjacent].mean())
    output[MOTIF_FIELDS[1]] = float((((normalized[:-1] - 1) * (normalized[1:] - 1)) < 0)[adjacent].mean())
    triples = valid[:-2] & valid[1:-1] & valid[2:]
    if triples.any():
        output[MOTIF_FIELDS[2]] = float((short[:-2] & short[2:])[triples].mean())
    return output


def _cache_one_record(record, config, group, path, deadline, motifs):
    """Each process owns an isolated chronological baseline for one recording."""
    from threadpoolctl import threadpool_limits

    from .data import cached_signal_reader, file_hash, read_intervals
    from .labels import eligible_onsets
    from .train import iter_record_rows

    count, fields = 0, set()
    began = time.monotonic()
    events = []
    path = Path(path)
    checkpoint = path.with_suffix(".checkpoint.json")
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        if saved["sha256"] != _digest({k: v for k, v in saved.items() if k != "sha256"}):
            raise ValueError("Record checkpoint metadata checksum mismatch")
        if not path.exists() or file_hash(path) != saved["rows_sha256"]:
            raise ValueError("Record checkpoint rows checksum mismatch")
        if saved["result"]["input_contract"] != _record_contract(record):
            raise ValueError("Record checkpoint acquisition identity changed")
        return saved["result"]
    with (
        threadpool_limits(limits=1),
        cached_signal_reader() as read_signal,
        path.with_suffix(".partial").open("w") as stream,
    ):
        intervals = read_intervals(record, n_is_other=config["audit"].get("n_rhythm_is_other", False))
        events = [
            {
                "patient_id": record.patient_id,
                "record_id": record.record_id,
                "onset": onset,
                "group_id": group,
            }
            for onset in eligible_onsets(intervals, record.duration_seconds, config)
        ]
        for row in iter_record_rows(record, config, reader=read_signal):
            if time.monotonic() > deadline:
                raise TimeoutError(
                    "Extraction budget exhausted; partial cache is not eligible for experiments"
                )
            row["group_id"] = group
            if motifs and row["timestamp"] >= config["target"]["history_seconds"]:
                signal = read_signal(
                    record, row["timestamp"] - config["target"]["history_seconds"], row["timestamp"]
                )
                row["features"].update(rr_motifs(signal, record.fs))
            fields.update(row["features"])
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            count += 1
    path.with_suffix(".partial").replace(path)
    if time.monotonic() > deadline:
        raise TimeoutError("Extraction budget exhausted before recording identity verification")
    result = {
        "record": asdict(record),
        "rows": count,
        "input_contract": _record_contract(record),
        "events": events,
        "feature_fields": sorted(fields),
        "wall_seconds": time.monotonic() - began,
        "peak_worker_rss_bytes": _peak_memory(),
    }
    saved = {"result": result, "rows_sha256": file_hash(path)}
    saved["sha256"] = _digest(saved)
    _write(checkpoint, saved)
    return result


def cache_records(
    records, config, group_ids, output, allowed_groups, max_hours=8, motifs=False, workers=2, *, resume=False
):
    """Bounded record-parallel extraction; deterministic chronological output.

    Only explicitly allowed groups are opened. Worker baselines are isolated;
    parent merges records in supplied order rather than completion order. Missing
    records, deadline exhaustion or worker errors never produce an eligible index.
    """
    from .data import file_hash

    if not isinstance(workers, int) or not 1 <= workers <= 4:
        raise ValueError("Cache workers must be between one and four")
    require_complete_acquisition(records)
    allowed = set(allowed_groups)
    selected = [r for r in records if group_ids[r.patient_id] in allowed]
    if not selected or not allowed <= set(group_ids.values()):
        raise ValueError("Cache requires nonempty audited allowed groups")
    recipe = {
        "cache_version": CACHE_VERSION,
        "config": config,
        "group_ids": group_ids,
        "allowed_groups": sorted(allowed),
        "motifs": motifs,
        "records": [_record_contract(record) for record in selected],
    }
    output = Path(output)
    if resume:
        if (output / "index.json").exists():
            raise ValueError("A completed cache cannot be resumed or overwritten")
        if json.loads((output / "extraction-start.json").read_text()) != recipe:
            raise ValueError("Resumed cache recipe/acquisition identity changed")
    else:
        output = _fresh(output)
        _write(output / "extraction-start.json", recipe)
    started, deadline = time.monotonic(), time.monotonic() + max_hours * 3600
    part_paths = [output / f"record-{number:04d}.jsonl" for number in range(len(selected))]
    arguments = [
        (record, config, group_ids[record.patient_id], str(path), deadline, motifs)
        for record, path in zip(selected, part_paths, strict=True)
    ]
    indexed, events, feature_fields, row_count = [], [], set(), 0
    pool = None
    if workers > 1:
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
        pending = [pool.submit(_cache_one_record, *argument) for argument in arguments]
    else:
        pending = None
    try:
        with (output / "rows.jsonl.partial").open("w") as stream:
            for number, argument in enumerate(arguments):
                result = pending[number].result() if pending is not None else _cache_one_record(*argument)
                if time.monotonic() > deadline:
                    raise TimeoutError("Extraction budget exhausted during deterministic merge")
                with part_paths[number].open() as part:
                    shutil.copyfileobj(part, stream, length=1024 * 1024)
                events.extend(result.pop("events"))
                feature_fields.update(result.pop("feature_fields"))
                indexed.append(result)
                row_count += result["rows"]
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    (output / "rows.jsonl.partial").replace(output / "rows.jsonl")
    index = {
        "cache_version": CACHE_VERSION,
        "feature_version": FEATURE_VERSION,
        "config": config,
        "config_sha256": _digest(config),
        "group_ids": group_ids,
        "allowed_groups": sorted(allowed),
        "records": indexed,
        "events": events,
        "row_count": row_count,
        "rows_sha256": file_hash(output / "rows.jsonl"),
        "wall_seconds": time.monotonic() - started,
        "workers": workers,
        "motifs": motifs,
        "feature_fields": sorted(feature_fields),
        "input_sha256": _digest([r["input_contract"] for r in indexed]),
    }
    index["sha256"] = _digest(index)
    _write(output / "index.json", index)
    for path in part_paths:
        path.unlink()
        path.with_suffix(".checkpoint.json").unlink()
    return index


class _FeatureView(Mapping):
    __slots__ = ("values", "names", "positions")

    def __init__(self, values, names, positions):
        self.values, self.names, self.positions = values, names, positions

    def __getitem__(self, key):
        value = self.values[self.positions[key]]
        return None if np.isnan(value) else float(value)

    def __iter__(self):
        return iter(self.names)

    def __len__(self):
        return len(self.names)


class _RowView(Mapping):
    __slots__ = ("cache", "index")
    _keys = (
        "patient_id",
        "group_id",
        "record_id",
        "timestamp",
        "duration_seconds",
        "features",
        "accepted",
        "reason",
        "label",
        "label_reason",
    )

    def __init__(self, cache, index):
        self.cache, self.index = cache, index

    def __getitem__(self, key):
        meta = self.cache.metadata[self.index]
        if key == "features":
            return _FeatureView(self.cache.features[self.index], self.cache.names, self.cache.positions)
        if key in ("patient_id", "group_id", "record_id"):
            return self.cache.records[int(meta["record"])][key]
        if key in ("reason", "label_reason"):
            return self.cache.categories[key][int(meta[key])]
        if key == "label":
            return None if meta[key] == -1 else int(meta[key])
        if key == "accepted":
            return bool(meta[key])
        if key in ("timestamp", "duration_seconds"):
            return float(meta[key])
        raise KeyError(key)

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)


class CompactRows(Sequence):
    """Zero-copy numeric features, small mapping views, no million Python dictionaries."""

    def __init__(self, path, contract):
        self.features = np.load(path / "features.npy", mmap_mode="r")
        self.metadata = np.load(path / "metadata.npy", mmap_mode="r")
        self.names, self.records, self.categories = (
            contract["features"],
            contract["records"],
            contract["categories"],
        )
        self.positions = {name: i for i, name in enumerate(self.names)}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return _RowView(self, index)


def _compact_cache(path, index):
    from .data import file_hash

    contract_path = path / "compact.json"
    if contract_path.exists():
        contract = json.loads(contract_path.read_text())
        if contract["rows_sha256"] != index["rows_sha256"]:
            raise ValueError("Compact cache belongs to different chronological rows")
        if index.get("feature_fields") is not None and contract["features"] != index["feature_fields"]:
            raise ValueError("Compact cache feature schema differs from index")
        for name in ("features.npy", "metadata.npy"):
            if file_hash(path / name) != contract["checksums"][name]:
                raise ValueError("Compact cache checksum mismatch")
        return CompactRows(path, contract)
    names = index.get("feature_fields")
    if names is None:
        keys = set()
        with (path / "rows.jsonl").open() as handle:
            for line in handle:
                keys.update(json.loads(line)["features"])
        names = sorted(keys)
    count = index["row_count"]
    features = np.lib.format.open_memmap(
        path / "features.npy", mode="w+", dtype="float64", shape=(count, len(names))
    )
    features[:] = np.nan
    dtype = [
        ("record", "int32"),
        ("timestamp", "float64"),
        ("duration_seconds", "float64"),
        ("accepted", "int8"),
        ("label", "int8"),
        ("reason", "int16"),
        ("label_reason", "int16"),
    ]
    metadata = np.lib.format.open_memmap(path / "metadata.npy", mode="w+", dtype=dtype, shape=(count,))
    positions = {name: i for i, name in enumerate(names)}
    records, record_codes, categories = [], {}, {"reason": [None], "label_reason": [None]}
    line_count = 0
    with (path / "rows.jsonl").open() as handle:
        for number, line in enumerate(handle):
            if number >= count:
                raise ValueError("Cached timeline has extra rows")
            row = json.loads(line)
            group = _group(row)
            if group not in index["allowed_groups"] or index["group_ids"][row["patient_id"]] != group:
                raise ValueError("A cached row violates audited group membership")
            key = (row["patient_id"], row["record_id"])
            if key not in record_codes:
                record_codes[key] = len(records)
                records.append(
                    {"patient_id": row["patient_id"], "record_id": row["record_id"], "group_id": group}
                )
            meta = metadata[number]
            meta["record"] = record_codes[key]
            for field in ("timestamp", "duration_seconds", "accepted"):
                meta[field] = row[field]
            meta["label"] = -1 if row["label"] is None else row["label"]
            for field in ("reason", "label_reason"):
                value = row.get(field)
                if value not in categories[field]:
                    categories[field].append(value)
                meta[field] = categories[field].index(value)
            for field, value in row["features"].items():
                features[number, positions[field]] = np.nan if value is None else value
            line_count += 1
    if line_count != count:
        raise ValueError("Cached timeline is incomplete")
    features.flush()
    metadata.flush()
    contract = {
        "rows_sha256": index["rows_sha256"],
        "features": names,
        "records": records,
        "categories": categories,
        "checksums": {name: file_hash(path / name) for name in ("features.npy", "metadata.npy")},
    }
    _write(contract_path, contract)
    return CompactRows(path, contract)


def read_cache(path):
    from .data import file_hash

    path = Path(path)
    index = json.loads((path / "index.json").read_text())
    if index["cache_version"] != CACHE_VERSION or index["feature_version"] != FEATURE_VERSION:
        raise ValueError("Cache representation version differs")
    if (
        file_hash(path / "rows.jsonl") != index["rows_sha256"]
        or _digest(index["config"]) != index["config_sha256"]
    ):
        raise ValueError("Cached rows/config checksum mismatch")
    if _digest({k: v for k, v in index.items() if k != "sha256"}) != index.get("sha256"):
        raise ValueError("Cache index metadata checksum mismatch; onset/group identities are immutable")
    rows = _compact_cache(path, index)
    return rows, index["events"], index


def validate_live_inputs(records, index):
    """Bind on-demand sequence reads to the feature-cache acquisition, including labels."""
    from .progress import Progress

    with Progress("source_hash_validation", total=len(records)) as progress:
        require_complete_acquisition(records)
        expected = {item["record"]["record_id"]: item["input_contract"] for item in index["records"]}
        if len(expected) != len(index["records"]) or {record.record_id for record in records} != set(expected):
            raise ValueError("Live recordings differ from the immutable development cache membership")
        for completed, record in enumerate(records, 1):
            if _record_contract(record) != expected[record.record_id]:
                raise ValueError("Live source content/labels differ from the cached acquisition contract")
            if index["group_ids"].get(record.patient_id) not in index["allowed_groups"]:
                raise ValueError("Live source patient lies outside the development cache")
            progress.update(completed=completed)
    return _digest([expected[item["record"]["record_id"]] for item in index["records"]])


def _partitions(rows, seed):
    """70/15/15 within the supplied outer-fitting patients, never window splitting."""
    usable = _usable(rows)
    groups = sorted({_group(r) for r in rows})
    classes = {}
    for row in usable:
        classes.setdefault(_group(row), set()).add(row["label"])
    if len(groups) < 7:
        raise ValueError("At least seven outer-fitting groups required for fit/calibration/policy")
    rng = np.random.default_rng(seed)
    count = max(1, round(len(groups) * 0.15))
    for _ in range(200):
        shuffled = rng.permutation(groups).tolist()
        partitions = [shuffled[2 * count :], shuffled[:count], shuffled[count : 2 * count]]
        if all(
            set().union(*(classes.get(group, set()) for group in partition)) == {0, 1}
            for partition in partitions
        ):
            return tuple(partitions)
    raise ValueError("Cannot form disjoint fit/calibration/policy groups with both classes")


def feature_groups(rows):
    keys = [r["features"] for r in rows]
    if keys and isinstance(keys[0], _FeatureView):
        keys = keys[:1]
    core = feature_schema(keys, "ecg")
    engineered = feature_schema(keys, "ecg_engineered")
    personal = [k for k in feature_schema(keys, "personal_engineered") if not k.startswith("clinical_")]
    clinical = [
        k
        for k in feature_schema(keys, "clinical")
        if k in {"clinical_age", "clinical_sex_male", "clinical_age_available", "clinical_sex_male_available"}
    ]
    availability = [k for k in clinical if k.endswith("_available")]
    dispersion = {"ecg_rr_median", "ecg_rr_iqr", "ecg_rr_mad", "ecg_rr_cv", "ecg_sd1", "ecg_sd2"}
    dynamics = {k for k in ENGINEERED_ECG_FIELDS if "60s" in k or "300s" in k}
    nonlinear = {"ecg_rr_lag1_corr", "ecg_rr_sample_entropy"}
    found = set().union(*(row.keys() for row in keys))
    return {
        "core": core,
        "core_patient_only": core,
        "core_window_class": core,
        "core_no_missing": core,
        "core_robustscale": core,
        "core_cadence30": core,
        "current_recipe": core,
        "dispersion": sorted(set(core) | (found & dispersion)),
        "dynamics": sorted(set(core) | (found & dynamics)),
        "nonlinear": sorted(set(core) | (found & nonlinear)),
        "engineered": engineered,
        "personal": personal,
        "clinical": clinical,
        "availability": availability,
        "combined": sorted(set(engineered + clinical)),
        "availability_control": sorted(set(engineered + availability)),
        "motifs": sorted(set(engineered) | (found & set(MOTIF_FIELDS))),
    }


def _parameters(family, seed, trials):
    rng = np.random.default_rng(seed)
    if family == "logistic":
        pool = [{"C": c} for c in (0.01, 0.1, 1, 10, 100, 1000)]
    elif family == "elasticnet":
        pool = [{"C": c, "l1_ratio": ratio} for c in (0.01, 0.1, 1, 10) for ratio in (0.25, 0.5, 0.75)]
    else:
        pool = [
            {"max_depth": depth, "reg_lambda": reg, "n_estimators": trees}
            for depth in (2, 3)
            for reg in (3, 10)
            for trees in (40, 80, 120)
        ]
    rng.shuffle(pool)
    return pool[: min(trials, 6)]


def _values_only(values):
    return values[:, : values.shape[1] // 2]


def _fit(rows, family, schema, parameters, seed, weighting="patient_class", precomputed_x=None):
    usable = _usable(rows)
    y = np.asarray([r["label"] for r in usable])
    if set(y) != {0, 1} or not schema:
        raise ValueError("Model fitting requires both classes and a nonempty fixed feature schema")
    weights = training_weights(y, [_group(r) for r in usable], weighting)
    x = (
        matrix([r["features"] for r in usable], schema)
        if precomputed_x is None
        else np.asarray(precomputed_x)
    )
    if x.shape != (len(usable), 2 * len(schema)):
        raise ValueError("Precomputed matrix must match the usable rows and frozen schema")
    pipeline = build_pipeline(family, parameters, seed)
    if parameters.get("_no_missing"):
        pipeline.steps.insert(0, ("indicators", FunctionTransformer(_values_only)))
    if parameters.get("_robustscale") and family != "xgboost":
        pipeline.steps[next(i for i, (name, _) in enumerate(pipeline.steps) if name == "scale")] = (
            "scale",
            RobustScaler(),
        )
    kwargs = {"model__sample_weight": weights}
    if family != "xgboost" and not parameters.get("_robustscale"):
        kwargs["scale__sample_weight"] = weights
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(x, y, **kwargs)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise ValueError("Candidate did not converge")
    return pipeline


def _tune(rows, family, schema, seed, trials, log, deadline, weighting="patient_class", preprocessing=None):
    usable = _usable(rows)
    y, groups = np.array([r["label"] for r in usable]), np.array([_group(r) for r in usable])
    if len(set(groups)) < 3:
        raise ValueError("Inner tuning requires three independent groups")
    raw_matrix = matrix([row["features"] for row in usable], schema)
    folds = list(StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups))
    tested = []
    candidates = (
        [{"C": 0.1}, {"C": 1.0}][:trials]
        if family == "logistic" and (preprocessing or {}).get("_current_search")
        else _parameters(family, seed, trials)
    )
    for number, params in enumerate(candidates):
        params.update(preprocessing or {})
        if time.monotonic() >= deadline:
            raise TimeoutError("Investigation compute budget exhausted")
        start = time.monotonic()
        trial = {
            "model": family,
            "features": schema,
            "parameters": params,
            "trial": number,
            "seed": seed,
            "validation": "three_inner_grouped_folds",
            "weighting": weighting,
        }
        try:
            scores = np.full(len(y), np.nan)
            for fit, hold in folds:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Investigation compute budget exhausted")
                pipeline = _fit(
                    [usable[i] for i in fit],
                    family,
                    schema,
                    params,
                    seed,
                    weighting,
                    precomputed_x=raw_matrix[fit],
                )
                scores[hold] = pipeline.predict_proba(raw_matrix[hold])[:, 1]
            if not np.isfinite(scores).all():
                raise ValueError("Incomplete inner out-of-fold coverage")
            trial.update(status="complete", group_macro_log_loss=patient_macro_log_loss(y, scores, groups))
            tested.append(trial)
        except ValueError as error:
            trial.update(status="failed", reason=str(error))
        trial["wall_seconds"] = time.monotonic() - start
        log(trial)
    if not tested:
        raise ValueError("Every inner trial failed")
    return min(tested, key=lambda t: t["group_macro_log_loss"])["parameters"]


class BlendPredictor:
    """One prespecified equal-weight linear/nonlinear blend; research-only evidence."""

    def __init__(self, first, second, config, manifest):
        self.first, self.second, self.config, self.manifest = first, second, config, manifest
        self.version = manifest["version"]
        self.schema = sorted(set(first.schema + second.schema))

    def probabilities(self, rows):
        return (self.first.probabilities(rows) + self.second.probabilities(rows)) / 2

    def predict(self, features):
        members = [self.first.predict(features), self.second.predict(features)]
        evidence = [
            dict(item, member=member.version)
            for member, result in zip((self.first, self.second), members, strict=True)
            for item in result["evidence"][:3]
        ]
        return {
            "risk": float(np.mean([member["risk"] for member in members])),
            "evidence": evidence,
            "uncertainty": "Research ensemble; each contribution explains its own member score, not the fused probability or causation.",
        }


def predict_rows(predictor, rows):
    output = [dict(_compact_prediction(row), risk=None) for row in rows]
    available = [i for i, row in enumerate(rows) if row["accepted"]]
    if available:
        risks = predictor.probabilities([rows[i]["features"] for i in available])
        for index, risk in zip(available, risks, strict=True):
            output[index]["risk"] = float(np.clip(risk, 0, np.nextafter(1.0, 0.0)))
    return output


def policy_search_identity(rows, events, config):
    """Bind reuse to the whole risk stream and state/coverage contract, not the requested budget."""
    a = config["alert"]
    contract = dict(target=config["target"],
                    state={k: a.get(k) for k in ("policy_version", "max_duration_seconds", "rearm_seconds", "persistence_seconds", "cooldown_seconds")},
                    release_ratio=a["release_threshold"] / max(a["threshold"], 1e-12),
                    coverage="one_cadence_support_v1" if a.get("policy_version") == "finite-validity-v1" else "legacy_hold_to_next",
                    events=[dict(e) for e in events])
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True, allow_nan=False).encode())
    fields = ("patient_id", "group_id", "record_id", "timestamp", "duration_seconds", "accepted", "risk", "label", "reason", "label_reason")
    for row in rows:
        value = {k: row.get(k) for k in fields}
        value.update({k: row[k] for k in row if k.endswith(("_identity", "_sha256"))})
        digest.update(json.dumps(value, sort_keys=True, allow_nan=False).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def select_policy(rows, events, config, budget=0.25, search=None, *, deadline=None):
    """Policy fitting uses only policy patients, with all warning burden retained."""
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Policy selection compute budget exhausted")
    config = deepcopy(config)
    finite = config["alert"].get("policy_version") == "finite-validity-v1"
    bound_search = search is not None and any("search_identity" in item for item in search)
    identity = policy_search_identity(rows, events, config) if finite or bound_search else None
    if search is not None and (finite or bound_search) and any(item.get("search_identity") != identity for item in search):
        raise ValueError("Policy search identity differs from stream/events/state contract")
    if finite and ({r["label"] for r in rows if r.get("accepted") and r.get("risk") is not None and r.get("label") in (0, 1)} != {0, 1} or not events):
        raise ValueError("Finite policy role requires both classes and eligible events")
    ratio = config["alert"]["release_threshold"] / max(config["alert"]["threshold"], 1e-12)
    candidates = (
        []
        if search is None
        else [
            dict(
                item,
                meets_constraint=policy_constraints(item, config, budget=budget)["passed"] if finite else item["false_warnings_per_hour"] is not None
                and item["event_sensitivity"] is not None
                and item["false_warnings_per_hour"] <= budget,
            )
            for item in search
        ]
    )
    # Fixed grid, including near-zero and near-one useful at real low prevalence.
    quantile_levels = [0.5, 0.75, 0.9, 0.925, 0.95, 0.975, 0.99, 0.995, 0.999]
    eligible_risks = [
        row["risk"]
        for row in rows
        if row.get("accepted") and row.get("label") in (0, 1) and row.get("risk") is not None
    ]
    adaptive = (
        np.quantile(eligible_risks, quantile_levels).tolist() if eligible_risks and search is None else []
    )
    thresholds = (
        sorted(set(np.r_[np.linspace(0.01, 0.99, 25), 0.001, 0.005, 0.025, 0.995, 0.999, 1.0, adaptive]))
        if search is None
        else []
    )
    for threshold in thresholds:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Policy selection compute budget exhausted")
        policy = deepcopy(config)
        policy["alert"].update(
            threshold=float(threshold),
            release_threshold=float(threshold * ratio),
            max_false_warnings_per_hour=budget,
            min_event_sensitivity=0.0,
        )
        report = evaluate_stream(rows, events, policy, bootstrap_samples=0, include_probability=False)
        rate, sensitivity = report["false_warnings_per_hour"], report["event_sensitivity"]
        candidates.append(
            {
                "threshold": float(threshold),
                "grid_scope": "fixed grid plus predeclared quantiles of accepted labeled policy-group risks only",
                "quantile_levels": quantile_levels,
                "policy_eligible_windows": len(eligible_risks),
                "release_threshold": float(threshold * ratio),
                "meets_constraint": rate is not None and sensitivity is not None and rate <= budget,
                **{
                    k: report[k]
                    for k in (
                        "event_sensitivity",
                        "false_warnings_per_hour",
                        "warnings",
                        "unassessable_warnings",
                        "warning_hours",
                        "eligible_events",
                    )
                },
            }
        )
        if finite:
            candidates[-1].update(search_identity=identity, **policy_constraints(report, policy, budget=budget),
                                  meets_constraint=policy_constraints(report, policy, budget=budget)["passed"],
                                  budget_request=budget,
                                  occupancy_request={k: policy["alert"][k] for k in ("max_burden_occupancy", "max_eligible_negative_burden_occupancy")},
                                  **{k: report[k] for k in ("burden_occupancy", "eligible_negative_burden_occupancy", "burden_seconds", "monitored_seconds", "eligible_seconds", "eligible_negative_seconds", "eligible_negative_burden_seconds", "unresolved_elevated_seconds")})
    if finite and search is not None:
        for item in candidates:
            item.update(**policy_constraints(item, config, budget=budget), budget_request=budget,
                        occupancy_request={k: config["alert"][k] for k in ("max_burden_occupancy", "max_eligible_negative_burden_occupancy")})
    passed = [c for c in candidates if c["meets_constraint"]]
    if not passed:
        return None, candidates
    best = min(
        passed,
        key=lambda p: (
            -p["event_sensitivity"],
            p["false_warnings_per_hour"],
            p["warnings"],
            p["burden_seconds"] if finite else p["warning_hours"],
            -p["threshold"],
        ),
    )
    config["alert"].update(
        threshold=best["threshold"],
        release_threshold=best["release_threshold"],
        max_false_warnings_per_hour=budget,
        min_event_sensitivity=0.0,
    )
    return config, candidates


def _frozen_policies(rows, events, config, searches, selected):
    policies = {}
    for budget in (0.1, 0.25, 1):
        policy, _ = select_policy(rows, events, config, budget, search=searches)
        policies[str(budget)] = policy["alert"] if policy else None
    policies["0.25"] = selected["alert"]
    return policies


def _peak_memory():
    used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(used if sys.platform == "darwin" else used * 1024)


def _compact_prediction(row):
    return {key: value for key, value in row.items() if key != "features"}


def _log_writer(output):
    def log(value):
        value = dict(
            value,
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            peak_process_rss_bytes=_peak_memory(),
        )
        with (output / "experiments.jsonl").open("a") as handle:
            handle.write(json.dumps(value, allow_nan=False) + "\n")
            handle.flush()

    return log


def _rank(report, cost=0):
    rate = report["false_warnings_per_hour"]
    sensitivity = report["event_sensitivity"]
    brier = report["probability"].get("group_macro_brier", report["probability"].get("patient_macro_brier"))
    return (
        rate is not None and rate <= 0.25 and sensitivity is not None,
        sensitivity if sensitivity is not None else -1,
        -(rate if rate is not None else float("inf")),
        -report["warnings"],
        -report["warning_hours"],
        -(brier if brier is not None else float("inf")),
        -cost,
    )


def paired_group_comparison(first, second, groups, seed=42, samples=500):
    """Paired bootstrap of whole independent groups; no window resampling."""

    def indexed(report):
        counts = {}
        for patient in report["per_patient"]:
            group = groups[patient["patient_id"]]
            target = counts.setdefault(group, Counter())
            for key in (
                "eligible_events",
                "matched_events",
                "false_warnings",
                "eligible_hours",
                "warnings",
                "warning_hours",
            ):
                target[key] += patient.get(key, 0)
        return counts

    left, right = indexed(first), indexed(second)
    ids = sorted(set(left) & set(right))
    if not ids or set(left) != set(right):
        return {"status": "unavailable", "reason": "Paired reports need identical independent groups"}
    rng, differences = np.random.default_rng(seed), []
    false_deltas, warning_deltas, duration_deltas, brier_deltas = [], [], [], []
    for _ in range(samples):
        a, b = Counter(), Counter()
        sampled = rng.choice(ids, size=len(ids), replace=True)
        for group in sampled:
            a.update(left[group])
            b.update(right[group])
        if a["eligible_events"] and b["eligible_events"]:
            differences.append(
                a["matched_events"] / a["eligible_events"] - b["matched_events"] / b["eligible_events"]
            )
        if a["eligible_hours"] and b["eligible_hours"]:
            false_deltas.append(
                a["false_warnings"] / a["eligible_hours"] - b["false_warnings"] / b["eligible_hours"]
            )
            warning_deltas.append(a["warnings"] / a["eligible_hours"] - b["warnings"] / b["eligible_hours"])
            duration_deltas.append(
                a["warning_hours"] / a["eligible_hours"] - b["warning_hours"] / b["eligible_hours"]
            )
        if first.get("per_group_brier") and second.get("per_group_brier"):
            valid = [g for g in sampled if g in first["per_group_brier"] and g in second["per_group_brier"]]
            if valid:
                brier_deltas.append(
                    float(
                        np.mean([first["per_group_brier"][g] - second["per_group_brier"][g] for g in valid])
                    )
                )
    return {
        "status": "complete",
        "metric": "event_sensitivity_first_minus_second",
        "groups": len(ids),
        "samples": len(differences),
        "ci95": np.percentile(differences, [2.5, 97.5]).tolist() if differences else None,
        "false_warning_rate_ci95": np.percentile(false_deltas, [2.5, 97.5]).tolist()
        if false_deltas
        else None,
        "total_warning_rate_ci95": np.percentile(warning_deltas, [2.5, 97.5]).tolist()
        if warning_deltas
        else None,
        "warning_occupancy_ci95": np.percentile(duration_deltas, [2.5, 97.5]).tolist()
        if duration_deltas
        else None,
        "group_macro_brier_ci95": np.percentile(brier_deltas, [2.5, 97.5]).tolist() if brier_deltas else None,
        "limitation": "Descriptive group bootstrap; shared training folds induce dependency; repeated search comparisons are not multiplicity-adjusted.",
    }


def select_reports(combined, groups, seed=42):
    """Compare complete classical and neural outer folds under the same rule."""
    contenders = {
        name: report
        for name, report in combined.items()
        if name not in {"no_warning", "training_prevalence"} and report.get("complete_outer_folds") == 3
    }
    ranked = sorted(
        contenders,
        key=lambda name: _rank(contenders[name], contenders[name].get("cost_seconds", 0)),
        reverse=True,
    )
    baseline = "logistic_current_recipe" if "logistic_current_recipe" in contenders else "logistic_core"
    comparisons = (
        {
            name: paired_group_comparison(report, contenders[baseline], groups, seed)
            for name, report in contenders.items()
        }
        if baseline in contenders
        else {}
    )
    selected = baseline if baseline in contenders and _rank(contenders[baseline])[0] else None
    for name in ranked:
        if not _rank(contenders[name])[0]:
            continue
        comparison = comparisons.get(name, {})
        ci = comparison.get("ci95")
        secondary_better = any(
            comparison.get(key) and comparison[key][1] < 0
            for key in (
                "false_warning_rate_ci95",
                "total_warning_rate_ci95",
                "warning_occupancy_ci95",
                "group_macro_brier_ci95",
            )
        )
        tied_sensitivity = (
            selected is not None
            and contenders[name]["event_sensitivity"] == contenders[baseline]["event_sensitivity"]
            and _rank(contenders[name], contenders[name].get("cost_seconds", 0))
            > _rank(contenders[baseline], contenders[baseline].get("cost_seconds", 0))
        )
        if selected is None or (
            name != baseline and ci and (ci[0] > 0 or tied_sensitivity and secondary_better)
        ):
            selected = name
            break
    return {"selected_candidate": selected, "ranking": ranked, "paired_comparisons_to_core": comparisons}


class ProbabilityRows(Sequence):
    """Compact sufficient data for probability metrics after alert evaluation."""

    def __init__(self, path, patients, groups):
        self.values = np.load(path, mmap_mode="r")
        self.patients, self.groups = patients, groups

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        row = self.values[index]
        return {
            "patient_id": self.patients[int(row["patient"])],
            "group_id": self.groups[int(row["group"])],
            "label": None if row["label"] < 0 else int(row["label"]),
            "risk": None if np.isnan(row["risk"]) else float(row["risk"]),
        }


def _pool_fold(output, name, fold, predictions, report):
    """Keep numeric probabilities and fold counts, not every model's Python timeline."""
    patients = sorted({row["patient_id"] for row in predictions})
    groups = sorted({_group(row) for row in predictions})
    patient_codes, group_codes = {p: i for i, p in enumerate(patients)}, {g: i for i, g in enumerate(groups)}
    path = output / f"fold-{fold}-{name}-probability.npy"
    values = np.lib.format.open_memmap(
        path,
        mode="w+",
        shape=(len(predictions),),
        dtype=[("label", "int8"), ("risk", "float64"), ("patient", "int32"), ("group", "int32")],
    )
    errors = {}
    for number, row in enumerate(predictions):
        group = _group(row)
        values[number] = (
            -1 if row["label"] is None else row["label"],
            np.nan if row["risk"] is None else row["risk"],
            patient_codes[row["patient_id"]],
            group_codes[group],
        )
        if row["label"] in (0, 1) and row["risk"] is not None:
            total, count = errors.get(group, (0.0, 0))
            errors[group] = (total + (row["label"] - row["risk"]) ** 2, count + 1)
    values.flush()
    return {
        "report": report,
        "probabilities": ProbabilityRows(path, patients, groups),
        "group_errors": errors,
    }


def run_ladder(
    rows,
    events,
    config,
    output,
    max_hours=8,
    seed=42,
    trials=6,
    specs=None,
    fitting_cadence_seconds=120,
    cache_contract=None,
):
    """Three outer grouped folds; all inner fitting/calibration/policy groups disjoint.

    Only DEVELOPMENT rows may be supplied. Original primary test membership must
    be stored by the caller and is bound to the subsequent frozen package.
    """
    output = _fresh(output)
    deadline, started = time.monotonic() + max_hours * 3600, time.monotonic()
    log = _log_writer(output)
    usable = _usable(rows)
    all_groups = sorted({_group(r) for r in rows})
    event_groups = {_group(e) for e in events}
    y = np.array([int(g in event_groups) for g in all_groups])
    groups = np.array(all_groups)
    if len(set(groups)) < 12 or {r["label"] for r in usable} != {0, 1}:
        raise ValueError("Nested ladder requires at least twelve development groups and both classes")
    schemas = feature_groups(usable)
    specs = specs or [
        ("logistic", "current_recipe"),
        ("logistic", "core"),
        ("logistic", "engineered"),
        ("logistic", "combined"),
        ("logistic", "availability_control"),
        ("logistic", "clinical"),  # Static-only reference, not evidence of fusion.
        ("xgboost", "core"),
        ("xgboost", "engineered"),
        ("xgboost", "combined"),
        ("xgboost", "personal"),
        ("elasticnet", "engineered"),
        ("elasticnet", "personal"),
        ("elasticnet", "combined"),
        ("logistic", "core_patient_only"),
        ("logistic", "core_window_class"),
        ("logistic", "core_no_missing"),
        ("logistic", "core_robustscale"),
        ("logistic", "dispersion"),
        ("logistic", "dynamics"),
        ("logistic", "nonlinear"),
        ("logistic", "motifs"),
    ]
    pooled, results, times = {}, [], Counter()
    folds = list(StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups))
    iterations = [
        (candidate_number, fold, train, hold, spec)
        for candidate_number, spec in enumerate(specs)
        for fold, (train, hold) in enumerate(folds)
    ]
    fold_contexts, stored_predictors, blended_folds = {}, {}, set()
    budget_exhausted = False
    for candidate_number, fold, train, hold, active_spec in iterations:
        if budget_exhausted:
            break
        fitting_groups, outer_groups = set(groups[train]), set(groups[hold])
        if fitting_groups & outer_groups:
            raise ValueError("Outer groups overlap")
        if fold not in fold_contexts:
            fit_groups, cal_groups, policy_groups = _partitions(_subset(rows, fitting_groups), seed + fold)
            fit_rows, cal_rows, policy_rows, outer_rows = [
                _subset(rows, g) for g in (fit_groups, cal_groups, policy_groups, outer_groups)
            ]
            assignments = {
                "fit": fit_groups,
                "calibration": cal_groups,
                "policy": policy_groups,
                "outer": sorted(outer_groups),
            }
            _write(output / f"fold-{fold}-groups.json", assignments)
            fold_contexts[fold] = (fit_rows, cal_rows, policy_rows, outer_rows, assignments)
        fit_rows, cal_rows, policy_rows, outer_rows, assignments = fold_contexts[fold]
        training_prevalence = float(np.mean([r["label"] for r in _usable(fit_rows)]))
        controls = (
            (("no_warning", None), ("training_prevalence", training_prevalence))
            if candidate_number == 0
            else ()
        )
        for constant, probability in controls:
            constant_rows = [dict(r, risk=probability if r["accepted"] else None) for r in outer_rows]
            if constant == "no_warning":
                constant_config = deepcopy(config)
                constant_config["alert"].update(threshold=1.0, release_threshold=1.0)
                # risk zero remains an available forecast; no-warning is a policy, not abstention.
                constant_rows = [dict(r, risk=0.0 if r["accepted"] else None) for r in outer_rows]
            else:
                fitting_control = [dict(r, risk=probability if r["accepted"] else None) for r in policy_rows]
                constant_config, _ = select_policy(fitting_control, _events(events, policy_rows), config)
                constant_config = constant_config or deepcopy(config)
            for r in constant_rows:
                r["record_id"] = f"fold-{fold}:" + r["record_id"]
            target_events = [
                dict(e, record_id=f"fold-{fold}:" + e["record_id"]) for e in _events(events, outer_rows)
            ]
            report = evaluate_stream(constant_rows, target_events, constant_config, bootstrap_samples=0)
            results.append({"model": constant, "fold": fold, "status": "complete", "report": report})
            pooled.setdefault(constant, []).append(_pool_fold(output, constant, fold, constant_rows, report))
        fold_predictors = stored_predictors.setdefault(fold, {})
        for family, feature_set in [active_spec]:
            name = f"{family}_{feature_set}"
            if time.monotonic() >= deadline:
                budget_exhausted = True
                break
            schema = schemas[feature_set]
            experiment = {
                "experiment_id": f"seed-{seed}/fold-{fold}/{name}",
                "model": name,
                "fold": fold,
                "seed": seed,
                "features": schema,
                "partition_groups": assignments,
                "preprocessing": "fit-only median + explicit missing indicators; weighted scaling for linear",
                "augmentation": "none",
                "loss": "weighted binary log loss",
                "optimizer": "sklearn/xgboost",
                "learning_rate": 0.05 if family == "xgboost" else None,
                "scheduler": None,
                "batch_size": None,
                "epochs": None,
                "validation": "three_outer_grouped_folds",
                "metric": "event_sensitivity_at_0.25_false_warnings_per_eligible_hour",
            }
            began = time.monotonic()
            try:
                if feature_set == "motifs" and not set(schema) & set(MOTIF_FIELDS):
                    raise ValueError("Motif group absent: build a motifs-enabled cache first")
                selected_fit = (
                    fit_rows
                    if feature_set in {"core_cadence30", "current_recipe"}
                    else [row for row in fit_rows if abs(row["timestamp"] % fitting_cadence_seconds) < 1e-9]
                )
                weighting = (
                    "patient_only"
                    if feature_set == "core_patient_only"
                    else "window_class"
                    if feature_set == "core_window_class"
                    else "patient_class"
                )
                preprocessing = {
                    "_no_missing": feature_set == "core_no_missing",
                    "_robustscale": feature_set == "core_robustscale",
                    "_current_search": feature_set.startswith("core"),
                }
                experiment["fit_cadence_seconds"] = (
                    30 if feature_set in {"core_cadence30", "current_recipe"} else fitting_cadence_seconds
                )
                experiment["weighting"] = weighting
                if feature_set == "current_recipe":
                    pipeline, schema, recipe_details = _fit_model(
                        fit_rows, "logistic", "ecg", seed + fold, "patient_class"
                    )
                    params = recipe_details["chosen"]["parameters"]
                    experiment["current_recipe_details"] = recipe_details
                else:
                    params = _tune(
                        selected_fit,
                        family,
                        schema,
                        seed + fold,
                        trials,
                        lambda t: log(dict(t, parent_experiment_id=experiment["experiment_id"])),
                        deadline,
                        weighting,
                        preprocessing,
                    )
                    pipeline = _fit(selected_fit, family, schema, params, seed + fold, weighting)
                calibrator = _calibrate(
                    pipeline, schema, cal_rows, seed + fold, config["training"]["calibration_c"]
                )
                predictor = Predictor(
                    pipeline, calibrator, schema, config, {"version": name, "synthetic": False}
                )
                if name in {"logistic_core", "xgboost_personal"}:
                    fold_predictors[name] = predictor
                policy_predictions = predict_rows(predictor, policy_rows)
                main_policy, main_search = select_policy(
                    policy_predictions, _events(events, policy_rows), config, 0.25, deadline=deadline
                )
                policies = {
                    budget: select_policy(
                        policy_predictions,
                        _events(events, policy_rows),
                        config,
                        budget,
                        main_search,
                        deadline=deadline,
                    )
                    for budget in (0.1, 0.25, 1)
                }
                chosen, searches = policies[0.25]
                experiment["operating_constraint_available"] = chosen is not None
                if chosen is None:
                    chosen = deepcopy(config)
                    chosen["alert"].update(
                        threshold=1.0, release_threshold=1.0, max_false_warnings_per_hour=0.25
                    )
                infer_start = time.monotonic()
                predictions = predict_rows(predictor, outer_rows)
                inference_seconds = time.monotonic() - infer_start
                frontier = {}
                for budget, (policy, _) in policies.items():
                    frontier[str(budget)] = (
                        evaluate_stream(predictions, _events(events, outer_rows), policy, bootstrap_samples=0)
                        if policy
                        else None
                    )
                report = frontier["0.25"] or evaluate_stream(
                    predictions, _events(events, outer_rows), chosen, bootstrap_samples=0
                )
                for r in predictions:
                    r["record_id"] = f"fold-{fold}:" + r["record_id"]
                calibration_ablations = {}
                outer_score_rows = [r for r in outer_rows if r["accepted"]]
                outer_matrix = matrix([r["features"] for r in outer_score_rows], schema)
                raw_scores = pipeline.predict_proba(outer_matrix)[:, 1]
                outer_scores = raw_score(pipeline, outer_matrix)
                raw_iter = iter(raw_scores)
                raw_predictions = [
                    dict(_compact_prediction(r), risk=float(next(raw_iter)) if r["accepted"] else None)
                    for r in outer_rows
                ]
                from .evaluation import probability_metrics

                calibration_ablations["raw"] = probability_metrics(raw_predictions)
                for strength in (1.0, 100.0):
                    alternative = (
                        calibrator
                        if strength == config["training"]["calibration_c"]
                        else _calibrate(pipeline, schema, cal_rows, seed + fold, strength)
                    )
                    alternative_risks = iter(alternative.predict_proba(outer_scores.reshape(-1, 1))[:, 1])
                    alternate_rows = [
                        dict(
                            _compact_prediction(r),
                            risk=float(next(alternative_risks)) if r["accepted"] else None,
                        )
                        for r in outer_rows
                    ]
                    calibration_ablations[str(strength)] = probability_metrics(alternate_rows)
                experiment.update(
                    status="complete",
                    calibration_ablations=calibration_ablations,
                    parameters=params,
                    report=report,
                    acceptance_gate=report["event_sensitivity"] is not None
                    and report["event_sensitivity"] >= (config["alert"].get("min_event_sensitivity") or 0.5)
                    and report["false_warnings_per_hour"] is not None
                    and report["false_warnings_per_hour"] <= 0.25,
                    frontier=frontier,
                    policy_search=searches,
                    selected_policy=chosen["alert"],
                    inference_seconds=inference_seconds,
                    inference_ms_per_available_window=1000
                    * inference_seconds
                    / max(1, sum(r["accepted"] for r in outer_rows)),
                    parameter_count=int(pipeline["model"].coef_.size + pipeline["model"].intercept_.size)
                    if hasattr(pipeline["model"], "coef_")
                    else None,
                )
                joblib.dump({"predictor": predictor, "policy": chosen}, output / f"fold-{fold}-{name}.joblib")
                experiment["predictions_artifact"] = _write_predictions(
                    output / f"fold-{fold}-{name}-predictions.jsonl.gz", predictions
                )
                pooled.setdefault(name, []).append(_pool_fold(output, name, fold, predictions, report))
            except (ValueError, TimeoutError) as error:
                experiment.update(status="failed", reason=str(error))
                if isinstance(error, TimeoutError):
                    budget_exhausted = True
            experiment["training_and_selection_seconds"] = time.monotonic() - began
            times[name] += experiment["training_and_selection_seconds"]
            log(experiment)
            results.append(experiment)
            if budget_exhausted:
                break
        if budget_exhausted:
            break
        if (
            not budget_exhausted
            and fold not in blended_folds
            and {"logistic_core", "xgboost_personal"} <= set(fold_predictors)
        ):
            blended_folds.add(fold)
            began = time.monotonic()
            blend = BlendPredictor(
                fold_predictors["logistic_core"],
                fold_predictors["xgboost_personal"],
                config,
                {"version": "blend_core_xgboost", "synthetic": False},
            )
            policy_predictions = predict_rows(blend, policy_rows)
            try:
                chosen, searches = select_policy(
                    policy_predictions, _events(events, policy_rows), config, deadline=deadline
                )
            except TimeoutError as error:
                budget_exhausted = True
                log({"model": "blend_core_xgboost", "fold": fold, "status": "failed", "reason": str(error)})
                break
            trial = {
                "experiment_id": f"seed-{seed}/fold-{fold}/blend_core_xgboost",
                "model": "blend_core_xgboost",
                "fold": fold,
                "seed": seed,
                "members": ["logistic_core", "xgboost_personal"],
                "weights": [0.5, 0.5],
                "validation": "three_outer_grouped_folds",
                "augmentation": "none",
            }
            if chosen is not None:
                predictions = predict_rows(blend, outer_rows)
                report = evaluate_stream(
                    predictions, _events(events, outer_rows), chosen, bootstrap_samples=0
                )
                trial.update(
                    status="complete", report=report, selected_policy=chosen["alert"], policy_search=searches
                )
                for row in predictions:
                    row["record_id"] = f"fold-{fold}:" + row["record_id"]
                joblib.dump(
                    {"predictor": blend, "policy": chosen}, output / f"fold-{fold}-blend_core_xgboost.joblib"
                )
                trial["predictions_artifact"] = _write_predictions(
                    output / f"fold-{fold}-blend_core_xgboost-predictions.jsonl.gz", predictions
                )
                pooled.setdefault("blend_core_xgboost", []).append(
                    _pool_fold(output, "blend_core_xgboost", fold, predictions, report)
                )
            else:
                trial.update(status="failed", reason="No ensemble policy meets the fixed development budget")
            trial["training_and_selection_seconds"] = time.monotonic() - began
            times["blend_core_xgboost"] += trial["training_and_selection_seconds"]
            log(trial)
            results.append(trial)
    combined = {}
    for name, parts in pooled.items():
        if len(parts) != 3:
            continue
        # Each fold has a separately fitted policy; aggregate counts, not a single post-hoc threshold.
        reports = [part["report"] for part in parts]
        counts = Counter()
        for report in reports:
            for key in (
                "eligible_events",
                "matched_events",
                "missed_events",
                "events_missed_during_abstention",
                "warnings",
                "false_warnings",
                "unassessable_warnings",
                "known_excluded_warnings",
                "unknown_outcome_warnings",
                "cooldown_suppressed_updates",
                "monitored_seconds",
                "eligible_seconds",
                "available_seconds",
                "rejected_seconds",
                "warning_seconds",
                "eligible_hours",
                "warning_hours",
                "monitored_hours",
                "rejected_hours",
            ):
                counts[key] += report[key]
        counts["event_sensitivity"] = (
            counts["matched_events"] / counts["eligible_events"] if counts["eligible_events"] else None
        )
        counts["false_warnings_per_hour"] = (
            counts["false_warnings"] / counts["eligible_hours"] if counts["eligible_hours"] else None
        )
        all_rows = chain.from_iterable(part["probabilities"] for part in parts)
        from .evaluation import _rates, probability_metrics

        rates = _rates(counts)
        reasons = {}
        for key in (
            "exclusion_hours",
            "rejection_hours",
            "unassessable_warning_reasons",
            "known_excluded_warning_reasons",
            "unknown_outcome_warning_reasons",
        ):
            values = Counter()
            for report in reports:
                values.update(report[key])
            reasons[key] = dict(values)

        group_errors = {}
        for part in parts:
            for group, (total, count) in part["group_errors"].items():
                prior_total, prior_count = group_errors.get(group, (0.0, 0))
                group_errors[group] = (prior_total + total, prior_count + count)
        if name == "blend_core_xgboost":
            times[name] += times["logistic_core"] + times["xgboost_personal"]
        combined[name] = {
            **counts,
            **rates,
            **reasons,
            "per_group_brier": {group: total / count for group, (total, count) in group_errors.items()},
            "probability": probability_metrics(all_rows),
            "per_patient": [p for report in reports for p in report["per_patient"]],
            "complete_outer_folds": 3,
            "cost_seconds": times[name],
        }
    selection = select_reports(combined, {r["patient_id"]: _group(r) for r in rows}, seed)
    selected = selection["selected_candidate"]
    ranked, comparisons = selection["ranking"], selection["paired_comparisons_to_core"]
    research_winner = selected
    summary = {
        "research_winner": research_winner,
        "extraction_contract": _extraction_contract(config),
        "development_cache_contract": cache_contract,
        "ensemble_deployment_note": "Blend is separately evaluated; serving adaptation required before production promotion.",
        "seed": seed,
        "primary_budget": 0.25,
        "status": "budget_exhausted" if budget_exhausted else "complete",
        "scope": "development-only nested group validation; primary and external tests untouched",
        "combined": combined,
        "paired_comparisons_to_core": comparisons,
        "selected_candidate": selected,
        "ranking": ranked,
        "complete_outer_models": sorted(combined),
        "traversal": "model-first; complete original baseline before challenger search",
        "specs": specs,
        "fitting_cadence_seconds": fitting_cadence_seconds,
        "tuning_trials": min(trials, 6),
        "wall_seconds": time.monotonic() - started,
        "peak_process_rss_bytes": _peak_memory(),
        "selection_rule": "Retain core unless paired group sensitivity improves or tied sensitivity has significant lower burden/Brier, at the fixed budget.",
        "acceptance_gate_threshold": config["alert"].get("min_event_sensitivity") or 0.5,
        "acceptance_gate_passed": bool(
            selected
            and combined[selected]["event_sensitivity"]
            >= (config["alert"].get("min_event_sensitivity") or 0.5)
        ),
        "failed_experiments": [
            {k: r.get(k) for k in ("model", "fold", "reason")} for r in results if r["status"] == "failed"
        ],
    }
    summary["sha256"] = _digest(summary)
    _write(output / "summary.json", summary)
    return summary


def freeze_finalist(
    rows,
    events,
    config,
    split,
    candidate,
    output,
    seed=42,
    trials=6,
    max_hours=1,
    fitting_cadence_seconds=120,
    selection_summary=None,
    development_contract=None,
    research_baseline=False,
):
    """Refit a DEVELOPMENT-selected candidate and freeze membership before test extraction."""
    if research_baseline and (
        selection_summary is None
        or selection_summary.get("selected_candidate") is not None
        or candidate != "logistic_current_recipe"
    ):
        raise ValueError("Research baseline requires no qualifying winner and the exact current recipe")
    if selection_summary is not None:
        if _digest({k: v for k, v in selection_summary.items() if k != "sha256"}) != selection_summary.get(
            "sha256"
        ):
            raise ValueError("Development selection summary checksum mismatch")
        if (
            not research_baseline and selection_summary["selected_candidate"] != candidate
        ) or selection_summary["seed"] != seed:
            raise ValueError("Finalist differs from the recorded development selection")
        if (
            selection_summary["extraction_contract"] != _extraction_contract(config)
            or selection_summary.get("development_cache_contract") != development_contract
        ):
            raise ValueError("Finalist data/recipe differs from the recorded development selection")
        if selection_summary["fitting_cadence_seconds"] != fitting_cadence_seconds:
            raise ValueError("Finalist fitting cadence differs from development experiments")
        if selection_summary["tuning_trials"] != min(trials, 6):
            raise ValueError("Finalist tuning budget differs from development experiments")
    output = _fresh(output)
    if any(_group(r) in set(split["test_groups"]) for r in rows):
        raise ValueError("Primary test rows cannot enter finalist fitting")
    if {_group(r) for r in rows} != set(split["development_groups"]):
        raise ValueError("Unbound development groups")
    original = {k: v for k, v in split.items() if k != "sha256"}
    if _digest(original) != split["sha256"]:
        raise ValueError("Frozen split checksum mismatch")
    if candidate == "blend_core_xgboost":
        return _freeze_blend(
            rows,
            events,
            config,
            split,
            output,
            seed,
            trials,
            max_hours,
            fitting_cadence_seconds,
            selection_summary,
            development_contract,
        )
    family, feature_set = candidate.split("_", 1)
    fit, cal, policy = _partitions(rows, seed)
    fit_rows, cal_rows, policy_rows = [_subset(rows, group) for group in (fit, cal, policy)]
    schema = feature_groups(_usable(fit_rows))[feature_set]
    cadence = 30 if feature_set in {"core_cadence30", "current_recipe"} else fitting_cadence_seconds
    fit_rows = [row for row in fit_rows if abs(row["timestamp"] % cadence) < 1e-9]
    weighting = (
        "patient_only"
        if feature_set == "core_patient_only"
        else "window_class"
        if feature_set == "core_window_class"
        else "patient_class"
    )
    preprocessing = {
        "_no_missing": feature_set == "core_no_missing",
        "_robustscale": feature_set == "core_robustscale",
        "_current_search": feature_set.startswith("core"),
    }
    log = _log_writer(output)
    if feature_set == "current_recipe":
        pipeline, schema, recipe_details = _fit_model(fit_rows, "logistic", "ecg", seed, "patient_class")
        params = recipe_details["chosen"]["parameters"]
        _write(output / "current_recipe.json", recipe_details)
    else:
        params = _tune(
            fit_rows,
            family,
            schema,
            seed,
            trials,
            log,
            time.monotonic() + max_hours * 3600,
            weighting,
            preprocessing,
        )
        pipeline = _fit(fit_rows, family, schema, params, seed, weighting)
    calibrator = _calibrate(pipeline, schema, cal_rows, seed, config["training"]["calibration_c"])
    predictor = Predictor(pipeline, calibrator, schema, config, {"version": candidate, "synthetic": False})
    policy_predictions, policy_events = predict_rows(predictor, policy_rows), _events(events, policy_rows)
    selected, searches = select_policy(policy_predictions, policy_events, config)
    if selected is None:
        if not research_baseline:
            raise ValueError("Final development policy does not meet budget")
        selected = deepcopy(config)
        selected["alert"].update(threshold=1.0, release_threshold=0.8, min_event_sensitivity=0.0)
    contract = {
        "primary_split": split,
        "extraction": _extraction_contract(config),
        "development_cache_contract": development_contract,
        "selection_summary_sha256": selection_summary.get("sha256") if selection_summary else None,
        "selection_role": "predeclared_research_baseline" if research_baseline else "development_finalist",
        "operating_policies": _frozen_policies(policy_predictions, policy_events, config, searches, selected),
        "motifs": bool(set(schema) & set(MOTIF_FIELDS)),
    }
    contract["sha256"] = _digest(contract)
    package = {
        "pipeline": pipeline,
        "calibrator": calibrator,
        "schema": schema,
        "config": selected,
        "cohort_contract": contract,
    }
    joblib.dump(package, output / "model.joblib")
    from .data import file_hash

    development_passed = bool(selection_summary and selection_summary.get("acceptance_gate_passed") is True)
    final_policy_passed = max(
        (p["event_sensitivity"] for p in searches if p["meets_constraint"]), default=0
    ) >= (config["alert"].get("min_event_sensitivity") or 0.5)
    manifest = {
        "version": "investigation-" + candidate + "-" + split["sha256"][:12],
        "synthetic": False,
        "model_sha256": file_hash(output / "model.joblib"),
        "primary_split": split,
        "feature_version": FEATURE_VERSION,
        "parameters": params,
        "candidate": candidate,
        "fit_cadence_seconds": cadence,
        "weighting": weighting,
        "cohort_contract": contract,
        "selection_role": contract["selection_role"],
        "deployment_ready": not research_baseline
        and feature_set != "motifs"
        and development_passed
        and final_policy_passed,
        "development_acceptance_gate": development_passed,
        "final_policy_acceptance_gate": final_policy_passed,
        "acceptance_gate": development_passed and final_policy_passed,
        "fitting_groups": fit,
        "calibration_groups": cal,
        "policy_groups": policy,
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write(output / "manifest.json", manifest)
    _write(output / "config.json", selected)
    _write(output / "feature_schema.json", {"features": schema})
    _write(output / "policy_search.json", searches)
    return manifest


def _freeze_blend(
    rows,
    events,
    config,
    split,
    output,
    seed,
    trials,
    max_hours,
    fitting_cadence_seconds=120,
    selection_summary=None,
    development_contract=None,
):
    """Refit two declared members on identical fit/calibration/policy groups."""
    started = time.monotonic()
    members = []
    for name in ("logistic_core", "xgboost_personal"):
        remaining = max_hours - (time.monotonic() - started) / 3600
        if remaining <= 0:
            raise TimeoutError("Final ensemble budget exhausted")
        freeze_finalist(
            rows, events, config, split, name, output / name, seed, trials, remaining, fitting_cadence_seconds
        )
        # Member artifacts can be below the acceptance gate yet useful in a blend.
        path = output / name
        member_manifest = json.loads((path / "manifest.json").read_text())
        package = joblib.load(path / "model.joblib")
        members.append(
            Predictor(
                package["pipeline"],
                package["calibrator"],
                package["schema"],
                package["config"],
                member_manifest,
            )
        )
    _, _, policy_groups = _partitions(rows, seed)
    policy_rows = _subset(rows, policy_groups)
    blend = BlendPredictor(*members, config, {"version": "blend_core_xgboost"})
    policy_predictions, policy_events = predict_rows(blend, policy_rows), _events(events, policy_rows)
    selected, searches = select_policy(policy_predictions, policy_events, config)
    if selected is None:
        raise ValueError("Final ensemble policy does not meet the fixed budget")
    contract = {
        "primary_split": split,
        "extraction": _extraction_contract(config),
        "development_cache_contract": development_contract,
        "selection_summary_sha256": selection_summary.get("sha256") if selection_summary else None,
        "operating_policies": _frozen_policies(policy_predictions, policy_events, config, searches, selected),
        "motifs": False,
    }
    contract["sha256"] = _digest(contract)
    development_passed = bool(
        selection_summary
        and selection_summary.get("acceptance_gate_passed") is True
        and selection_summary.get("sha256")
        == _digest({key: value for key, value in selection_summary.items() if key != "sha256"})
    )
    final_policy_passed = max(
        (p["event_sensitivity"] for p in searches if p["meets_constraint"]), default=0
    ) >= (config["alert"].get("min_event_sensitivity") or 0.5)
    manifest = {
        "version": "investigation-blend-" + split["sha256"][:12],
        "synthetic": False,
        "primary_split": split,
        "cohort_contract": contract,
        "feature_version": FEATURE_VERSION,
        "candidate": "blend_core_xgboost",
        "members": [m.version for m in members],
        "deployment_ready": development_passed and final_policy_passed,
        "development_acceptance_gate": development_passed,
        "final_policy_acceptance_gate": final_policy_passed,
        "acceptance_gate": development_passed and final_policy_passed,
        "kind": "probability_blend",
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    blend.manifest, blend.version, blend.config = manifest, manifest["version"], selected
    joblib.dump(
        {"predictor": blend, "schema": blend.schema, "config": selected, "cohort_contract": contract},
        output / "model.joblib",
    )
    from .data import file_hash

    manifest["model_sha256"] = file_hash(output / "model.joblib")
    _write(output / "manifest.json", manifest)
    _write(output / "config.json", selected)
    _write(output / "feature_schema.json", {"features": blend.schema})
    _write(output / "policy_search.json", searches)
    return manifest


def evaluate_frozen(artifact, rows, events, output, cohort="primary", index=None):
    """One explicit held-out evaluation; refuses reused outputs or unknown primary groups."""
    artifact = Path(artifact)
    predictor = Predictor.load(artifact, research=True)
    manifest = predictor.manifest
    contract = manifest["cohort_contract"]
    if _digest({k: v for k, v in contract.items() if k != "sha256"}) != contract["sha256"]:
        raise ValueError("Cohort contract checksum mismatch")
    split = manifest["primary_split"]
    if split != contract["primary_split"]:
        raise ValueError("Manifest primary split differs from the embedded frozen cohort contract")
    if _digest({k: v for k, v in split.items() if k != "sha256"}) != split["sha256"]:
        raise ValueError("Frozen split checksum mismatch")
    if cohort == "primary":
        if {_group(r) for r in rows} != set(split["test_groups"]):
            raise ValueError("Primary evaluation requires exact frozen independent-group membership")
        expected_records = {
            record
            for record, identity in split["record_contracts"].items()
            if split["group_ids"][identity["patient_id"]] in set(split["test_groups"])
        }
        if {r["record_id"] for r in rows} != expected_records:
            raise ValueError("Primary recording membership differs from the frozen split")
        if index is None:
            raise ValueError("Primary evaluation requires the immutable cache/source contract")
        for record in index["records"]:
            identifier = record["record"]["record_id"]
            if record.get("input_contract") != split["record_contracts"].get(identifier):
                raise ValueError("Primary source content differs from frozen input identities")
        if _extraction_contract(index["config"]) != contract["extraction"]:
            raise ValueError("Primary extraction configuration differs from frozen recipe")
        if any(split["group_ids"].get(r["patient_id"]) != _group(r) for r in rows):
            raise ValueError("Primary patient aliases differ from frozen mapping")
    elif cohort == "external":
        if index is None or _extraction_contract(index["config"]) != contract["extraction"]:
            raise ValueError("External extraction configuration differs from frozen recipe")
    else:
        raise ValueError("Cohort must be primary or external")
    if contract.get("motifs") and not (index and index.get("motifs")):
        raise ValueError("Frozen motif features require a motifs-enabled held-out cache")
    ledger = artifact / "heldout_access.jsonl"
    previous = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    identity = index.get("input_sha256") if index else _digest(sorted({_group(r) for r in rows}))
    if any(item["cohort"] == cohort and item["input_sha256"] == identity for item in previous):
        raise ValueError("This frozen model has already consumed the held-out cohort")
    output = _fresh(output)
    with ledger.open("a") as handle:
        handle.write(
            json.dumps(
                {
                    "cohort": cohort,
                    "input_sha256": identity,
                    "accessed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            )
            + "\n"
        )
    started = time.monotonic()
    predicted = predict_rows(predictor, rows)
    report = evaluate_stream(predicted, events, predictor.config)
    frontier = {"0.25": dict(report)}
    for budget, alert in contract.get("operating_policies", {}).items():
        if budget == "0.25":
            continue
        policy = deepcopy(predictor.config)
        if alert is not None:
            policy["alert"] = alert
        frontier[budget] = evaluate_stream(predicted, events, policy) if alert is not None else None
    report.update(
        frontier=frontier,
        model_sha256=manifest["model_sha256"],
        cache_input_sha256=index.get("input_sha256") if index else None,
        source_annotation_sha256={
            item["record"]["record_id"]: item["input_contract"].get("annotation_sha256")
            for item in index["records"]
        }
        if index
        else {},
        cohort=cohort,
        model_version=predictor.version,
        evaluated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        wall_seconds=time.monotonic() - started,
        primary_split_sha256=split["sha256"],
        cache_rows_sha256=index.get("rows_sha256") if index else None,
        independent_group_note="External confidence intervals resample record groups; cross-record patient identity is unpublished and subject disjointness is unverified."
        if cohort == "external"
        else "Primary confidence intervals resample audited independent patient groups.",
        transfer_note="Missing clinical fields remain missing; no external fitting/calibration/policy tuning."
        if cohort == "external"
        else None,
    )
    _write_predictions(output / "predictions.jsonl.gz", predicted)
    from .data import file_hash

    report["predictions_sha256"] = file_hash(output / "predictions.jsonl.gz")
    _write(output / "evaluation.json", report)
    return report


def audit_development(rows, events, config, split, index=None, output=None):
    """Describe completed DEVELOPMENT cache; availability is an oracle upper bound, not accuracy."""
    began = time.monotonic()
    if _digest({key: value for key, value in split.items() if key != "sha256"}) != split.get("sha256"):
        raise ValueError("Development audit requires an intact primary split")
    permitted = set(split["development_groups"])
    if not isinstance(rows, CompactRows):
        raise TypeError("Development audit requires the completed compact cache returned by read_cache")
    if {record["group_id"] for record in rows.records} != permitted or any(
        _group(event) not in permitted for event in events
    ):
        raise ValueError("Development audit cannot include held-out or omit development groups")
    if index is not None and (
        set(index["allowed_groups"]) != permitted
        or any(
            item["input_contract"] != split["record_contracts"].get(item["record"]["record_id"])
            for item in index["records"]
        )
    ):
        raise ValueError("Audit source identities differ from the frozen development cohort")
    metadata, records, categories = rows.metadata, rows.records, rows.categories
    values, names = rows.features, rows.names
    eligible = np.isin(metadata["label"], (0, 1))
    accepted = np.asarray(metadata["accepted"], dtype=bool)
    seconds = np.zeros(len(rows))
    record_indices = {}
    duplicates = 0
    for code, record in enumerate(records):
        indices = np.flatnonzero(metadata["record"] == code)
        indices = indices[np.argsort(metadata["timestamp"][indices], kind="stable")]
        times = metadata["timestamp"][indices]
        duplicates += int(np.sum(np.diff(times) == 0))
        ends = np.r_[times[1:], metadata["duration_seconds"][indices[-1]]]
        seconds[indices] = np.maximum(0.0, ends - times)
        record_indices[(record["patient_id"], record["record_id"])] = indices
    reasons = {
        kind: {
            str(reason or "unspecified"): float(seconds[mask & (metadata[kind] == code)].sum() / 3600)
            for code, reason in enumerate(categories[kind])
            if np.any(mask & (metadata[kind] == code))
        }
        for kind, mask in (("reason", ~accepted), ("label_reason", ~eligible))
    }
    feature_report = {}
    fit_eligible = eligible & accepted
    quantiles = [0.01, 0.25, 0.5, 0.75, 0.99]
    for column, name in enumerate(names):
        raw = values[:, column]
        finite = np.isfinite(raw)
        selected = raw[fit_eligible & finite]
        feature_report[name] = {
            "missing_all_fraction": float(np.mean(~finite)),
            "missing_accepted_eligible_fraction": float(np.mean(~finite[fit_eligible]))
            if fit_eligible.any()
            else None,
            "infinite_values": int(np.isinf(raw).sum()),
            "accepted_eligible_quantiles": np.quantile(selected, quantiles).tolist()
            if len(selected)
            else None,
        }
    gap, horizon = config["target"]["gap_seconds"], config["target"]["horizon_seconds"]
    event_details = []
    for event in events:
        indices = record_indices.get((event["patient_id"], event["record_id"]), np.array([], dtype=int))
        interval = indices[
            (metadata["timestamp"][indices] >= event["onset"] - horizon)
            & (metadata["timestamp"][indices] <= event["onset"] - gap)
        ]
        available = interval[accepted[interval] & eligible[interval]]
        event_details.append(
            {
                **event,
                "forecast_interval_windows": len(interval),
                "eligible_windows": int(eligible[interval].sum()),
                "available_eligible_windows": len(available),
                "available": bool(len(available)),
                "rejection_window_counts": dict(
                    Counter(
                        categories["reason"][int(metadata["reason"][i])] or "unspecified"
                        for i in interval
                        if not accepted[i]
                    )
                ),
                "annotation_exclusion_window_counts": dict(
                    Counter(
                        categories["label_reason"][int(metadata["label_reason"][i])] or "unspecified"
                        for i in interval
                        if not eligible[i]
                    )
                ),
            }
        )
    group_for_record = np.array([record["group_id"] for record in records])
    positive_groups = set(group_for_record[metadata["record"][fit_eligible & (metadata["label"] == 1)]])
    result = {
        "scope": "Development-only descriptive audit; no primary test signals or outcomes are read",
        "primary_split_sha256": split["sha256"],
        "development_cache_sha256": index.get("sha256") if index else None,
        "windows": len(rows),
        "independent_groups": len(permitted),
        "records": len(records),
        "accepted_windows": int(accepted.sum()),
        "accepted_eligible_windows": int(fit_eligible.sum()),
        "positive_windows": int(np.sum(metadata["label"] == 1)),
        "negative_windows": int(np.sum(metadata["label"] == 0)),
        "accepted_positive_windows": int(np.sum(fit_eligible & (metadata["label"] == 1))),
        "accepted_negative_windows": int(np.sum(fit_eligible & (metadata["label"] == 0))),
        "accepted_positive_groups": sorted(positive_groups),
        "eligible_event_groups": sorted({_group(event) for event in events}),
        "positive_prevalence_accepted_eligible": float(np.mean(metadata["label"][fit_eligible]))
        if fit_eligible.any()
        else None,
        "exposure_hours": {
            "monitored": float(seconds.sum() / 3600),
            "eligible_including_rejected": float(seconds[eligible].sum() / 3600),
            "available": float(seconds[accepted].sum() / 3600),
            "available_eligible": float(seconds[fit_eligible].sum() / 3600),
            "rejected": float(seconds[~accepted].sum() / 3600),
        },
        "rejection_hours": reasons["reason"],
        "annotation_exclusion_hours": reasons["label_reason"],
        "duplicate_decision_timestamps": duplicates,
        "feature_quantile_levels": quantiles,
        "features": feature_report,
        "oracle_availability": {
            "eligible_events": len(events),
            "events_with_available_eligible_decision": sum(event["available"] for event in event_details),
            "event_sensitivity_ceiling": sum(event["available"] for event in event_details) / len(events)
            if events
            else None,
            "events": event_details,
            "interpretation": "An oracle may choose any available eligible decision in the inclusive 120–600 second interval. This ignores alert persistence, cooldown, false warnings and learnability, so it is an upper bound, not an attainable performance claim; unavailable events remain in its denominator.",
        },
        "shape_cost": {
            "feature_columns": len(names),
            "feature_array_bytes": int(values.nbytes),
            "accepted_eligible_30s_rows": int(fit_eligible.sum()),
            "accepted_eligible_120s_rows": int(np.sum(fit_eligible & (metadata["timestamp"] % 120 == 0))),
            "full_model_matrix_bytes_upper_bound": int(fit_eligible.sum()) * 2 * len(names) * 8,
            "assumptions": "Dense float64 values plus missingness indicators; actual outer fitting matrices use a subset of groups/features and fit-only 120-second cadence except the exact current 30-second recipe. Fit duration requires a measured training benchmark, not shape alone.",
        },
        "wall_seconds": time.monotonic() - began,
    }
    if output:
        _write(output, result)
    return result
