"""Bound offline ECGFounder study, immutable pairs and complete comparison artifacts."""

import copy
import gzip
import hashlib
import io
import json
import math
import resource
import sys
import time
from collections import ChainMap, Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from .config import validate_config
from .ecgfounder import CHECKPOINT_BYTES, CHECKPOINT_SHA256
from .ecgfounder_compare import (
    BATCH_ROWS,
    VerifiedTables,
    calibration_summary,
    fit_calibrator,
    fit_nuisance_head,
    fit_null_heads,
    fit_primary_heads,
    fit_transforms,
    orientation_labels,
    pair_row_masses,
    score_rows,
)
from .ecgfounder_probe import read_bound_manifest, verify_files
from .ecgfounder_runner import (
    CHECKPOINT_PATH,
    CYCLE,
    ORIGINAL_DECLARATION_SHA256,
    ROLE_NAMES,
    SPEC_SHA256,
    _canonical_sha,
    _cost,
    _path,
    _roles,
    _storage,
    coverage_cell,
)
from .ecgfounder_tables import REASONS, identity_digest
from .evaluation import evaluate_stream, policy_constraints
from .features import FEATURE_VERSION
from .investigation import CACHE_VERSION, CompactRows, _digest, _write, policy_search_identity
from .precursor import (
    LINEAR_SCHEMA,
    PRESETS,
    PRIMARY_POLICY,
    EventPair,
    _person_rows,
    _same_numeric_json,
    compact_cache_binding,
    compare_person_counts,
    freeze_policy_frontiers,
    fresh_output,
    negative_drift_pairs,
    source_manifest,
)
from .progress import Progress

EXPECTED_ROWS, EXPECTED_PEOPLE, EXPECTED_RECORDS, EXPECTED_EVENTS = 629444, 121, 132, 241
BRIDGE_HELPERS = tuple(
    "src/cardiotwin/" + name + ".py"
    for name in ("investigation", "precursor", "modeling", "features", "labels", "data")
)
REQUIRED_SOURCE = (
    *BRIDGE_HELPERS,
    "src/cardiotwin/ecgfounder_study.py",
    "src/cardiotwin/ecgfounder_compare.py",
    "src/cardiotwin/ecgfounder_runner.py",
    "src/cardiotwin/ecgfounder_tables.py",
    "src/cardiotwin/ecgfounder.py",
    "src/cardiotwin/config.py",
    "src/cardiotwin/progress.py",
    "pyproject.toml",
    "uv.lock",
    "scripts/run_ecgfounder_study.py",
)
BRIDGE_PINS = {
    "reports/ml-investigation/next-cycle/precursor-review-20261012-provenance/receipt.json": "9e8e561d2817055b95318af3a0c01ae45aa680a0aa313ce184dade5ca208755a",
    "reports/ml-investigation/next-cycle/precursor-review-20261012-provenance/retrospective-source-validation.json": "8659522a0ddd1f38fd911fc132fbaabe217a8fd07575e345c90e1e512d0b6809",
    CYCLE
    + "/repair-panel-20261014-b/study-implementation/parent-source-bridge-ruling.md": "c3ac7b4c6dd2dd28970a177a6eb1df713e0620a2ac1e12ac002ff5f730a5c72a",
    CYCLE
    + "/repair-panel-20261014-b/study-implementation/current-source-bridge-verification.json": "6cf50ff41340d3c1790d1e15842361b3a5b3e8d2493e3411bfd455f536909394",
}
# The old audit omitted these digests; prospectively preserve its saved artifact bytes.
ORIGINAL_AUDIT_PINS = {
    "metadata_audit": (
        "reports/ml-investigation/next-cycle/precursor-diagnostic-20261012-cadence-fix/metadata-audit.json",
        "eebb940d184e12284df8827817cbff38e176a4d6d50bc9cecc6a2befc52a007a",
    ),
    "pair_support": (
        "reports/ml-investigation/next-cycle/precursor-diagnostic-20261012-cadence-fix/pair-support.json",
        "e8078fbb7564fd0ce83ba72469a1edf1b34f63c94ed4923c3dfa5f8aac59e4a0",
    ),
}


class IndexedRow(Mapping):
    def __init__(self, row, index):
        self.row, self.index = row, index

    def __getitem__(self, key):
        return self.index if key == "row_index" else self.row[key]

    def __iter__(self):
        yield "row_index"
        yield from self.row

    def __len__(self):
        return len(self.row) + 1


class IndexedRows(Sequence):
    def __init__(self, rows):
        self.original = rows

    def __len__(self):
        return len(self.original)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return IndexedRow(self.original[index], int(index))


class FeatureRows(Sequence):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]["features"]


@dataclass
class PreparedStudy:
    rows: IndexedRows
    tables: VerifiedTables
    events: list
    index: dict
    roles: list
    pairs: dict
    original_support: dict
    config: dict
    receipt: dict
    root: Path
    files: dict
    source: dict
    payload: dict
    storage_check: object
    closed: bool = False

    @property
    def features(self):
        return FeatureRows(self.rows)

    def close(self):
        if not self.closed:
            self.tables.close()
            for array in (self.rows.original.features, self.rows.original.metadata):
                array._mmap.close()
            self.closed = True

    def __enter__(self):
        if self.closed:
            raise ValueError("Prepared study is closed")
        return self

    def __exit__(self, *args):
        self.close()


def _verify(bindings, root, guard):
    for name, checksum in bindings.items():
        guard()
        verify_files({name: checksum}, root=root)
    guard()


def _rss_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)


def _resources(storage_check):
    rss = _rss_bytes()
    if not math.isfinite(rss) or not 0 <= rss <= 10_000_000_000:
        raise RuntimeError("Study CPU resource cap exceeded")
    return dict(device="cpu", peak_rss_bytes=rss, storage=_storage(storage_check))


def _bound(payload, key, root):
    name = payload[key]
    if name not in payload["files_sha256"]:
        raise ValueError("Study required input is not hash bound: " + key)
    return read_bound_manifest(_path(root, name), payload["files_sha256"][name])


def _bridge(payload, root, original, source):
    if {
        payload[k]
        for k in ("source_validation_receipt", "source_validation", "bridge_ruling", "bridge_evidence")
    } != set(BRIDGE_PINS) or any(
        payload["files_sha256"].get(name) != checksum for name, checksum in BRIDGE_PINS.items()
    ):
        raise ValueError("Pinned parent current-source validation bridge required")
    evidence = _bound(payload, "bridge_evidence", root)
    prior = _bound(payload, "source_validation_receipt", root)
    validation = _bound(payload, "source_validation", root)
    ruling = payload["bridge_ruling"]
    if ruling not in payload["files_sha256"] or not _path(root, ruling).read_text().strip():
        raise ValueError("Explicit parent source bridge ruling required")
    files = payload["files_sha256"]
    helpers = evidence["current_helper_hashes_match_prior_validation"]
    if (
        evidence["status"] != "verified_current_source_identity_bridge_only"
        or evidence["source_receipt_path"] != payload["source_validation_receipt"]
        or evidence["source_receipt_sha256"] != files[payload["source_validation_receipt"]]
        or evidence["validation_path"] != payload["source_validation"]
        or evidence["validation_sha256"] != files[payload["source_validation"]]
        or evidence["original_cycle_declaration_sha256"] != ORIGINAL_DECLARATION_SHA256
        or evidence["compact_binding_exact_match"] is not True
        or prior["status"] != "complete"
        or prior["artifacts"].get(Path(payload["source_validation"]).name)
        != files[payload["source_validation"]]
        or validation["status"] != "passed"
        or validation["binding"] != original["compact_binding"]
        or validation["rows_checked"] != EXPECTED_ROWS
        or validation["feature_values_checked"]
        != EXPECTED_ROWS * len(original["compact_binding"]["feature_schema"])
        or evidence["rows_checked_in_prior_validation"] != validation["rows_checked"]
        or evidence["feature_values_checked_in_prior_validation"] != validation["feature_values_checked"]
        or set(helpers) != set(BRIDGE_HELPERS)
        or any(
            source.get(n) != helpers[n] or prior["artifacts"].get("source-package/" + n) != helpers[n]
            for n in BRIDGE_HELPERS
        )
        or any(
            value.get("historical_fit_time_identity_verified") is not False
            for value in (evidence, prior, validation)
        )
    ):
        raise ValueError("Current source/compact validation bridge differs")
    return prior


def _load_original(payload, root, original, source):
    cache = _path(root, payload["cache"])
    for name in ("index.json", "compact.json", "features.npy", "metadata.npy"):
        key = str((cache / name).relative_to(root))
        if key not in payload["files_sha256"]:
            raise ValueError("Required compact input missing")
    binding = compact_cache_binding(cache, expected=original["compact_binding"])
    config = _bound(payload, "config", root)
    validate_config(config)
    index = json.loads((cache / "index.json").read_text())
    compact = json.loads((cache / "compact.json").read_text())
    if (
        index["cache_version"] != CACHE_VERSION
        or index["feature_version"] != FEATURE_VERSION
        or index["config_sha256"] != _digest(index["config"])
        or config != index["config"]
        or index["sha256"] != _digest({k: v for k, v in index.items() if k != "sha256"})
        or index["row_count"] != EXPECTED_ROWS
        or len(index["records"]) != EXPECTED_RECORDS
        or len(index["allowed_groups"]) != EXPECTED_PEOPLE
        or len(set(index["allowed_groups"])) != EXPECTED_PEOPLE
        or len(index["events"]) != EXPECTED_EVENTS
        or not set(LINEAR_SCHEMA) <= set(binding["feature_schema"])
        or set(compact["categories"]) != {"reason", "label_reason"}
        or any(not values for values in compact["categories"].values())
    ):
        raise ValueError("Original cache/config/version/schema/count identity differs")
    split = _bound(payload, "split", root)
    if (
        split["seed"] != 42
        or set(split["development_groups"]) != set(index["allowed_groups"])
        or set(split["development_groups"]) & set(split["test_groups"])
    ):
        raise ValueError("Original development split differs")
    mapping = {}
    reverse = {}
    records = {}
    for entry in compact["records"]:
        rid, pid, group = (entry[k] for k in ("record_id", "patient_id", "group_id"))
        if (
            rid in records
            or not all(isinstance(v, str) and v for v in (rid, pid, group))
            or mapping.setdefault(pid, group) != group
            or reverse.setdefault(group, pid) != pid
            or index["group_ids"].get(pid) != group
            or split["group_ids"].get(pid) != group
        ):
            raise ValueError("Original patient/group/record mapping differs")
        records[rid] = (pid, group)
    if (
        len(records) != EXPECTED_RECORDS
        or len(mapping) != EXPECTED_PEOPLE
        or set(reverse) != set(index["allowed_groups"])
        or {e["record"]["record_id"] for e in index["records"]} != set(records)
        or any(records[e["record"]["record_id"]][0] != e["record"]["patient_id"] for e in index["records"])
    ):
        raise ValueError("Original record membership differs")
    for field, value in dict(
        history_seconds=600,
        horizon_seconds=600,
        gap_seconds=120,
        cadence_seconds=30,
        min_episode_seconds=30,
        washout_seconds=600,
    ).items():
        if config["target"][field] != value:
            raise ValueError("Fixed forecasting target differs")
    return cache, binding, index, compact, config, split, records


def _coverage_authority(payload, root, original, source, binding, compact):
    extraction = _bound(payload, "extraction_declaration", root)
    files = payload["files_sha256"]
    if (
        extraction["version"] != "ecgfounder-full-extraction-v1"
        or extraction["compact_binding"] != binding
        or extraction["cache"] != payload["cache"]
        or extraction["roles"] != payload["roles"]
        or extraction["raw_manifest"] != payload["raw_manifest"]
        or any(files.get(n) != v for n, v in extraction["files_sha256"].items())
        or any(source.get(n) != v for n, v in extraction["source_sha256"].items())
        or extraction["files_sha256"].get(CHECKPOINT_PATH) != CHECKPOINT_SHA256
        or any(extraction["files_sha256"].get(n) != v for n, v in original["files_sha256"].items())
    ):
        raise ValueError("Extraction declaration/input/source authority differs")
    raw = _bound(payload, "raw_manifest", root)
    if not raw["files_sha256"]:
        raise ValueError("Original raw manifest missing")
    runner = _bound(payload, "runner_receipt", root)
    before = _bound(payload, "runner_preflight", root)
    coverage = _bound(payload, "coverage_receipt", root)
    table_path = str((_path(root, payload["tables"]) / "receipt.json").relative_to(root))
    if table_path not in files:
        raise ValueError("Writer receipt must be independently bound")
    writer = read_bound_manifest(_path(root, table_path), files[table_path])
    checkpoint = dict(path=CHECKPOINT_PATH, sha256=CHECKPOINT_SHA256, bytes=CHECKPOINT_BYTES)
    identity = extraction["metadata_identity_sha256"]
    if (
        runner["status"] != "complete"
        or runner["coverage_passed"] is not True
        or runner["tables_complete"] is not True
        or runner["rows_completed"] != EXPECTED_ROWS
        or runner["core"]["rows_completed"] != EXPECTED_ROWS
        or runner["core"]["status"] != "complete"
        or runner["core"]["record_complete"] is not True
        or runner["checkpoint"] != checkpoint
        or before["checkpoint"] != checkpoint
        or before["status"] != "preflight_passed_not_executed"
        or before["declaration_sha256"] != files[payload["extraction_declaration"]]
        or (before["row_count"], before["people"], before["records"])
        != (EXPECTED_ROWS, EXPECTED_PEOPLE, EXPECTED_RECORDS)
        or before["identity_sha256"] != identity
        or coverage["identity_sha256"] != identity
        or coverage["rows"] != EXPECTED_ROWS
        or coverage["passed"] is not True
        or runner["table_receipt_sha256"] != files[table_path]
        or coverage["table_receipt_sha256"] != files[table_path]
        or runner["coverage_sha256"] != files[payload["coverage_receipt"]]
        or writer["status"] != "complete"
        or writer["complete"] is not True
        or len(coverage["cells"]) != 15
        or {(c["fold"], c["role"]) for c in coverage["cells"]}
        != {(f, n) for f in range(3) for n in ROLE_NAMES}
        or any(
            c
            != dict(
                fold=c["fold"],
                role=c["role"],
                **coverage_cell(c["total_rows"], c["original_accepted"], c["common_available"]),
            )
            or c["passed"] is not True
            for c in coverage["cells"]
        )
    ):
        raise ValueError("COMPLETE runner/raw-post/writer/coverage authority required")
    contract = dict(
        row_count=EXPECTED_ROWS,
        identity_sha256=identity,
        records=compact["records"],
        files_sha256={
            **extraction["files_sha256"],
            payload["extraction_declaration"]: files[payload["extraction_declaration"]],
        },
        source_sha256=extraction["source_sha256"],
    )
    if (
        runner.get("preflight_sha256") != files[payload["runner_preflight"]]
        or runner.get("hashes_after")
        != dict(files_sha256={**contract["files_sha256"], **contract["source_sha256"]}, hash_errors={})
        or runner.get("raw_hashes_after") != dict(files_sha256=raw["files_sha256"], hash_errors={})
    ):
        raise ValueError("Complete runner raw/source postcheck authority differs")
    return contract, coverage, files[table_path]


def _original_audit(payload, root, prior, index, config, binding, records):
    if any(
        payload[key] != name or payload["files_sha256"].get(name) != checksum
        for key, (name, checksum) in ORIGINAL_AUDIT_PINS.items()
    ):
        raise ValueError("Exact original saved metadata/support artifact identities required")
    audit = _bound(payload, "audit_receipt", root)
    metadata = _bound(payload, "metadata_audit", root)
    roles = _bound(payload, "roles", root)
    support = _bound(payload, "pair_support", root)
    files = payload["files_sha256"]
    identity = prior["identity"]
    if (
        audit["status"] != "passed"
        or audit["metadata_status"] != "passed"
        or audit["pair_support_status"] != "passed"
        or audit["failed_criteria"]
        or metadata["status"] != "passed"
        or support["status"] != "passed"
        or support["failed_criteria"]
        or identity["diagnostic_sha256"] != files[payload["audit_receipt"]]
        or audit["cache_index_sha256"] != binding["index_sha256"]
        or audit["cache_identity"] != index["sha256"]
        or audit["config_sha256"] != _digest(config)
        or audit["split_sha256"] != files[payload["split"]]
        or roles["cache_index_sha256"] != binding["index_sha256"]
        or audit["roles_sha256"] != _digest(roles["roles"])
        or audit["pairs_sha256"] != files[payload["pairs"]]
        or (audit["rows"], audit["people"], audit["events"])
        != (EXPECTED_ROWS, EXPECTED_PEOPLE, EXPECTED_EVENTS)
        or any(
            identity[k] != audit[k]
            for k in ("cache_index_sha256", "config_sha256", "split_sha256", "roles_sha256", "pairs_sha256")
        )
    ):
        raise ValueError("Passed original metadata/roles/pairs audit identity differs")
    for name, checksum in metadata["metadata_sha256"].items():
        path = _path(root, payload["metadata_root"] + "/" + name)
        if files.get(str(path.relative_to(root))) != checksum:
            raise ValueError("Original metadata identity is not bound")
    for name in ("src/cardiotwin/data.py", "src/cardiotwin/features.py", "src/cardiotwin/labels.py"):
        if audit["source"]["files"].get(name) != payload["source_sha256"].get(name):
            raise ValueError("Original audited extraction helpers differ")
    membership, _ = _roles(roles["roles"], set(index["allowed_groups"]))
    if Counter(g for role in roles["roles"] for g in role["outer"]) != Counter(index["allowed_groups"]):
        raise ValueError("Original outer partitions must cover every person once")
    events = {}
    for e in index["events"]:
        key = (e["group_id"], e["patient_id"], e["record_id"], float(e["onset"]))
        if (
            e["record_id"] not in records
            or records[e["record_id"]] != (e["patient_id"], e["group_id"])
            or not math.isfinite(key[-1])
            or key[-1] < 0
            or key in events
        ):
            raise ValueError("Immutable original event identity differs")
        events[key] = e
    return audit, roles["roles"], support, membership, events


def _saved_pairs(payload, root, rows, roles, support, membership, events, guard):
    result = {f: {n: [] for n in ROLE_NAMES} for f in range(3)}
    seen = set()
    negatives = set()
    with _path(root, payload["pairs"]).open() as handle:
        for line in handle:
            guard()
            value = json.loads(line)
            fold, name = value.pop("fold"), value.pop("role")
            pair = EventPair(**value)
            if (
                type(fold) is not int
                or fold not in result
                or name not in ROLE_NAMES
                or membership.get((fold, pair.group_id)) != name
                or any(
                    type(i) is not int or not 0 <= i < len(rows)
                    for i in (pair.positive_index, pair.negative_index)
                )
            ):
                raise ValueError("Saved pair fold/role/global index differs")
            positive, negative = rows[pair.positive_index], rows[pair.negative_index]
            event = (pair.group_id, pair.patient_id, pair.record_id, pair.onset)
            key = (fold, name, event)
            if (
                key in seen
                or (fold, name, pair.negative_index) in negatives
                or event not in events
                or any(
                    r[k] != getattr(pair, k)
                    for r in (positive, negative)
                    for k in ("group_id", "patient_id", "record_id")
                )
                or not positive["accepted"]
                or not negative["accepted"]
                or positive["label"] != 1
                or negative["label"] != 0
                or not 120 <= pair.onset - positive["timestamp"] <= 600
                or not 1800 <= pair.onset - negative["timestamp"] <= 3600
                or negative["timestamp"] > positive["timestamp"] - 600
            ):
                raise ValueError("Saved pair identity/label/time/nonoverlap differs")
            seen.add(key)
            negatives.add((fold, name, pair.negative_index))
            result[fold][name].append(pair)
    if sorted(f["fold"] for f in support["folds"]) != [0, 1, 2]:
        raise ValueError("Original support folds differ")
    for f in support["folds"]:
        for name in ROLE_NAMES:
            cell = f["roles"][name]
            chosen = result[f["fold"]][name]
            people = Counter(p.group_id for p in chosen)
            eligible = [e for e in events.values() if membership[f["fold"], e["group_id"]] == name]
            if (
                cell["paired_events"] != len(chosen)
                or cell["paired_people"] != len(people)
                or cell["pair_counts_by_person"] != dict(people)
                or cell["eligible_events"] != len(eligible)
                or cell["positive_people"] != len({e["group_id"] for e in eligible})
                or not 0 <= len(chosen) <= cell["observable_events"] <= cell["eligible_events"]
                or not 0 <= len(people) <= cell["positive_people"]
            ):
                raise ValueError("Original saved support denominators/counts differ")
    return result


def prepare_study(declaration, expected_sha256, *, root, storage_check, guard):
    """Validate authority; caller guard enforces the original bootstrap/work/calendar limits."""
    root = Path(root).resolve()
    guard()
    _resources(storage_check)
    payload = read_bound_manifest(Path(declaration), expected_sha256)
    if not Path(declaration).resolve().is_relative_to(root) or payload["version"] != "ecgfounder-study-v1":
        raise ValueError("Prospective study declaration within root required")
    stages = payload["allowed_stages"]
    if (
        not stages
        or len(stages) != len(set(stages))
        or "preflight" not in stages
        or not set(stages) <= {"preflight", "support", "run", "replay"}
    ):
        raise ValueError("Explicit conditional study stages required")
    files, source = payload["files_sha256"], payload["source_sha256"]
    _verify(files, root, guard)
    _verify(source, root, guard)
    current = source_manifest()["files"]
    if any(source.get(n) != v for n, v in current.items()) or any(n not in source for n in REQUIRED_SOURCE):
        raise ValueError("Current executed study source is omitted or changed")
    original_path = CYCLE + "/declaration-binding.json"
    spec_path = CYCLE + "/spec.md"
    if files.get(original_path) != ORIGINAL_DECLARATION_SHA256 or files.get(spec_path) != SPEC_SHA256:
        raise ValueError("Frozen original scientific declaration differs")
    original = read_bound_manifest(_path(root, original_path), ORIGINAL_DECLARATION_SHA256)
    if any(files.get(n) != v for n, v in original["files_sha256"].items()):
        raise ValueError("Inherited original input differs")
    if (
        files.get(CHECKPOINT_PATH) != CHECKPOINT_SHA256
        or _path(root, CHECKPOINT_PATH).stat().st_size != CHECKPOINT_BYTES
    ):
        raise ValueError("Pinned checkpoint input required")
    prior = _bridge(payload, root, original, source)
    cache, binding, index, compact, config, split, records = _load_original(payload, root, original, source)
    contract, coverage, table_receipt_sha = _coverage_authority(
        payload, root, original, source, binding, compact
    )
    cost = _bound(payload, "cost_receipt", root)
    review = _bound(payload, "cost_review", root)
    if review["cost_sha256"] != files[payload["cost_receipt"]]:
        raise ValueError("Cost review identity differs")
    cost_result = _cost(cost, review, files, source, binding, datetime.now(timezone.utc))
    resources = _resources(storage_check)
    roles = _bound(payload, "roles", root)["roles"]
    membership, _ = _roles(roles, set(index["allowed_groups"]))
    original_rows = None
    tables = None
    try:
        original_rows = CompactRows(cache, compact)
        rows = IndexedRows(original_rows)
        dtype = np.dtype(
            [
                ("record", "<i4"),
                ("timestamp", "<f8"),
                ("duration_seconds", "<f8"),
                ("accepted", "i1"),
                ("label", "i1"),
                ("reason", "<i2"),
                ("label_reason", "<i2"),
            ]
        )
        if (
            original_rows.metadata.dtype != dtype
            or original_rows.features.dtype != np.dtype("<f8")
            or original_rows.features.flags.writeable
            or original_rows.metadata.flags.writeable
        ):
            raise ValueError("Read-only original numeric array contract differs")

        def guarded_rows():
            for row in rows:
                guard()
                yield row

        tables = VerifiedTables.open(
            _path(root, payload["tables"]),
            guarded_rows(),
            contract,
            root=root,
            receipt_sha256=table_receipt_sha,
            compact_path=str((cache / "compact.json").relative_to(root)),
            checkpoint_path=CHECKPOINT_PATH,
            feature_schema=binding["feature_schema"],
        )
        counts = {(f, n): [0, 0, 0] for f in range(3) for n in ROLE_NAMES}
        previous = {}
        with Progress("ecgfounder.study_identity_coverage", total=EXPECTED_ROWS, device="cpu") as progress:
            for i, row in enumerate(rows):
                guard()
                rid = row["record_id"]
                t = row["timestamp"]
                meta = original_rows.metadata[i]
                if (
                    not np.isfinite(t)
                    or t <= previous.get(rid, -math.inf)
                    or row["group_id"] not in index["allowed_groups"]
                    or any(
                        meta[k] < 0 or meta[k] >= len(compact["categories"][k])
                        for k in ("reason", "label_reason")
                    )
                    or meta["accepted"] not in (0, 1)
                ):
                    raise ValueError("Original compact row metadata/chronology differs")
                previous[rid] = t
                for f in range(3):
                    cell = counts[f, membership[f, row["group_id"]]]
                    cell[0] += 1
                    cell[1] += row["accepted"]
                    cell[2] += bool(tables.rows[i]["available"])
                progress.update(i + 1)
        if identity_digest(guarded_rows()) != contract["identity_sha256"]:
            raise ValueError("Original global row identity differs")
        for cell in coverage["cells"]:
            if cell != dict(
                fold=cell["fold"], role=cell["role"], **coverage_cell(*counts[cell["fold"], cell["role"]])
            ):
                raise ValueError("Original accepted coverage denominator differs")
        audit, roles, support, membership, events = _original_audit(
            payload, root, prior, index, config, binding, records
        )
        for meta in original_rows.metadata:
            guard()
            if meta["label"] not in (-1, 0, 1):
                raise ValueError("Original label code differs")
        pairs = _saved_pairs(payload, root, rows, roles, support, membership, events, guard)
        all_files = {**files, str(Path(declaration).resolve().relative_to(root)): expected_sha256}
        _verify(all_files, root, guard)
        _verify(source, root, guard)
        resources = _resources(storage_check)
        receipt = dict(
            status="prepared",
            declaration_sha256=expected_sha256,
            files_sha256=all_files,
            source_sha256=source,
            compact_binding=binding,
            rows=EXPECTED_ROWS,
            people=EXPECTED_PEOPLE,
            records=EXPECTED_RECORDS,
            events=EXPECTED_EVENTS,
            identity_sha256=contract["identity_sha256"],
            table_receipt_sha256=table_receipt_sha,
            runner_receipt_sha256=files[payload["runner_receipt"]],
            coverage_sha256=files[payload["coverage_receipt"]],
            cost=cost_result,
            storage=resources["storage"],
            resources=resources,
            hash_validation=dict(
                whole_table_sha256_passes=2,
                bytes=2
                * sum(
                    (_path(root, payload["tables"]) / n).stat().st_size
                    for n in ("delta.npy", "nuisance.npy", "rows.npy")
                ),
            ),
            allowed_stages=stages,
            historical_fit_time_identity_verified=False,
            fits_started=0,
            full_run_eligible=False,
        )
        return PreparedStudy(
            rows,
            tables,
            index["events"],
            index,
            roles,
            pairs,
            support,
            config,
            receipt,
            root,
            all_files,
            source,
            payload,
            storage_check,
        )
    except BaseException:
        if tables is not None:
            tables.close()
        if original_rows is not None:
            for array in (original_rows.features, original_rows.metadata):
                array._mmap.close()
        raise


def _verify_tables(prepared, guard):
    table_root = _path(prepared.root, prepared.payload["tables"])
    name = str((table_root / "receipt.json").relative_to(prepared.root))
    receipt = read_bound_manifest(table_root / "receipt.json", prepared.files[name])
    bindings = {
        str((table_root / name).relative_to(prepared.root)): value
        for name, value in receipt["tables_sha256"].items()
    }
    _verify(bindings, prepared.root, guard)
    return dict(
        whole_table_sha256_passes=1, bytes=sum(_path(prepared.root, n).stat().st_size for n in bindings)
    )


def audit_common_support(prepared, *, guard):
    """Filter only saved pairs; original observable/event/person denominators never shrink."""
    if prepared.closed or prepared.tables._closed or "support" not in prepared.receipt["allowed_stages"]:
        raise ValueError("Complete open preparation and declared support stage required")
    _resources(prepared.storage_check)
    _verify(prepared.files, prepared.root, guard)
    _verify(prepared.source, prepared.root, guard)
    hashes_before = _verify_tables(prepared, guard)
    cells = []
    failed = []
    surviving = {f: {n: [] for n in ROLE_NAMES} for f in range(3)}
    original = {f["fold"]: f["roles"] for f in prepared.original_support["folds"]}
    classes = {(f, n): Counter() for f in range(3) for n in ("calibration", "policy")}
    membership, _ = _roles(prepared.roles, set(prepared.index["allowed_groups"]))
    for i, row in enumerate(prepared.rows):
        guard()
        if prepared.tables.rows[i]["available"] and row["label"] in (0, 1):
            for f in range(3):
                name = membership[f, row["group_id"]]
                if name in ("calibration", "policy"):
                    classes[f, name][row["label"]] += 1
    for f in range(3):
        for name in ROLE_NAMES:
            guard()
            excluded = []
            for pair in prepared.pairs[f][name]:
                missing = [
                    i
                    for i in (pair.positive_index, pair.negative_index)
                    if not prepared.tables.rows[i]["available"]
                ]
                if missing:
                    excluded.append(
                        dict(
                            **asdict(pair),
                            missing_members=[
                                dict(
                                    row_index=i,
                                    reasons={
                                        r: int(n)
                                        for r, n in zip(
                                            REASONS, prepared.tables.rows[i]["reason_counts"], strict=True
                                        )
                                        if n
                                    },
                                )
                                for i in missing
                            ],
                        )
                    )
                else:
                    surviving[f][name].append(pair)
            chosen = surviving[f][name]
            people = len({p.group_id for p in chosen})
            denominator = original[f][name]
            cell = dict(
                fold=f,
                role=name,
                original_pairs=len(prepared.pairs[f][name]),
                observable_events=denominator["observable_events"],
                positive_people=denominator["positive_people"],
                surviving_pairs=len(chosen),
                surviving_people=people,
                event_fraction=len(chosen) / denominator["observable_events"]
                if denominator["observable_events"]
                else 0.0,
                person_fraction=people / denominator["positive_people"]
                if denominator["positive_people"]
                else 0.0,
                excluded_pairs=excluded,
                surviving_original_indices=[[p.positive_index, p.negative_index] for p in chosen],
            )
            if name == "fit" and (
                5 * len(chosen) < 4 * denominator["observable_events"]
                or 5 * people < 4 * denominator["positive_people"]
                or not denominator["observable_events"]
                or not denominator["positive_people"]
            ):
                failed.append(f"fold_{f}_fit_original_support")
            if name == "stop" and people < 2:
                failed.append(f"fold_{f}_stop_pair_people")
            if name in ("calibration", "policy"):
                cell["natural_common_class_counts"] = dict(classes[f, name])
                if set(classes[f, name]) != {0, 1}:
                    failed.append(f"fold_{f}_{name}_classes")
            cells.append(cell)
    _verify(prepared.files, prepared.root, guard)
    _verify(prepared.source, prepared.root, guard)
    hashes_after = _verify_tables(prepared, guard)
    resources = _resources(prepared.storage_check)
    return dict(
        status="passed" if not failed else "support_failed",
        failed_criteria=failed,
        cells=cells,
        surviving_pairs={f: {n: [asdict(p) for p in surviving[f][n]] for n in ROLE_NAMES} for f in range(3)},
        preparation_sha256=_canonical_sha(prepared.receipt),
        files_sha256=prepared.files,
        source_sha256=prepared.source,
        resources=resources,
        hash_validation={k: hashes_before[k] + hashes_after[k] for k in hashes_before},
        allowed_stages=prepared.receipt["allowed_stages"],
        historical_fit_time_identity_verified=False,
        fits_started=0,
        full_run_eligible=False,
    )


STUDY_ARMS = ("timing_pointwise", "timing_ecgfounder_delta_pointwise")
POLICY_NAMES = tuple(f"far_{far:g}_occupancy_{occupancy:g}" for far, occupancy in PRESETS) + (
    "legacy_bridge",
)


def _support_identity(support):
    # Runtime RSS can rise between identical support scans; its measurement is not scientific identity.
    return _canonical_sha({k: v for k, v in support.items() if k != "resources"})


def _role_indices(prepared, groups):
    groups = set(groups)
    selected = np.asarray([r[1] in groups for r in prepared.tables.records])
    return np.flatnonzero(selected[prepared.tables.rows["record_index"]])


class _CommonRows(Sequence):
    def __init__(self, prepared):
        self.prepared = prepared

    def __len__(self):
        return len(self.prepared.rows)

    def __getitem__(self, index):
        return ChainMap(
            {"accepted": bool(self.prepared.tables.rows[index]["available"])}, self.prepared.rows[index]
        )


def _pair_diagnostics(head, prepared, pairs, work):
    if not pairs:
        return dict(
            pairs=0,
            people=0,
            person_macro_concordance=None,
            person_macro_difference=None,
            person_macro_loss=None,
            person_differences={},
        )
    work()
    indices = np.asarray([p.positive_index for p in pairs] + [p.negative_index for p in pairs])
    chunks = []
    for start in range(0, len(indices), BATCH_ROWS):
        work()
        chunks.append(head.logits(prepared.features, prepared.tables, indices[start : start + BATCH_ROWS]))
    logits = np.concatenate(chunks)
    if not np.isfinite(logits).all():
        raise ValueError("Every diagnostic pair must have finite common-available raw scores")
    differences = logits[: len(pairs)] - logits[len(pairs) :]
    people = {}
    for pair, value in zip(pairs, differences, strict=True):
        people.setdefault(pair.group_id, []).append(float(value))
    return dict(
        pairs=len(pairs),
        people=len(people),
        person_differences=people,
        person_macro_difference=float(np.mean([np.mean(v) for v in people.values()])),
        person_macro_concordance=float(
            np.mean(
                [
                    np.mean(np.where(np.asarray(v) > 0, 1.0, np.where(np.asarray(v) == 0, 0.5, 0.0)))
                    for v in people.values()
                ]
            )
        ),
        person_macro_loss=float(
            np.mean([np.mean(np.logaddexp(0.0, -np.asarray(v))) for v in people.values()])
        ),
    )


def _identity_variance(head, prepared, indices, work):
    people = {}
    for start in range(0, len(indices), BATCH_ROWS):
        work()
        selected = indices[start : start + BATCH_ROWS]
        logits = head.logits(prepared.features, prepared.tables, selected)
        for index, logit in zip(selected, logits, strict=True):
            if not prepared.tables.rows[index]["available"]:
                continue
            if not math.isfinite(logit):
                raise ValueError("Nonfinite natural fit/stop identity score")
            group = prepared.rows[int(index)]["group_id"]
            cell = people.setdefault(group, dict(count=0, mean=0.0, m2=0.0))
            cell["count"] += 1
            delta = float(logit) - cell["mean"]
            cell["mean"] += delta / cell["count"]
            cell["m2"] += delta * (float(logit) - cell["mean"])
    result = [
        dict(group_id=g, count=v["count"], mean=v["mean"], population_variance=max(0.0, v["m2"] / v["count"]))
        for g, v in sorted(people.items())
    ]
    return dict(
        people=result,
        includes_unknown_labels=True,
        equal_person_within_variance=float(np.mean([v["population_variance"] for v in result]))
        if result
        else None,
        between_person_mean_variance=float(np.var([v["mean"] for v in result])) if result else None,
    )


def _artifact_bytes(output):
    return sum(p.stat().st_size for p in output.rglob("*") if p.is_file())


def _artifact_write(output, name, data, cap, storage_check):
    if _artifact_bytes(output) + len(data) > cap:
        raise RuntimeError("Declared study artifact byte cap exceeded")
    _storage(storage_check, len(data))
    path = output / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(data)
    temporary.replace(path)
    _resources(storage_check)
    return dict(path=name, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def _json_artifact(output, name, value, cap, storage_check):
    return _artifact_write(
        output, name, json.dumps(value, sort_keys=True, allow_nan=False).encode(), cap, storage_check
    )



def _report_artifact(output, name, value, cap, storage_check):
    data = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    return _artifact_write(output, name, gzip.compress(data, mtime=0), cap, storage_check)


def _read_report(path):
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def _package_manifest(head, calibrator, identity):
    width = dict(timing=46, candidate=1070, nuisance=52)[head.kind]
    model = head.model
    if (
        model.coef_.shape != (1, width)
        or not np.isfinite(model.coef_).all()
        or not np.isfinite(model.intercept_).all()
        or list(model.classes_) != [0, 1]
    ):
        raise ValueError("Fixed fitted head coefficient/class contract differs")
    parameters = model.get_params(deep=False)
    if parameters["C"] != 0.1 or parameters["random_state"] != 42 or parameters["fit_intercept"] is not False:
        raise ValueError("Fixed pointwise fitted parameters differ")
    transform = head.transforms.manifest()
    manifest = dict(
        version="ecgfounder-study-model-v1",
        identity=identity,
        kind=head.kind,
        scope=head.scope,
        allowed_groups=sorted(head.allowed_groups) if head.allowed_groups is not None else None,
        model_parameters=parameters,
        coefficients=model.coef_.tolist(),
        intercept=model.intercept_.tolist(),
        transform=transform,
        transform_sha256=head.transforms.identity_sha256(),
        calibration=calibration_summary(calibrator) if calibrator is not None else None,
    )
    if head.kind == "nuisance":
        manifest["nuisance_transform"] = dict(
            imputer_statistics=head.nuisance_imputer.statistics_.tolist(),
            **{
                "scaler_" + name.removesuffix("_"): getattr(head.nuisance_scaler, name).tolist()
                for name in ("mean_", "var_", "scale_")
            },
        )
    return manifest


def _package_artifact(output, name, head, calibrator, identity, cap, storage_check):
    manifest = _package_manifest(head, calibrator, identity)
    # Binary loading is reserved for replay after its externally bound exhaustive manifest verifies.
    buffer = io.BytesIO()
    joblib.dump(dict(head=head, calibrator=calibrator, manifest=manifest), buffer)
    artifact = _artifact_write(output, name, buffer.getvalue(), cap, storage_check)
    numeric = _json_artifact(output, name + ".json", manifest, cap, storage_check)
    return dict(**artifact, numeric_manifest=numeric["path"], numeric_manifest_sha256=numeric["sha256"])


def _checked_predictions(prepared, head, calibrator, indices, work):
    original = iter(indices)
    for prediction in score_rows(head, calibrator, prepared.rows, prepared.tables, indices):
        work()
        index = next(original, None)
        if index is None:
            raise ValueError("Extra scored original row")
        row = prepared.rows[int(index)]
        available = bool(prepared.tables.rows[index]["available"])
        expected = {k: row[k] for k in row if k not in ("features", "accepted")}
        if (
            any(prediction.get(k) != v for k, v in expected.items())
            or prediction.get("accepted") is not available
            or prediction.get("original_accepted") is not bool(row["accepted"])
        ):
            raise ValueError("Complete scored original metadata/availability differs")
        if available:
            if (
                not math.isfinite(prediction["raw_logit"])
                or not math.isfinite(prediction["risk"])
                or not 0 < prediction["risk"] < 1
            ):
                raise ValueError("Finite available original logit/risk required")
        elif prediction.get("raw_logit") is not None or prediction.get("risk") is not None:
            raise ValueError("Common abstention must retain null score/risk")
        yield prediction
    if next(original, None) is not None:
        raise ValueError("Incomplete scored original chronology")


def _prediction_artifact(output, name, predictions, cap, storage_check, work):
    path = output / name
    temporary = path.with_suffix(path.suffix + ".partial")
    count = 0
    original_accepted = common_available = 0
    with temporary.open("wb") as handle:
        batch = []
        for prediction in predictions:
            batch.append(json.dumps(prediction, allow_nan=False).encode() + b"\n")
            count += 1
            original_accepted += bool(prediction["original_accepted"])
            common_available += bool(prediction["accepted"])
            if len(batch) == BATCH_ROWS:
                work()
                block = gzip.compress(b"".join(batch), mtime=0)
                if _artifact_bytes(output) + len(block) > cap:
                    raise RuntimeError("Declared study artifact byte cap exceeded")
                _storage(storage_check, len(block))
                handle.write(block)
                handle.flush()
                batch = []
        if batch:
            block = gzip.compress(b"".join(batch), mtime=0)
            if _artifact_bytes(output) + len(block) > cap:
                raise RuntimeError("Declared study artifact byte cap exceeded")
            _storage(storage_check, len(block))
            handle.write(block)
    temporary.replace(path)
    _resources(storage_check)
    from .data import file_hash

    return dict(
        path=name,
        sha256=file_hash(path),
        bytes=path.stat().st_size,
        rows=count,
        availability=dict(
            original_accepted_rows=original_accepted,
            common_available_rows=common_available,
            newly_unavailable_rows=original_accepted - common_available,
        ),
    )


class _PredictionRows:
    def __init__(self, path, work):
        self.path, self.work = path, work

    def __iter__(self):
        with gzip.open(self.path, "rt") as handle:
            for line in handle:
                self.work()
                yield json.loads(line)


def _advancement(entries, comparisons):
    primary = comparisons[PRIMARY_POLICY]
    deltas = []
    for fold in range(3):
        arms = {v["arm"]: v for v in entries if v["fold"] == fold}
        values = [arms[n]["reports"][PRIMARY_POLICY]["event_sensitivity"] for n in STUDY_ARMS]
        deltas.append(values[1] - values[0] if all(v is not None for v in values) else None)
    candidate = [v for v in entries if v["arm"] == STUDY_ARMS[1]]
    failed = []
    if primary["sensitivity_delta"] is None or primary["sensitivity_delta"] <= 0:
        failed.append("nonpositive_pooled_sensitivity_delta")
    if sum(v is not None and v > 0 for v in deltas) < 2:
        failed.append("fewer_than_two_improved_folds")
    if any(v["calibration"]["raw_score_slope"] <= 0 for v in candidate):
        failed.append("nonpositive_candidate_calibration_slope")
    if any(not v["primary_constraints"]["passed"] for v in candidate):
        failed.append("outer_primary_constraints")
    direction = all(
        v["sensitivity_delta"] is not None and v["sensitivity_delta"] >= 0
        for v in primary["leave_one_person_out"]
    ) and all(
        comparisons[f"far_0.25_occupancy_{b:g}"]["sensitivity_delta"] is not None
        and comparisons[f"far_0.25_occupancy_{b:g}"]["sensitivity_delta"] > 0
        for b in (0.01, 0.05, 0.1)
    )
    low = primary["bootstrap"]["event_sensitivity_delta"]["low"]
    promising = not failed
    supported = promising and low is not None and low > 0 and direction
    return dict(
        decision="advance" if supported else "uncertain" if promising else "reject",
        failed_advancement_criteria=failed,
        promising_development_lead=promising,
        supported_development_improvement=supported,
        fold_sensitivity_deltas=deltas,
        direction_survives_person_and_preset_checks=direction,
        independent_replay_required=True,
        clinical_claim=False,
        confirmation="Reused development cohort; fresh independent confirmation required",
    )


def execute_study(prepared, support, output, *, guard, storage_check, deadline):
    """Execute the fixed declared study; the caller retains the original session hard watchdog."""
    output = fresh_output(output)
    started = time.monotonic()
    receipt = dict(
        status="running",
        completed_arms=0,
        models_completed=0,
        calibrators_completed=0,
        fit_groups_attempted=0,
        full_run_eligible=False,
        historical_fit_time_identity_verified=False,
    )
    _write(output / "receipt.json", receipt)
    progress = Progress("ecgfounder.study_execution", total=6, device="cpu")

    def work():
        guard()
        if not math.isfinite(deadline) or time.monotonic() >= deadline:
            raise TimeoutError("Original study execution work deadline exceeded")

    artifacts = {}

    def record(artifact):
        artifacts[artifact["path"]] = artifact["sha256"]
        if "numeric_manifest" in artifact:
            artifacts[artifact["numeric_manifest"]] = artifact["numeric_manifest_sha256"]
        return artifact

    try:
        work()
        _resources(storage_check)
        if (
            prepared.closed
            or prepared.tables._closed
            or "run" not in prepared.receipt["allowed_stages"]
            or "run" not in support["allowed_stages"]
        ):
            raise ValueError("Open preparation and explicit conditional run stage required")
        if prepared.config["alert"].get("policy_version") != "finite-validity-v1" or any(
            prepared.config["alert"].get(k) != v
            for k, v in dict(
                max_duration_seconds=600,
                rearm_seconds=60,
                max_burden_occupancy=0.05,
                max_eligible_negative_burden_occupancy=0.05,
            ).items()
        ):
            raise ValueError("Exact original finite warning policy required")
        if prepared.config != _bound(prepared.payload, "config", prepared.root):
            raise ValueError("Prepared configuration changed")
        if prepared.roles != _bound(prepared.payload, "roles", prepared.root)[
            "roles"
        ] or prepared.original_support != _bound(prepared.payload, "pair_support", prepared.root):
            raise ValueError("Prepared original role/support interpretation changed")
        cap = prepared.payload.get("artifact_byte_cap", 250_000_000)
        if type(cap) is not int or not 0 < cap <= 250_000_000:
            raise ValueError("Bounded diagnostic/model artifact allocation required")
        current = audit_common_support(prepared, guard=work)
        if current["status"] != "passed" or _support_identity(current) != _support_identity(support):
            raise ValueError("Passed original common-support identity changed")
        identity = dict(
            preparation_sha256=_canonical_sha(prepared.receipt),
            support_sha256=_support_identity(current),
            source_sha256=prepared.source,
            files_sha256=prepared.files,
            compact_binding=prepared.receipt["compact_binding"],
            seed=42,
            arms=list(STUDY_ARMS),
        )
        receipt["identity"] = identity
        record(_json_artifact(output, "preparation.json", prepared.receipt, cap, storage_check))
        record(_json_artifact(output, "support.json", current, cap, storage_check))
        for name, checksum in prepared.source.items():
            work()
            copied = record(
                _artifact_write(
                    output,
                    "source-package/" + name,
                    _path(prepared.root, name).read_bytes(),
                    cap,
                    storage_check,
                )
            )
            if copied["sha256"] != checksum:
                raise ValueError("Copied study source differs from its required digest")
        record(_json_artifact(output, "source-manifest.json", prepared.source, cap, storage_check))
        record(_json_artifact(output, "config.json", prepared.config, cap, storage_check))
        record(_json_artifact(output, "roles.json", prepared.roles, cap, storage_check))
        record(_json_artifact(output, "events.json", prepared.events, cap, storage_check))
        ready = []
        diagnostics = []

        def checkpoint():
            receipt["elapsed_seconds"] = time.monotonic() - started
            receipt["resources"] = _resources(storage_check)
            record(_json_artifact(output, "checkpoint.json", receipt, cap, storage_check))
            progress.update(
                receipt["completed_arms"],
                force=True,
                models_completed=receipt["models_completed"],
                calibrators_completed=receipt["calibrators_completed"],
                phase=receipt.get("phase"),
            )

        # Every primary calibration and policy frontier across all folds freezes before outer access.
        for role in prepared.roles:
            work()
            fold = role["fold"]
            receipt["phase"] = f"fold_{fold}_fixed_fit_controls"
            chosen = {n: [EventPair(**v) for v in current["surviving_pairs"][fold][n]] for n in ROLE_NAMES}
            receipt["fit_groups_attempted"] += 1
            transforms = fit_transforms(prepared.features, prepared.tables, chosen["fit"])
            heads = fit_primary_heads(prepared.features, prepared.tables, chosen["fit"], transforms)
            if set(heads) != set(STUDY_ARMS):
                raise ValueError("Both fixed primary heads required")
            receipt["models_completed"] += 2
            checkpoint()
            diag = dict(fold=fold, identity=identity, roles={}, nulls=[], nuisance={})
            for name in ("fit", "stop"):
                indices = _role_indices(prepared, role[name])
                drift = negative_drift_pairs(_CommonRows(prepared), set(role[name]))
                diag["roles"][name] = {
                    arm: dict(
                        event_pairs=_pair_diagnostics(head, prepared, chosen[name], work),
                        negative_drift=_pair_diagnostics(head, prepared, drift, work),
                        identity=_identity_variance(head, prepared, indices, work),
                    )
                    for arm, head in heads.items()
                }
            seeds = []
            for seed, orientation, head in fit_null_heads(
                prepared.features, prepared.tables, chosen["fit"], transforms, stop_groups=role["stop"]
            ):
                work()
                seeds.append(seed)
                receipt["models_completed"] += 1
                package = record(
                    _package_artifact(
                        output, f"fold-{fold}-null-{seed}.joblib", head, None, identity, cap, storage_check
                    )
                )
                diag["nulls"].append(
                    dict(
                        seed=seed,
                        orientation=orientation,
                        stop=_pair_diagnostics(head, prepared, chosen["stop"], work),
                        model=package,
                    )
                )
                progress.update(models_completed=receipt["models_completed"], fold=fold, null_seed=seed)
            if seeds != list(range(101, 120)):
                raise ValueError("Exactly19 fixed null orientations required")
            work()
            nuisance = fit_nuisance_head(
                prepared.features, prepared.tables, chosen["fit"], transforms, stop_groups=role["stop"]
            )
            receipt["models_completed"] += 1
            diag["nuisance"] = dict(
                model=record(
                    _package_artifact(
                        output, f"fold-{fold}-nuisance.joblib", nuisance, None, identity, cap, storage_check
                    )
                ),
                **{n: _pair_diagnostics(nuisance, prepared, chosen[n], work) for n in ("fit", "stop")},
            )
            real = diag["roles"]["stop"][STUDY_ARMS[1]]["event_pairs"]["person_macro_concordance"]
            base = diag["roles"]["stop"][STUDY_ARMS[0]]["event_pairs"]["person_macro_concordance"]
            control = diag["nuisance"]["stop"]["person_macro_concordance"]
            diag["nuisance_equals_or_exceeds_candidate_stop_improvement"] = (
                all(v is not None for v in (real, base, control)) and control - base >= real - base
            )
            diagnostics.append(diag)
            record(_json_artifact(output, f"fold-{fold}-diagnostics.json", diag, cap, storage_check))
            for arm in STUDY_ARMS:
                work()
                receipt["phase"] = f"fold_{fold}_{arm}_calibration_policy"
                head = heads[arm]
                indices = _role_indices(prepared, role["calibration"])
                indices = np.asarray(
                    [
                        i
                        for i in indices
                        if prepared.tables.rows[i]["available"] and prepared.rows[int(i)]["label"] in (0, 1)
                    ],
                    dtype=np.int64,
                )
                logits = []
                for a in range(0, len(indices), BATCH_ROWS):
                    work()
                    logits.append(
                        head.logits(prepared.features, prepared.tables, indices[a : a + BATCH_ROWS])
                    )
                calibrator = fit_calibrator(
                    np.concatenate(logits) if logits else np.asarray([]),
                    np.asarray([prepared.rows[int(i)]["label"] for i in indices]),
                )
                receipt["calibrators_completed"] += 1
                calibration = calibration_summary(calibrator)
                prefix = f"fold-{fold}-{arm}"
                package = record(
                    _package_artifact(
                        output, prefix + "-model.joblib", head, calibrator, identity, cap, storage_check
                    )
                )
                policy_indices = _role_indices(prepared, role["policy"])
                policy_rows = list(_checked_predictions(prepared, head, calibrator, policy_indices, work))
                policy_events = [e for e in prepared.events if e["group_id"] in role["policy"]]
                policies, searches = freeze_policy_frontiers(
                    policy_rows, policy_events, prepared.config, deadline=deadline
                )
                del policy_rows
                if set(policies) != set(POLICY_NAMES):
                    raise ValueError("All nine finite presets plus legacy bridge required")
                freeze = dict(
                    identity=identity,
                    fold=fold,
                    arm=arm,
                    roles=role,
                    model=package,
                    calibration=calibration,
                    policies=policies,
                    searches=searches,
                    frozen_before_any_outer_scoring=True,
                    calibration_indices=indices.tolist(),
                    policy_indices=policy_indices.tolist(),
                )
                frozen = record(
                    _json_artifact(output, prefix + "-policy-freeze.json", freeze, cap, storage_check)
                )
                ready.append(
                    dict(
                        fold=fold,
                        arm=arm,
                        role=role,
                        head=head,
                        calibrator=calibrator,
                        calibration=calibration,
                        policies=policies,
                        model=package,
                        policy_freeze=frozen,
                    )
                )
                checkpoint()
        if len(ready) != 6 or receipt["models_completed"] != 66 or receipt["calibrators_completed"] != 6:
            raise ValueError("Complete fixed fit/calibration counts required before outer scoring")
        record(
            _json_artifact(
                output,
                "all-policy-freezes.json",
                dict(
                    identity=identity,
                    entries=[dict(fold=v["fold"], arm=v["arm"], **v["policy_freeze"]) for v in ready],
                ),
                cap,
                storage_check,
            )
        )
        entries = []
        people = {arm: {name: [] for name in POLICY_NAMES} for arm in STUDY_ARMS}
        memberships = {arm: [] for arm in STUDY_ARMS}
        for item in ready:
            work()
            _resources(storage_check)
            role = item["role"]
            fold = item["fold"]
            arm = item["arm"]
            prefix = f"fold-{fold}-{arm}"
            receipt["phase"] = prefix + "_outer_evaluation"
            indices = _role_indices(prepared, role["outer"])
            stream = record(
                _prediction_artifact(
                    output,
                    prefix + "-predictions.jsonl.gz",
                    _checked_predictions(prepared, item["head"], item["calibrator"], indices, work),
                    cap,
                    storage_check,
                    work,
                )
            )
            events = [e for e in prepared.events if e["group_id"] in role["outer"]]
            rows = _PredictionRows(output / stream["path"], work)
            reports = {}
            for name in POLICY_NAMES:
                work()
                _resources(storage_check)
                result = evaluate_stream(rows, events, item["policies"][name], bootstrap_samples=0)
                result["per_group"] = _person_rows(result, rows)
                if {v["group_id"] for v in result["per_group"]} != set(role["outer"]):
                    raise ValueError("All original outer people, including negative-only people, required")
                reports[name] = result
                people[arm][name].extend(result["per_group"])
            entry = dict(
                status="complete",
                identity=identity,
                fold=fold,
                arm=arm,
                calibration=item["calibration"],
                policies=item["policies"],
                model=item["model"],
                policy_freeze=item["policy_freeze"],
                predictions=stream["path"],
                predictions_sha256=stream["sha256"],
                outer_rows=stream["rows"],
                availability=stream["availability"],
                outer_events=len(events),
                outer_groups=role["outer"],
                reports=reports,
                primary_constraints=policy_constraints(
                    reports[PRIMARY_POLICY], item["policies"][PRIMARY_POLICY], budget=0.25
                ),
            )
            record(_report_artifact(output, prefix + "-receipt.json.gz", entry, cap, storage_check))
            entries.append(entry)
            memberships[arm].extend(indices.tolist())
            receipt["completed_arms"] += 1
            checkpoint()
        if any(sorted(v) != list(range(len(prepared.rows))) for v in memberships.values()) or any(
            sum(e["outer_events"] for e in entries if e["arm"] == arm) != len(prepared.events)
            for arm in STUDY_ARMS
        ):
            raise ValueError("Every original row and event must occur exactly once per arm")
        comparisons = {}
        receipt["phase"] = "paired_bootstrap"
        for name in POLICY_NAMES:
            work()
            _resources(storage_check)
            comparisons[name] = compare_person_counts(
                people[STUDY_ARMS[0]][name], people[STUDY_ARMS[1]][name], seed=42, samples=10000
            )
            if comparisons[name]["event_denominator"] != len(prepared.events) or comparisons[name][
                "people"
            ] != len(prepared.index["allowed_groups"]):
                raise ValueError("Complete paired original event/person denominators required")
        summary = dict(
            status="complete",
            identity=identity,
            folds=entries,
            comparisons=comparisons,
            diagnostics=diagnostics,
            rows=len(prepared.rows),
            people=len(prepared.index["allowed_groups"]),
            events=len(prepared.events),
            availability={
                arm: {
                    name: sum(v["availability"][name] for v in entries if v["arm"] == arm)
                    for name in ("original_accepted_rows", "common_available_rows", "newly_unavailable_rows")
                }
                for arm in STUDY_ARMS
            },
            timing_comparator_availability="Common availability; newly unavailable rows abstain in both arms",
            coverage_unchanged_between_matched_arms=True,
            **_advancement(entries, comparisons),
        )
        record(_report_artifact(output, "summary.json.gz", summary, cap, storage_check))
        _verify(prepared.files, prepared.root, work)
        _verify(prepared.source, prepared.root, work)
        final_hashes = _verify_tables(prepared, work)
        receipt["hash_validation"] = dict(
            current_execution={k: current["hash_validation"][k] + final_hashes[k] for k in final_hashes},
            inherited_preparation_and_support={
                k: prepared.receipt["hash_validation"][k] + support["hash_validation"][k]
                for k in final_hashes
            },
            scope="Whole-table SHA256 passes only; source/input/compact/row-identity passes are additional",
        )
        receipt.update(
            status="finalizing",
            decision=summary["decision"],
            elapsed_seconds=time.monotonic() - started,
            resources=_resources(storage_check),
            independent_replay_complete=False,
        )
        record(_json_artifact(output, "checkpoint.json", receipt, cap, storage_check))
        from .data import file_hash

        bound = _json_artifact(
            output,
            "artifact-manifest.json",
            dict(version="ecgfounder-study-artifacts-v1", identity=identity, files_sha256=artifacts),
            cap,
            storage_check,
        )
        members = {str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()}
        if members != set(artifacts) | {"receipt.json", "artifact-manifest.json"}:
            raise ValueError("Study artifact membership differs from declared writes")
        for name, digest in {**artifacts, bound["path"]: bound["sha256"]}.items():
            work()
            if file_hash(output / name) != digest:
                raise ValueError("Study artifact changed after writing: " + name)
        receipt["artifact_manifest_sha256"] = bound["sha256"]
        receipt["status"] = "complete"
        work()
        _json_artifact(output, "receipt.json", receipt, cap, storage_check)
        progress.finish()
        return receipt
    except BaseException as error:
        receipt.update(
            status="failed",
            error_type=type(error).__name__,
            original_error=str(error),
            elapsed_seconds=time.monotonic() - started,
            exact_intermediate_fit_count_known=False,
        )
        try:
            _write(output / "receipt.json", receipt)
        except BaseException:
            pass
        progress.finish("failed", error_type=type(error).__name__)
        raise


def _replay_run(run_dir, expected_receipt_sha256, work):
    """Verify externally trusted receipt and exhaustive membership before any binary load."""
    run_dir = Path(run_dir).resolve()
    receipt = read_bound_manifest(run_dir / "receipt.json", expected_receipt_sha256)
    if (
        receipt.get("status") != "complete"
        or receipt.get("completed_arms") != 6
        or receipt.get("models_completed") != 66
        or receipt.get("calibrators_completed") != 6
    ):
        raise ValueError("Replay requires all six source-bound completed primary entries")
    manifest = read_bound_manifest(run_dir / "artifact-manifest.json", receipt["artifact_manifest_sha256"])
    if manifest["version"] != "ecgfounder-study-artifacts-v1" or manifest["identity"] != receipt["identity"]:
        raise ValueError("Run manifest identity differs")
    paths = list(run_dir.rglob("*"))
    members = {str(p.relative_to(run_dir)) for p in paths if p.is_file()}
    if any(p.is_symlink() for p in paths) or members != set(manifest["files_sha256"]) | {
        "receipt.json",
        "artifact-manifest.json",
    }:
        raise ValueError("Replay artifact membership differs")
    _verify(manifest["files_sha256"], run_dir, work)
    required = {
        "preparation.json",
        "support.json",
        "source-manifest.json",
        "config.json",
        "roles.json",
        "events.json",
        "checkpoint.json",
        "all-policy-freezes.json",
        "summary.json.gz",
    }
    required.update("source-package/" + n for n in receipt["identity"]["source_sha256"])
    for fold in range(3):
        required.add(f"fold-{fold}-diagnostics.json")
        models = [f"fold-{fold}-null-{seed}.joblib" for seed in range(101, 120)] + [
            f"fold-{fold}-nuisance.joblib"
        ]
        for arm in STUDY_ARMS:
            prefix = f"fold-{fold}-{arm}"
            models.append(prefix + "-model.joblib")
            required.update(
                prefix + suffix
                for suffix in ("-policy-freeze.json", "-predictions.jsonl.gz", "-receipt.json.gz")
            )
        required.update(models)
        required.update(n + ".json" for n in models)
    if set(manifest["files_sha256"]) != required:
        raise ValueError("Exact declared artifact membership, including66 numeric model manifests, required")
    return receipt, manifest


def _stable_preparation(receipt):
    return {k: v for k, v in receipt.items() if k not in ("resources", "storage")}


def _replay_model_reference(run_dir, name, manifest):
    return dict(
        path=name,
        sha256=manifest["files_sha256"][name],
        bytes=_path(run_dir, name).stat().st_size,
        numeric_manifest=name + ".json",
        numeric_manifest_sha256=manifest["files_sha256"][name + ".json"],
    )


def _replay_package(run_dir, name, manifest, identity, pairs, kind, groups, work):
    from .data import file_hash

    work()
    path = _path(run_dir, name)
    if file_hash(path) != manifest["files_sha256"].get(name):
        raise ValueError("Trusted model changed before binary load")
    numeric = json.loads(_path(run_dir, name + ".json").read_text())
    package = joblib.load(path)
    if set(package) != {"head", "calibrator", "manifest"}:
        raise ValueError("Unexpected trusted model package schema")
    head, calibrator = package["head"], package["calibrator"]
    if not _same_numeric_json(package["manifest"], numeric) or not _same_numeric_json(
        _package_manifest(head, calibrator, identity), numeric
    ):
        raise ValueError("Model coefficients/transform/calibration numeric manifest differs")
    parameters = head.model.get_params(deep=False)
    recipe = dict(
        C=0.1, solver="lbfgs", max_iter=3000, fit_intercept=False, class_weight=None, random_state=42
    )
    if any(parameters.get(k) != v for k, v in recipe.items()):
        raise ValueError("Fixed model parameters differ")
    expected_scope = "primary" if groups is None else "fit_stop_only"
    if head.kind != kind or head.scope != expected_scope or head.allowed_groups != groups:
        raise ValueError("Model arm/diagnostic role scope differs")
    indices, masses, _ = pair_row_masses(pairs)
    transform = head.transforms.manifest()
    if (
        transform.get("timing_schema") != list(LINEAR_SCHEMA)
        or transform.get("timing_columns") != 46
        or transform.get("candidate_columns") != 1070
        or transform.get("delta_columns") != 1024
        or transform.get("delta_order") != "preclassifier_channel_order"
        or transform.get("fit_indices") != indices.tolist()
        or not _same_numeric_json(transform.get("row_masses"), masses.tolist())
    ):
        raise ValueError("Original fit-only transform schema/rows/masses differ")
    imputer = np.asarray(transform["timing_imputer"])
    if imputer.shape != (46,) or not np.isfinite(imputer).all():
        raise ValueError("Invalid numeric timing imputer schema/values")
    for key, width in (("timing_scaler", 46), ("delta_scaler", 1024)):
        for field in ("mean_", "var_", "scale_"):
            value = np.asarray(transform[key][field])
            if (
                value.shape != (width,)
                or not np.isfinite(value).all()
                or (field == "scale_" and (value <= 0).any())
            ):
                raise ValueError("Invalid numeric transform dimensions/values")
    if calibrator is not None:
        sigmoid = calibrator["sigmoid"].get_params(deep=False)
        if any(sigmoid[k] != v for k, v in dict(C=100.0, max_iter=1000, random_state=42).items()):
            raise ValueError("Fixed natural calibrator parameters differ")
    if (groups is None) != (calibrator is not None):
        raise ValueError("Only primary heads have natural calibrators")
    return head, calibrator


def _replay_policies(rows, events, config, freeze, work):
    """Re-evaluate frozen search records and their tie rule; never call a selector."""
    if set(freeze["policies"]) != set(POLICY_NAMES) or set(freeze["searches"]) != set(POLICY_NAMES):
        raise ValueError("All ten frozen policy/search entries required")
    levels = [0.5, 0.75, 0.9, 0.925, 0.95, 0.975, 0.99, 0.995, 0.999]
    risks = [r["risk"] for r in rows if r["accepted"] and r["label"] in (0, 1) and r["risk"] is not None]
    thresholds = sorted(
        set(
            np.r_[
                np.linspace(0.01, 0.99, 25),
                0.001,
                0.005,
                0.025,
                0.995,
                0.999,
                1.0,
                np.quantile(risks, levels),
            ]
        )
    )
    cache = {}
    for name in POLICY_NAMES:
        work()
        legacy = name == "legacy_bridge"
        base = copy.deepcopy(config)
        if legacy:
            budget = 0.25
            for key in (
                "policy_version",
                "max_duration_seconds",
                "rearm_seconds",
                "max_burden_occupancy",
                "max_eligible_negative_burden_occupancy",
            ):
                base["alert"].pop(key, None)
        else:
            _, far, _, burden = name.split("_")
            budget, burden = float(far), float(burden)
            base["alert"].update(max_burden_occupancy=burden, max_eligible_negative_burden_occupancy=burden)
        search = freeze["searches"][name]
        if not _same_numeric_json([v["threshold"] for v in search], thresholds):
            raise ValueError("Frozen policy threshold grid differs")
        identity = policy_search_identity(rows, events, base)
        ratio = base["alert"]["release_threshold"] / max(base["alert"]["threshold"], 1e-12)
        feasible = []
        for candidate in search:
            work()
            policy = copy.deepcopy(base)
            policy["alert"].update(
                threshold=candidate["threshold"],
                release_threshold=candidate["threshold"] * ratio,
                max_false_warnings_per_hour=budget,
                min_event_sensitivity=0.0,
            )
            cache_key = (legacy, candidate["threshold"])
            if cache_key not in cache:
                cache[cache_key] = evaluate_stream(
                    rows, events, policy, bootstrap_samples=0, include_probability=False
                )
            report = cache[cache_key]
            keys = (
                "event_sensitivity",
                "false_warnings_per_hour",
                "warnings",
                "unassessable_warnings",
                "warning_hours",
                "eligible_events",
            )
            expected = dict(
                threshold=candidate["threshold"],
                release_threshold=policy["alert"]["release_threshold"],
                grid_scope="fixed grid plus predeclared quantiles of accepted labeled policy-group risks only",
                quantile_levels=levels,
                policy_eligible_windows=len(risks),
                **{k: report[k] for k in keys},
            )
            passed = (
                report["false_warnings_per_hour"] is not None
                and report["event_sensitivity"] is not None
                and report["false_warnings_per_hour"] <= budget
            )
            if not legacy:
                constraints = policy_constraints(report, policy, budget=budget)
                passed = constraints["passed"]
                expected.update(
                    search_identity=identity,
                    **constraints,
                    budget_request=budget,
                    occupancy_request={
                        k: policy["alert"][k]
                        for k in ("max_burden_occupancy", "max_eligible_negative_burden_occupancy")
                    },
                    **{
                        k: report[k]
                        for k in (
                            "burden_occupancy",
                            "eligible_negative_burden_occupancy",
                            "burden_seconds",
                            "monitored_seconds",
                            "eligible_seconds",
                            "eligible_negative_seconds",
                            "eligible_negative_burden_seconds",
                            "unresolved_elevated_seconds",
                        )
                    },
                )
            expected["meets_constraint"] = passed
            if not _same_numeric_json(expected, candidate):
                raise ValueError("Frozen policy search metrics/budgets differ")
            if passed:
                feasible.append(candidate)
        if not feasible:
            raise ValueError("Frozen policy has no feasible search entry")
        best = min(
            feasible,
            key=lambda v: (
                -v["event_sensitivity"],
                v["false_warnings_per_hour"],
                v["warnings"],
                v["warning_hours"] if legacy else v["burden_seconds"],
                -v["threshold"],
            ),
        )
        base["alert"].update(
            threshold=best["threshold"],
            release_threshold=best["release_threshold"],
            max_false_warnings_per_hour=budget,
            min_event_sensitivity=0.0,
        )
        if not _same_numeric_json(base, freeze["policies"][name]):
            raise ValueError("Frozen policy config or original tie rule differs")


def replay_study(prepared, run_dir, output, *, expected_receipt_sha256, guard, storage_check, deadline):
    """Independently rescore trusted saved models and frozen policies, without any fitting."""
    output = fresh_output(output)
    run_dir = Path(run_dir).resolve()
    started = time.monotonic()
    receipt = dict(
        status="running", outcome_fits=0, packages_checked=0, completed_arms=0, full_run_eligible=False
    )
    _write(output / "receipt.json", receipt)

    def work():
        guard()
        if not math.isfinite(deadline) or time.monotonic() >= deadline:
            raise TimeoutError("Original replay work deadline exceeded")

    try:
        work()
        _resources(storage_check)
        if prepared.closed or prepared.tables._closed or "replay" not in prepared.receipt["allowed_stages"]:
            raise ValueError("Open preparation and declared replay stage required")
        original, manifest = _replay_run(run_dir, expected_receipt_sha256, work)
        identity = original["identity"]

        def read(name):
            path = _path(run_dir, name)
            return _read_report(path) if name.endswith(".json.gz") else json.loads(path.read_text())

        saved_preparation, saved_support = read("preparation.json"), read("support.json")
        checkpoint = {k: v for k, v in original.items() if k != "artifact_manifest_sha256"}
        checkpoint["status"] = "finalizing"
        if not _same_numeric_json(read("checkpoint.json"), checkpoint):
            raise ValueError("Final checkpoint differs from authoritative completed run receipt")
        current_support = audit_common_support(prepared, guard=work)
        if (
            _canonical_sha(saved_preparation) != identity["preparation_sha256"]
            or _support_identity(saved_support) != identity["support_sha256"]
            or _canonical_sha(_stable_preparation(saved_preparation))
            != _canonical_sha(_stable_preparation(prepared.receipt))
            or _canonical_sha(
                {k: v for k, v in saved_support.items() if k not in ("resources", "preparation_sha256")}
            )
            != _canonical_sha(
                {k: v for k, v in current_support.items() if k not in ("resources", "preparation_sha256")}
            )
        ):
            raise ValueError("Original preparation/support scientific identity differs")
        if any(
            identity[k] != expected
            for k, expected in dict(
                source_sha256=prepared.source,
                files_sha256=prepared.files,
                compact_binding=prepared.receipt["compact_binding"],
                seed=42,
                arms=list(STUDY_ARMS),
            ).items()
        ):
            raise ValueError("Run source/input/model recipe differs")
        for name, expected in (
            ("config.json", prepared.config),
            ("roles.json", prepared.roles),
            ("events.json", prepared.events),
            ("source-manifest.json", prepared.source),
        ):
            if read(name) != expected:
                raise ValueError("Original artifact config/role/event/source differs")
        for name, digest in prepared.source.items():
            if manifest["files_sha256"].get("source-package/" + name) != digest:
                raise ValueError("Copied executed source differs")
        summary = read("summary.json.gz")
        if summary["identity"] != identity or summary["status"] != "complete" or len(summary["folds"]) != 6:
            raise ValueError("Original complete summary identity differs")
        freezes = read("all-policy-freezes.json")
        if freezes["identity"] != identity or len(freezes["entries"]) != 6:
            raise ValueError("All policies must freeze before any outer scoring")
        entries, diagnostics = [], []
        people = {arm: {name: [] for name in POLICY_NAMES} for arm in STUDY_ARMS}
        seen = {arm: [] for arm in STUDY_ARMS}
        for role in prepared.roles:
            fold = role["fold"]
            pairs = {
                n: [EventPair(**v) for v in current_support["surviving_pairs"][fold][n]] for n in ROLE_NAMES
            }
            allowed = frozenset(p.group_id for p in pairs["fit"]) | frozenset(role["stop"])
            diag = dict(fold=fold, identity=identity, roles={}, nulls=[], nuisance={})
            primary = {}
            for arm, kind in zip(STUDY_ARMS, ("timing", "candidate"), strict=True):
                name = f"fold-{fold}-{arm}-model.joblib"
                primary[arm] = _replay_package(
                    run_dir, name, manifest, identity, pairs["fit"], kind, None, work
                )
                receipt["packages_checked"] += 1
            transforms = [head.transforms.identity_sha256() for head, _ in primary.values()]
            if len(set(transforms)) != 1:
                raise ValueError("Matched primary arms must share exact fit transforms")
            for name in ("fit", "stop"):
                indices = _role_indices(prepared, role[name])
                drift = negative_drift_pairs(_CommonRows(prepared), set(role[name]))
                diag["roles"][name] = {
                    arm: dict(
                        event_pairs=_pair_diagnostics(head, prepared, pairs[name], work),
                        negative_drift=_pair_diagnostics(head, prepared, drift, work),
                        identity=_identity_variance(head, prepared, indices, work),
                    )
                    for arm, (head, _) in primary.items()
                }
            saved_diag = read(f"fold-{fold}-diagnostics.json")
            if [v["seed"] for v in saved_diag["nulls"]] != list(range(101, 120)):
                raise ValueError("All original null orientations required")
            for saved in saved_diag["nulls"]:
                seed = saved["seed"]
                name = f"fold-{fold}-null-{seed}.joblib"
                if saved["model"] != _replay_model_reference(run_dir, name, manifest):
                    raise ValueError("Null model artifact reference differs")
                head, _ = _replay_package(
                    run_dir, name, manifest, identity, pairs["fit"], "candidate", allowed, work
                )
                _, orientation = orientation_labels(pairs["fit"], seed)
                if head.transforms.identity_sha256() != transforms[0]:
                    raise ValueError("Null transform differs from shared real-fit transform")
                diag["nulls"].append(
                    dict(
                        seed=seed,
                        orientation=orientation,
                        stop=_pair_diagnostics(head, prepared, pairs["stop"], work),
                        model=saved["model"],
                    )
                )
                receipt["packages_checked"] += 1
            name = f"fold-{fold}-nuisance.joblib"
            if saved_diag["nuisance"]["model"] != _replay_model_reference(run_dir, name, manifest):
                raise ValueError("Nuisance model artifact reference differs")
            nuisance, _ = _replay_package(
                run_dir, name, manifest, identity, pairs["fit"], "nuisance", allowed, work
            )
            if nuisance.transforms.identity_sha256() != transforms[0]:
                raise ValueError("Nuisance shared transform differs")
            receipt["packages_checked"] += 1
            diag["nuisance"] = dict(
                model=saved_diag["nuisance"]["model"],
                **{n: _pair_diagnostics(nuisance, prepared, pairs[n], work) for n in ("fit", "stop")},
            )
            real = diag["roles"]["stop"][STUDY_ARMS[1]]["event_pairs"]["person_macro_concordance"]
            base = diag["roles"]["stop"][STUDY_ARMS[0]]["event_pairs"]["person_macro_concordance"]
            control = diag["nuisance"]["stop"]["person_macro_concordance"]
            diag["nuisance_equals_or_exceeds_candidate_stop_improvement"] = (
                all(v is not None for v in (real, base, control)) and control - base >= real - base
            )
            if not _same_numeric_json(diag, saved_diag):
                raise ValueError("Independent fit/stop controls differ")
            diagnostics.append(diag)
            for arm in STUDY_ARMS:
                work()
                _resources(storage_check)
                head, calibrator = primary[arm]
                prefix = f"fold-{fold}-{arm}"
                entry = read(prefix + "-receipt.json.gz")
                freeze = read(prefix + "-policy-freeze.json")
                expected_freeze = next(v for v in freezes["entries"] if v["fold"] == fold and v["arm"] == arm)
                if (
                    entry["identity"] != identity
                    or entry["fold"] != fold
                    or entry["arm"] != arm
                    or entry["status"] != "complete"
                    or entry["model"] != _replay_model_reference(run_dir, prefix + "-model.joblib", manifest)
                    or entry["predictions"] != prefix + "-predictions.jsonl.gz"
                    or entry["predictions_sha256"] != manifest["files_sha256"][entry["predictions"]]
                    or expected_freeze != dict(fold=fold, arm=arm, **entry["policy_freeze"])
                    or expected_freeze["sha256"] != manifest["files_sha256"][prefix + "-policy-freeze.json"]
                    or freeze["identity"] != identity
                    or freeze["roles"] != role
                    or freeze["fold"] != fold
                    or freeze["arm"] != arm
                    or not freeze["frozen_before_any_outer_scoring"]
                    or freeze["model"] != entry["model"]
                    or freeze["policies"] != entry["policies"]
                ):
                    raise ValueError("Frozen calibration/policy/role authority differs")
                indices = _role_indices(prepared, role["calibration"])
                indices = [
                    int(i)
                    for i in indices
                    if prepared.tables.rows[i]["available"] and prepared.rows[int(i)]["label"] in (0, 1)
                ]
                policy_indices = _role_indices(prepared, role["policy"])
                if (
                    freeze["calibration_indices"] != indices
                    or freeze["policy_indices"] != policy_indices.tolist()
                ):
                    raise ValueError("Original natural calibration/policy membership differs")
                calibration = calibration_summary(calibrator)
                if not _same_numeric_json(calibration, freeze["calibration"]):
                    raise ValueError("Independent calibration parameters differ")
                policy_rows = list(_checked_predictions(prepared, head, calibrator, policy_indices, work))
                _replay_policies(
                    policy_rows,
                    [e for e in prepared.events if e["group_id"] in role["policy"]],
                    prepared.config,
                    freeze,
                    work,
                )
                del policy_rows
                indices = _role_indices(prepared, role["outer"])
                saved_rows = iter(_PredictionRows(_path(run_dir, entry["predictions"]), work))
                original_accepted = available = count = 0
                for fresh in _checked_predictions(prepared, head, calibrator, indices, work):
                    saved = next(saved_rows, None)
                    if not _same_numeric_json(fresh, saved):
                        raise ValueError("Independent outer original chronology/logit/risk differs")
                    original_accepted += fresh["original_accepted"]
                    available += fresh["accepted"]
                    count += 1
                if next(saved_rows, None) is not None:
                    raise ValueError("Extra original outer prediction")
                events = [e for e in prepared.events if e["group_id"] in role["outer"]]
                reports = {}
                for name in POLICY_NAMES:
                    work()
                    rows = _PredictionRows(_path(run_dir, entry["predictions"]), work)
                    report = evaluate_stream(rows, events, freeze["policies"][name], bootstrap_samples=0)
                    report["per_group"] = _person_rows(report, rows)
                    if {v["group_id"] for v in report["per_group"]} != set(role["outer"]):
                        raise ValueError("Original outer person membership differs")
                    reports[name] = report
                    people[arm][name].extend(report["per_group"])
                fresh_entry = dict(
                    entry,
                    calibration=calibration,
                    reports=reports,
                    outer_rows=count,
                    outer_events=len(events),
                    outer_groups=role["outer"],
                    availability=dict(
                        original_accepted_rows=int(original_accepted),
                        common_available_rows=int(available),
                        newly_unavailable_rows=int(original_accepted - available),
                    ),
                    primary_constraints=policy_constraints(
                        reports[PRIMARY_POLICY], freeze["policies"][PRIMARY_POLICY], budget=0.25
                    ),
                )
                if not _same_numeric_json(fresh_entry, entry):
                    raise ValueError("Independent evaluation report/denominator differs")
                entries.append(fresh_entry)
                seen[arm].extend(indices.tolist())
                receipt["completed_arms"] += 1
                receipt["elapsed_seconds"] = time.monotonic() - started
                _write(output / "receipt.json", receipt)
        if any(sorted(v) != list(range(len(prepared.rows))) for v in seen.values()) or any(
            sum(v["outer_events"] for v in entries if v["arm"] == arm) != len(prepared.events)
            for arm in STUDY_ARMS
        ):
            raise ValueError("Every original row/event must reconcile exactly once per arm")
        comparisons = {}
        for name in POLICY_NAMES:
            work()
            _resources(storage_check)
            comparisons[name] = compare_person_counts(
                people[STUDY_ARMS[0]][name], people[STUDY_ARMS[1]][name], seed=42, samples=10000
            )
            if comparisons[name]["people"] != len(prepared.index["allowed_groups"]) or comparisons[name][
                "event_denominator"
            ] != len(prepared.events):
                raise ValueError("Original bootstrap person/event denominators differ")
        fresh_summary = dict(
            summary,
            folds=entries,
            diagnostics=diagnostics,
            comparisons=comparisons,
            rows=len(prepared.rows),
            people=len(prepared.index["allowed_groups"]),
            events=len(prepared.events),
            availability={
                arm: {
                    key: sum(v["availability"][key] for v in entries if v["arm"] == arm)
                    for key in ("original_accepted_rows", "common_available_rows", "newly_unavailable_rows")
                }
                for arm in STUDY_ARMS
            },
            **_advancement(entries, comparisons),
        )
        if (
            not _same_numeric_json(fresh_summary, summary)
            or original["decision"] != fresh_summary["decision"]
        ):
            raise ValueError("Independent paired bootstrap/advancement summary differs")
        _verify(prepared.files, prepared.root, work)
        _verify(prepared.source, prepared.root, work)
        hashes = _verify_tables(prepared, work)
        _replay_run(run_dir, expected_receipt_sha256, work)
        _report_artifact(output, "review.json.gz", fresh_summary, 250000000, storage_check)
        receipt.update(
            status="complete",
            decision=fresh_summary["decision"],
            rows_per_arm=len(prepared.rows),
            events_per_arm=len(prepared.events),
            people_per_arm=len(prepared.index["allowed_groups"]),
            independent_replay_complete=True,
            run_receipt_sha256=expected_receipt_sha256,
            elapsed_seconds=time.monotonic() - started,
            resources=_resources(storage_check),
            hash_validation={k: current_support["hash_validation"][k] + hashes[k] for k in hashes},
        )
        work()
        _write(output / "receipt.json", receipt)
        return receipt
    except BaseException as error:
        receipt.update(
            status="failed",
            original_error=str(error),
            error_type=type(error).__name__,
            elapsed_seconds=time.monotonic() - started,
        )
        try:
            _write(output / "receipt.json", receipt)
        except BaseException:
            pass
        raise
