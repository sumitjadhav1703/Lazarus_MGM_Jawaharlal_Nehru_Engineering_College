"""Source-bound, outcome-blind engineering audit; never fits or submits a model."""

import argparse
import json
import resource
import shutil
import sys
import time
import traceback
from pathlib import Path

from cardiotwin.data import RecordInfo, cached_signal_reader, file_hash
from cardiotwin.investigation import _digest, _record_contract, _write, read_cache
from cardiotwin.morphology import (
    MORPHOLOGY_SCHEMA,
    extract_morphology,
    measure_histories,
    measurement_gate,
    select_audit_rows,
)
from cardiotwin.precursor import compact_cache_binding, fresh_output, source_manifest
from cardiotwin.progress import Progress


def synthetic_binding(directory):
    directory = Path(directory)
    receipt = json.loads((directory / "receipt.json").read_text())
    if receipt["status"] != "passed" or receipt["cases"] != 135 or receipt["failed_checks"]:
        raise ValueError("A passed complete fixed synthetic validation is required")
    paths = {directory / "receipt.json": file_hash(directory / "receipt.json")}
    for name in ("cases", "predeclaration"):
        path = directory / ("cases.jsonl" if name == "cases" else "predeclaration.json")
        if file_hash(path) != receipt[f"{name}_sha256"]:
            raise ValueError("Synthetic evidence identity changed")
        paths[path] = file_hash(path)
    declaration = json.loads((directory / "predeclaration.json").read_text())
    if (declaration["mode"] != "shared-channel0-v1"
            or _digest(declaration["source"]["files"]) != declaration["source"]["sha256"]):
        raise ValueError("Synthetic source declaration differs")
    required = {"src/cardiotwin/morphology.py", "src/cardiotwin/features.py", "scripts/run_anchor_noise_validation.py"}
    if not required.issubset(declaration["source"]["files"]):
        raise ValueError("Synthetic source binding is incomplete")
    for name, checksum in declaration["source"]["files"].items():
        if file_hash(name) != checksum:
            raise ValueError(f"Validated synthetic source/declaration changed: {name}")
        paths[Path(name)] = checksum
    return {str(p): checksum for p, checksum in paths.items()}


def measure_shared_challenge(selected, records, reader, output, *, deadline):
    output = Path(output)
    if json.loads((output / "selected.json").read_text()) != selected:
        raise ValueError("Saved challenge selection differs")
    if len(selected) != 36 or [sum(r["fold"] == f for r in selected) for f in range(3)] != [12]*3:
        raise ValueError("Exactly twelve requests per original fold required")
    required = {("iridia-af:patient_021", "record_021", 56520.),
                ("iridia-af:patient_150", "record_165", 167910.),
                ("iridia-af:patient_137", "record_151", 84660.)}
    counts, sentinels = [0, 0, 0], {}
    with (output / "measurements.jsonl").open("x") as handle, Progress(
        "morphology.shared_challenge", total=36, device="cpu", mode="shared-channel0-v1"
    ) as progress:
        for n, item in enumerate(selected, 1):
            if time.monotonic() >= deadline:
                raise TimeoutError("Shared challenge deadline reached")
            record = records[item["record_id"]]
            signal = reader(record, item["timestamp"]-600, item["timestamp"])
            measurement = extract_morphology(signal, record.fs, anchor_mode="shared-channel0-v1", overlays=True)
            counts[item["fold"]] += int(measurement["available"])
            identity = (item["patient_id"], item["record_id"], item["timestamp"])
            if identity in required:
                prior = sentinels.get(item["record_id"], {}).get("passed", True)
                sentinels[item["record_id"]] = dict(passed=prior and measurement["available"], identity=list(identity))
            handle.write(json.dumps({**item, "measurement": measurement}, allow_nan=False)+"\n")
            handle.flush()
            progress.update(completed=n, force=True, valid_by_fold=counts.copy())
    coverage = measurement_gate(counts)
    passed = coverage["status"] == "passed" and len(sentinels) == 3 and all(s["passed"] for s in sentinels.values())
    return dict(status="passed" if passed else "failed", coverage=coverage, sentinels=sentinels)


def run(args):
    started = time.monotonic()
    output = fresh_output(args.output)
    try:
        audit = Path(args.audit)
        receipt = json.loads((audit / "receipt.json").read_text())
        validation = json.loads(Path(args.source_validation).read_text())
        if (receipt["status"] != "passed" or receipt["metadata_status"] != "passed"
                or receipt["pair_support_status"] != "passed" or validation["status"] != "passed"):
            raise ValueError("Passed metadata/support and source-validation evidence required")
        with Progress("morphology.compact_binding", device="cpu"):
            binding = compact_cache_binding(args.cache, expected=validation["binding"])
            rows, events, index = read_cache(args.cache)
        roles = json.loads((audit / "roles.json").read_text())
        if (binding["index_sha256"] != receipt["cache_index_sha256"]
                or _digest(roles["roles"]) != receipt["roles_sha256"]
                or file_hash(audit / "pairs.jsonl") != receipt["pairs_sha256"]):
            raise ValueError("Original audit/role/pair identities changed")
        if roles["cache_index_sha256"] != binding["index_sha256"]:
            raise ValueError("Roles bind another cohort")
        challenge = getattr(args, "shared_challenge", None)
        extra_evidence = {}
        if challenge:
            extra_evidence = synthetic_binding(args.synthetic)
            synthetic_declaration = json.loads((Path(args.synthetic) / "predeclaration.json").read_text())
            if synthetic_declaration["spec_sha256"] != file_hash(args.spec):
                raise ValueError("Shared challenge requires the validated scientific declaration")
            challenge = Path(challenge)
            original = json.loads((challenge / "predeclaration.json").read_text())
            original_receipt = json.loads((challenge / "receipt.json").read_text())
            if (file_hash(challenge / "selected.json") != original["selected_sha256"]
                    or original["compact_binding"] != binding
                    or file_hash(challenge / "predeclaration.json") != original_receipt["predeclaration_sha256"]
                    or file_hash(challenge / "measurements.jsonl") != original_receipt["measurements_sha256"]):
                raise ValueError("Original challenge identity changed")
            selected = json.loads((challenge / "selected.json").read_text())
            fit = {role["fold"]: set(role["fit"]) for role in roles["roles"]}
            for item in selected:
                row = rows[item["row_index"]]
                if (any(row[key] != item[key] for key in ("group_id", "patient_id", "record_id", "timestamp"))
                        or item["group_id"] not in fit[item["fold"]] or not row["accepted"] or row["timestamp"] < 600):
                    raise ValueError("Challenge row/fit role differs from original selection")
            extra_evidence.update({str(challenge / name): file_hash(challenge / name)
                                   for name in ("selected.json", "predeclaration.json", "receipt.json", "measurements.jsonl")})
        else:
            with Progress("morphology.select", total=len(rows), device="cpu"):
                selected = select_audit_rows(rows, roles["roles"])
        _write(output / "selected.json", selected)
        source = source_manifest()
        source["files"]["scripts/run_morphology_audit.py"] = file_hash(__file__)
        source["files"]["tests/test_morphology.py"] = file_hash("tests/test_morphology.py")
        source["files"]["tests/test_morphology_challenge.py"] = file_hash("tests/test_morphology_challenge.py")
        source["sha256"] = _digest(source["files"])
        evidence = {str(p): file_hash(p) for p in (audit / "receipt.json", audit / "roles.json",
                    audit / "pairs.jsonl", Path(args.source_validation), Path(args.spec))}
        evidence.update(extra_evidence)
        records = {entry["record"]["record_id"]: RecordInfo(**entry["record"]) for entry in index["records"]}
        contracts = {entry["record"]["record_id"]: entry["input_contract"] for entry in index["records"]}
        chosen = sorted({item["record_id"] for item in selected})
        declaration = dict(source=source, compact_binding=binding, evidence=evidence,
                           selected_sha256=file_hash(output / "selected.json"), schema=MORPHOLOGY_SCHEMA,
                           raw_contracts={key: contracts[key] for key in chosen}, rows=len(rows), events=len(events),
                           outcome_associations=False, new_fits=0, retry=False,
                           mode="shared-channel0-v1" if challenge else "independent", reused_challenge=bool(challenge))
        _write(output / "predeclaration.json", declaration)
        for name in source["files"]:
            destination = output / "source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(name, destination)
        with Progress("morphology.raw_hash_validation", total=len(chosen), device="cpu") as progress:
            for n, key in enumerate(chosen, 1):
                if _record_contract(records[key]) != contracts[key]:
                    raise ValueError(f"Raw acquisition changed: {key}")
                progress.update(completed=n, force=True)
        with cached_signal_reader() as reader:
            measure = measure_shared_challenge if challenge else measure_histories
            gate = measure(selected, records, reader, output, deadline=started+args.max_seconds)
        if compact_cache_binding(args.cache) != binding:
            raise ValueError("Compact binding changed during audit")
        for path, checksum in {**source["files"], **evidence}.items():
            if file_hash(path) != checksum:
                raise ValueError(f"Source/evidence changed during audit: {path}")
        result = dict(status="measurement_passed_pending_inspection" if gate["status"] == "passed"
                      else "measurement_failed", gate=gate, elapsed_seconds=time.monotonic()-started,
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
                      measurements_sha256=file_hash(output / "measurements.jsonl"),
                      predeclaration_sha256=file_hash(output / "predeclaration.json"),
                      fits=0, remote_jobs=0, retry=False)
        _write(output / "receipt.json", result)
        print(json.dumps(result), flush=True)
        return 0
    except BaseException as error:
        _write(output / "receipt.json", dict(status="failed", error_type=type(error).__name__,
               error=str(error), traceback=traceback.format_exc(), elapsed_seconds=time.monotonic()-started,
               fits=0, remote_jobs=0, retry=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "audit", "source-validation", "spec", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--max-seconds", type=float, default=1200)
    parser.add_argument("--shared-challenge", help="Original measurement directory: reuse exact36 requests")
    parser.add_argument("--synthetic", help="Passed source-bound fixed noise-validation directory")
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 1200:
        parser.error("Bounded audit allowance must be in (0,1200]")
    if bool(args.shared_challenge) != bool(args.synthetic):
        parser.error("--shared-challenge and --synthetic are required together")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
