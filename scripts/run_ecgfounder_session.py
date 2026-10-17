"""One source-bound full session after private source/dependency bootstrap; never submit or retry."""
# ruff: noqa: E402

import time

BOOTSTRAP_STARTED = time.monotonic()

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

os.environ["OMP_NUM_THREADS"] = "1"

from cardiotwin.ecgfounder import CHECKPOINT_BYTES, CHECKPOINT_SHA256
from cardiotwin.ecgfounder_runner import (
    CHECKPOINT_PATH,
    CYCLE,
    EXPECTED_ROWS,
    ROLE_NAMES,
    ExecutionGuard,
    _storage,
    coverage_cell,
)
from cardiotwin.ecgfounder_runner import (
    REQUIRED_SOURCE as RUNNER_SOURCE,
)
from cardiotwin.ecgfounder_study import REQUIRED_SOURCE as STUDY_SOURCE
from cardiotwin.investigation import _write

ENTRY = "scripts/run_ecgfounder_session.py"
REQUIRED_SOURCE = (
    *RUNNER_SOURCE,
    *STUDY_SOURCE,
    ENTRY,
    "scripts/run_ecgfounder_study.py",
    "scripts/kaggle_cycle.py",
    "scripts/kaggle_ecgfounder_engineering.py",
    "scripts/run_ecgfounder_engineering.py",
    "scripts/kaggle_ecgfounder_probe.py",
    "reports/ml-investigation/monitor-storage.py",
    CYCLE + "/vendor-source/net1d.py",
    CYCLE + "/vendor-source/LICENSE",
)
HASH_ACCOUNTING = {}
SESSION_OPERATIONS = dict(
    stage_files=1,
    checkpoint_acquisition=1,
    post_acquisition_checkpoint_verify=1,
    extraction_preflight=1,
    outcome_blind_extraction=1,
    prepare_study=2,
    standalone_support=1,
    execute_study=1,
    replay_study=1,
    final_source_verification=1,
    final_input_verification=1,
    final_checkpoint_verify=1,
    final_raw_verification=1,
    final_run_verification=2,
    worker_bundle_verification=1,
    supervisor_bundle_verification=1,
    entry_source_verification=2,
)


def sha(path, *, guard=lambda: None):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        size = 0
        while block := stream.read(1024 * 1024):
            guard()
            digest.update(block)
            size += len(block)
    item = HASH_ACCOUNTING.setdefault(str(path), dict(passes=0, bytes=0))
    item["passes"] += 1
    item["bytes"] += size
    return digest.hexdigest()


def path(root, name):
    relative = Path(name)
    root = Path(root).resolve()
    if relative.is_absolute() or ".." in relative.parts or str(relative) != name or name in ("", "."):
        raise ValueError("Canonical relative session path required")
    target = root / relative
    if any(p.is_symlink() for p in (target, *target.parents) if p.is_relative_to(root)):
        raise ValueError("Session paths cannot contain symlinks")
    return target


def bound(root, name, digest):
    target = path(root, name)
    if sha(target) != digest:
        raise ValueError("Bound session input changed: " + name)
    return json.loads(target.read_text())


def verify(root, files, guard, *, skip=()):
    for name, digest in files.items():
        guard()
        if name not in skip and sha(path(root, name), guard=guard) != digest:
            raise ValueError("Bound session source/input changed: " + name)


def load_script(root, name):
    spec = importlib.util.spec_from_file_location("session_" + name, path(root, "scripts/" + name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def session_guard(session_started):
    clock = ExecutionGuard(session_started=session_started, stage_seconds=41100)
    cli = load_script(Path(__file__).resolve().parents[1], "run_ecgfounder_study")
    deadline = cli.work_deadline(session_started)

    def guard():
        clock()
        if time.monotonic() >= deadline:
            raise TimeoutError("Original whole-session/calendar work deadline exhausted")

    guard.session_started = session_started
    return guard, deadline


def load_apis(root):
    from cardiotwin import ecgfounder_runner as runner
    from cardiotwin import ecgfounder_study as study
    from cardiotwin.precursor import _same_numeric_json, source_manifest

    driver = load_script(root, "run_ecgfounder_engineering")
    probe = load_script(root, "kaggle_ecgfounder_probe")

    def checkpoint(project, guard, storage):
        guard()
        _storage(storage, CHECKPOINT_BYTES + 2619117060)
        destination = path(project, CHECKPOINT_PATH)
        if destination.exists():
            verify_checkpoint(project, guard)
            return dict(status="verified_existing", retry=False)
        return probe.acquire_checkpoint(
            "https://huggingface.co/PKUDigitalHealth/ECGFounder/resolve/"
            + probe.REVISION
            + "/1_lead_ECGFounder.pth",
            destination,
            CHECKPOINT_BYTES,
            CHECKPOINT_SHA256,
            lambda event: print(json.dumps(dict(stage="checkpoint", **event)), flush=True),
        )

    return SimpleNamespace(
        source_manifest=source_manifest,
        cost=runner._cost,
        stage_files=driver.stage_files,
        checkpoint=checkpoint,
        preflight=runner.preflight,
        extract=runner.extract,
        prepare=study.prepare_study,
        support=study.audit_common_support,
        execute=study.execute_study,
        replay=study.replay_study,
        verify_run=study._replay_run,
        same=_same_numeric_json,
        raw_post=runner._verify_raw,
    )


def verify_checkpoint(root, guard):
    target = path(root, CHECKPOINT_PATH)
    if target.stat().st_size != CHECKPOINT_BYTES or sha(target, guard=guard) != CHECKPOINT_SHA256:
        raise ValueError("Pinned checkpoint bytes/hash differ")


def authority(root, declaration, digest, api, guard):
    value = bound(root, declaration, digest)
    if value["version"] != "ecgfounder-session-v1":
        raise ValueError("Fixed full-session contract required")
    files, source = value["files_sha256"], value["source_sha256"]
    current = api.source_manifest()["files"]
    if (
        any(source.get(n) != pin for n, pin in current.items())
        or not set(REQUIRED_SOURCE) <= set(source)
        or source[ENTRY] != sha(Path(__file__))
    ):
        raise ValueError("Actual caller and complete executed source closure required")
    verify(root, source, guard)
    extraction = bound(root, value["extraction_declaration"], files[value["extraction_declaration"]])
    template = bound(root, value["study_template"], files[value["study_template"]])
    output = value["output"]
    dynamic = {
        output + "/extraction/receipt.json",
        output + "/extraction/preflight.json",
        output + "/extraction/coverage.json",
        output + "/extraction/tables/receipt.json",
    }
    if (
        extraction["version"] != "ecgfounder-full-extraction-v1"
        or template["version"] != "ecgfounder-study-v1"
        or extraction["source_sha256"] != source
        or template["source_sha256"] != source
        or {n for n, pin in template["files_sha256"].items() if pin is None} != dynamic
        or template["runner_receipt"] != output + "/extraction/receipt.json"
        or template["runner_preflight"] != output + "/extraction/preflight.json"
        or template["coverage_receipt"] != output + "/extraction/coverage.json"
        or template["tables"] != output + "/extraction/tables"
        or value["study_declaration"] != output + "/study-declaration.json"
        or template["extraction_declaration"] != value["extraction_declaration"]
        or set(template["allowed_stages"]) != {"preflight", "support", "run", "replay"}
    ):
        raise ValueError("Uniform source and exact four engineering completion placeholders required")
    for key in ("cache", "roles", "raw_manifest", "compact_binding", "cost_receipt", "cost_review"):
        if template[key] != extraction[key]:
            raise ValueError("Prospective study/extraction field differs: " + key)
    for bindings in (extraction["files_sha256"], template["files_sha256"]):
        if any(pin is not None and files.get(n) != pin for n, pin in bindings.items()):
            raise ValueError("Prospective static inputs differ")
    if files.get(CHECKPOINT_PATH) != CHECKPOINT_SHA256:
        raise ValueError("Pinned checkpoint authority required")
    staging = value["staging"]
    required = {
        extraction["cache"] + "/" + n for n in ("index.json", "compact.json", "features.npy", "metadata.npy")
    }
    required.update(
        n for n in files if n.startswith(template["metadata_root"] + "/") and n.endswith("_ecg_labels.csv")
    )
    if set(staging) != required or len(set(staging.values())) != len(staging):
        raise ValueError("Only exact ordinary cache and required tiny annotation copies may be staged")
    verify(root, files, guard, skip=(*staging, CHECKPOINT_PATH))
    cost = bound(root, extraction["cost_receipt"], files[extraction["cost_receipt"]])
    review = bound(root, extraction["cost_review"], files[extraction["cost_review"]])
    if review["cost_sha256"] != files[extraction["cost_receipt"]]:
        raise ValueError("Complete cost review identity differs")
    if any(item.get("session_operations") != SESSION_OPERATIONS for item in (cost, review)):
        raise ValueError("Reviewed complete caller/replay/hash/export operation inventory required")
    result = api.cost(cost, review, files, source, extraction["compact_binding"], datetime.now(timezone.utc))
    return value, extraction, template, dynamic, result


def run_session(declaration, expected_sha256, *, root, input_root, data_root, session_started, storage_check):
    root, input_root, data_root = (Path(p).resolve() for p in (root, input_root, data_root))
    if not data_root.is_relative_to(input_root):
        raise ValueError("Raw inputs must remain inside the counted private input root")
    declaration = str(Path(declaration).resolve().relative_to(root))
    value = bound(root, declaration, expected_sha256)
    output = path(root, value["output"])
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(
        status="running",
        phase="authority",
        session_started=session_started,
        retry=False,
        full_run_eligible=False,
        independent_replay_complete=False,
        completed_stages=[],
        caller_local_hash_accounting=HASH_ACCOUNTING,
        hash_accounting_scope="Caller sha() only; callee digest/scoring passes are charged in session_operations and cost authority",
        session_operations=SESSION_OPERATIONS,
    )
    _write(output / "session-receipt.json", receipt)
    prepared = None

    def stage(name):
        guard()
        _storage(storage_check)
        receipt["phase"] = name
        receipt["elapsed_since_first_bootstrap_seconds"] = time.monotonic() - session_started
        _write(output / "session-receipt.json", receipt)
        print(
            json.dumps(
                dict(
                    stage=name,
                    utc=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=receipt["elapsed_since_first_bootstrap_seconds"],
                )
            ),
            flush=True,
        )

    try:
        api = load_apis(root)
        guard, deadline = session_guard(session_started)
        guard()
        value, extraction, template, dynamic, cost = authority(root, declaration, expected_sha256, api, guard)
        if cost["estimated_complete_seconds"] > deadline - time.monotonic():
            raise TimeoutError("Complete applicable costs exceed remaining original session")
        stage("staging")
        receipt["staging"] = api.stage_files(
            input_root,
            root,
            value["staging"],
            {n: value["files_sha256"][n] for n in value["staging"]},
            guard=guard,
        )
        stage("checkpoint")
        receipt["checkpoint"] = api.checkpoint(root, guard, storage_check)
        verify_checkpoint(root, guard)
        stage("outcome_blind_extraction")
        extraction_prepared = api.preflight(
            path(root, value["extraction_declaration"]),
            value["files_sha256"][value["extraction_declaration"]],
            root=root,
            data_root=data_root,
            storage_check=storage_check,
            guard=guard,
            checkpoint_path=CHECKPOINT_PATH,
        )
        result = api.extract(
            extraction_prepared,
            output / "extraction",
            session_started=session_started,
            storage_check=storage_check,
        )
        receipt["extraction"] = result
        coverage = json.loads((output / "extraction/coverage.json").read_text())
        if (
            result["status"] != "complete"
            or result["coverage_passed"] is not True
            or result["tables_complete"] is not True
            or result["rows_completed"] != EXPECTED_ROWS
            or coverage["passed"] is not True
            or coverage["rows"] != EXPECTED_ROWS
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
            raise ValueError("Complete extraction and all original coverage cells required before labels")
        receipt["completed_stages"].append("extraction")
        stage("concrete_study_binding")
        if bound(root, value["study_template"], value["files_sha256"][value["study_template"]]) != template:
            raise ValueError("Prospective template changed")
        concrete = copy.deepcopy(template)
        for name in dynamic:
            concrete["files_sha256"][name] = sha(path(root, name))
        restored = copy.deepcopy(concrete)
        for name in dynamic:
            restored["files_sha256"][name] = None
        if restored != template:
            raise ValueError("Only the four declared engineering hashes may change")
        study_declaration = path(root, value["study_declaration"])
        _write(study_declaration, concrete)
        study_sha = sha(study_declaration)
        receipt["study_declaration_sha256"] = study_sha
        stage("original_pair_support")
        prepared = api.prepare(
            study_declaration, study_sha, root=root, storage_check=storage_check, guard=guard
        )
        support = api.support(prepared, guard=guard)
        _write(output / "support.json", support)
        if support["status"] != "passed":
            raise ValueError("Original common pair/class support gate failed")
        receipt["completed_stages"].append("support")
        stage("fixed_comparison")
        run = output / "run"
        execution = api.execute(
            prepared, support, run, guard=guard, storage_check=storage_check, deadline=deadline
        )
        if execution["status"] != "complete" or json.loads((run / "receipt.json").read_text()) != execution:
            raise ValueError("Complete authoritative fixed-comparison receipt required")
        run_sha = sha(run / "receipt.json")
        receipt["run_receipt_sha256"] = run_sha
        receipt["completed_stages"].append("comparison")
        prepared.close()
        prepared = None
        stage("independent_no_fit_replay")
        prepared = api.prepare(
            study_declaration, study_sha, root=root, storage_check=storage_check, guard=guard
        )
        replay = api.replay(
            prepared,
            run,
            output / "replay",
            expected_receipt_sha256=run_sha,
            guard=guard,
            storage_check=storage_check,
            deadline=deadline,
        )
        if (
            replay["status"] != "complete"
            or replay["independent_replay_complete"] is not True
            or replay["outcome_fits"] != 0
        ):
            raise ValueError("Independent no-fit replay is mandatory")
        receipt["independent_replay_complete"] = True
        receipt["completed_stages"].append("replay")
        prepared.close()
        prepared = None
        stage("final_source_input_raw_artifact_postcheck")
        verify(root, value["source_sha256"], guard)
        verify(root, value["files_sha256"], guard, skip=(CHECKPOINT_PATH,))
        verify_checkpoint(root, guard)
        bound(root, declaration, expected_sha256)
        raw = bound(root, extraction["raw_manifest"], value["files_sha256"][extraction["raw_manifest"]])
        if api.raw_post(extraction_prepared, guard) != dict(files_sha256=raw["files_sha256"], hash_errors={}):
            raise ValueError("Final raw source authority differs")
        original, manifest = api.verify_run(run, run_sha, guard)
        if not api.same(
            json.loads((output / "replay/review.json").read_text()),
            json.loads((run / "summary.json").read_text()),
        ):
            raise ValueError("Independent replay review differs from trusted run summary")
        if json.loads((output / "replay/receipt.json").read_text()) != replay:
            raise ValueError("Independent replay receipt changed")
        members = {"run/" + n: (run / n, pin) for n, pin in manifest["files_sha256"].items()}
        members.update(
            {
                "run/receipt.json": (run / "receipt.json", run_sha),
                "run/artifact-manifest.json": (
                    run / "artifact-manifest.json",
                    original["artifact_manifest_sha256"],
                ),
            }
        )
        for name, origin in {
            "support.json": output / "support.json",
            "replay/receipt.json": output / "replay/receipt.json",
            "replay/review.json": output / "replay/review.json",
            "study-declaration.json": study_declaration,
            "study-template.json": path(root, value["study_template"]),
            "session-declaration.json": path(root, declaration),
            **{"extraction/" + Path(n).name: path(root, n) for n in dynamic if "/tables/" not in n},
            "extraction/writer-receipt.json": output / "extraction/tables/receipt.json",
        }.items():
            members[name] = origin, sha(origin)
        if sum(origin.stat().st_size for origin, pin in members.values()) > 250000000:
            raise RuntimeError("Bounded final result bundle exceeds250MB")
        bundle = output / "bundle"
        bundle.mkdir()
        for name, (origin, pin) in members.items():
            guard()
            target = path(bundle, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(origin, target)
        pins = {n: pin for n, (origin, pin) in members.items()}
        _write(bundle / "artifact-manifest.json", dict(files_sha256=pins))
        verify(bundle, pins, guard)
        api.verify_run(run, run_sha, guard)
        guard()
        _storage(storage_check)
        receipt.update(
            status="work_complete_pending_supervisor",
            decision=execution["decision"],
            bundle_manifest_sha256=sha(bundle / "artifact-manifest.json"),
            session_declaration_sha256=expected_sha256,
            elapsed_since_first_bootstrap_seconds=time.monotonic() - session_started,
        )
        _write(output / "session-receipt.json", receipt)
        return receipt
    except BaseException as error:
        receipt.update(status="failed", original_error=str(error), error_type=type(error).__name__)
        for name, value_to_write in (
            ("original-error.txt", traceback.format_exc()),
            ("session-receipt.json", receipt),
        ):
            try:
                if name.endswith(".txt"):
                    (output / name).write_text(value_to_write)
                else:
                    _write(output / name, value_to_write)
            except BaseException:
                pass
        raise
    finally:
        if prepared is not None:
            try:
                prepared.close()
            except BaseException:
                if receipt["status"] != "failed":
                    raise


def publish(output, supervisor, supervisor_sha256, *, guard):
    output = Path(output)
    receipt = json.loads((output / "session-receipt.json").read_text())
    try:
        guard()
        if (
            receipt["status"] != "work_complete_pending_supervisor"
            or not receipt["independent_replay_complete"]
        ):
            raise ValueError("Replayed complete worker result required")
        supervised = bound(Path(supervisor).parent, Path(supervisor).name, supervisor_sha256)
        if (
            supervised["status"] != "complete"
            or supervised["exit_code"] != 0
            or supervised["session_started"] != receipt["session_started"]
            or supervised["phase"] != "staging_identity"
            or supervised["engineering_started"] is not None
        ):
            raise ValueError("Successful independent whole-session supervisor required")
        if sha(output / "bundle/artifact-manifest.json") != receipt["bundle_manifest_sha256"]:
            raise ValueError("Final bundle authority changed")
        bundle = output / "bundle"
        manifest = json.loads((bundle / "artifact-manifest.json").read_text())
        paths = list(bundle.rglob("*"))
        members = {str(p.relative_to(bundle)) for p in paths if p.is_file()}
        if any(p.is_symlink() for p in paths) or members != set(manifest["files_sha256"]) | {
            "artifact-manifest.json"
        }:
            raise ValueError("Final bundle membership changed")
        verify(bundle, manifest["files_sha256"], guard)
        guard()
        receipt.update(status="complete", supervisor_sha256=supervisor_sha256)
        _write(output / "bundle/receipt.json", receipt)
        _write(output / "session-receipt.json", receipt)
        return receipt
    except BaseException as error:
        receipt.update(
            status="failed", phase="publication", original_error=str(error), error_type=type(error).__name__
        )
        try:
            _write(output / "session-receipt.json", receipt)
            (output / "original-error.txt").write_text(traceback.format_exc())
        except BaseException:
            pass
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "input-root", "data-root", "declaration", "declaration-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--session-start-monotonic", type=float, required=True)
    parser.add_argument("--worker-parent-pid", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    if args.session_start_monotonic > BOOTSTRAP_STARTED:
        raise ValueError("Original clock must precede caller bootstrap")
    value = bound(root, str(Path(args.declaration).resolve().relative_to(root)), args.declaration_sha256)
    guard, _ = session_guard(args.session_start_monotonic)
    verify(root, value["source_sha256"], guard)
    if Path(__file__).resolve() != path(root, ENTRY):
        raise ValueError("Actual staged full-session entrypoint required")
    cycle = load_script(root, "kaggle_cycle")
    roots = [Path(args.input_root), root]

    def storage():
        return {
            key: cycle.dataset_bytes(roots, allocated=allocated)
            for key, allocated in (("logical_bytes", False), ("allocated_bytes", True))
        }

    if args.worker_parent_pid is not None:
        if args.worker_parent_pid != os.getppid():
            raise ValueError("Live independent session supervisor required")
        run_session(
            args.declaration,
            args.declaration_sha256,
            root=root,
            input_root=args.input_root,
            data_root=args.data_root,
            session_started=args.session_start_monotonic,
            storage_check=storage,
        )
        return 0
    wrapper = load_script(root, "kaggle_ecgfounder_engineering")
    output = path(root, value["output"])
    supervisor = path(root, value["output"] + "-watchdog")
    wrapper.supervise(
        [
            sys.executable,
            str(path(root, ENTRY)),
            *(sys.argv[1:] if argv is None else argv),
            "--worker-parent-pid",
            str(os.getpid()),
        ],
        project=root,
        output=supervisor,
        roots=roots,
        session_started=args.session_start_monotonic,
        engineering_started=None,
        phase="staging_identity",
        table_root=output / "extraction/tables",
    )
    guard()
    publish(output, supervisor / "receipt.json", sha(supervisor / "receipt.json"), guard=guard)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
