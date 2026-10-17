"""Source-only capsule and one supervised ECGFounder full session; never submit or retry."""

# ruff: noqa: E402
import time

BOOTSTRAP_STARTED = globals().get("BOOTSTRAP_STARTED", time.monotonic())

import argparse
import base64
import hashlib
import importlib.util
import json
import math
import os
import re
import signal
import subprocess
import sys
import tomllib
import traceback
import zlib
from datetime import datetime, timezone
from pathlib import Path

ENTRY = "scripts/kaggle_ecgfounder_session.py"
CALLER = "scripts/run_ecgfounder_session.py"
CYCLE = "reports/ml-investigation/next-cycle/ecgfounder-cycle-20261013"
CALENDAR = CYCLE + "/calendar-addendum-20261017.json"
CALENDAR_SHA256 = "a804c7e2795a37d1ce13fd08539487ebf1162714b18e75c4cf29461edf058822"
WORKING, INPUTS = Path("/kaggle/working"), Path("/kaggle/input")
REQUIRED = (
    ENTRY,
    CALLER,
    "scripts/kaggle_ecgfounder_engineering.py",
    "scripts/kaggle_ecgfounder_probe.py",
    "scripts/kaggle_cycle.py",
    "pyproject.toml",
    "uv.lock",
)
BOOTSTRAP_OPERATIONS = dict(
    source_overlay=1,
    metadata_validation=2,
    calendar_before_overlay=1,
    supervised_worker=1,
    dependency_bootstrap=1,
    worker_source_verification=1,
    direct_caller=1,
    source_stamp_hashes=2,
    seal_bound_receipt_hashes=3,
    completion_seal=1,
)

SESSION_OPERATION_SCOPE = "conservative_upper_counts_including_standalone_publication"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bound_path(root, name):
    root, relative = Path(root).resolve(), Path(name)
    if relative.is_absolute() or ".." in relative.parts or str(relative) != name or name in ("", "."):
        raise ValueError("Canonical relative capsule path required")
    target = root / relative
    if any(p.is_symlink() for p in (target, *target.parents) if p.is_relative_to(root)):
        raise ValueError("Capsule symlink forbidden")
    return target


def read_bound(root, name, digest):
    target = bound_path(root, name)
    if sha(target) != digest:
        raise ValueError("Bound capsule identity changed: " + name)
    return json.loads(target.read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_script(project, name):
    spec = importlib.util.spec_from_file_location(
        "session_bootstrap_" + name, Path(project) / "scripts" / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metadata(project, declaration, digest):
    value = read_bound(project, declaration, digest)
    files, source = value["files_sha256"], value["source_sha256"]
    if (
        value["version"] != "ecgfounder-session-v1"
        or not set(REQUIRED) <= set(source)
        or files.get(CALENDAR) != CALENDAR_SHA256
    ):
        raise ValueError("Complete caller/bootstrap/calendar source authority required")
    extraction = read_bound(project, value["extraction_declaration"], files[value["extraction_declaration"]])
    template = read_bound(project, value["study_template"], files[value["study_template"]])
    if extraction["source_sha256"] != source or template["source_sha256"] != source:
        raise ValueError("Uniform prospective source identity required")
    if (
        extraction["files_sha256"].get(CALENDAR) != CALENDAR_SHA256
        or template["files_sha256"].get(CALENDAR) != CALENDAR_SHA256
    ):
        raise ValueError("Calendar addendum must bind extraction and prospective study")
    cost = read_bound(project, extraction["cost_receipt"], files[extraction["cost_receipt"]])
    review = read_bound(project, extraction["cost_review"], files[extraction["cost_review"]])
    if (
        review["status"] != "passed"
        or review["source_sha256"] != source
        or review["cost_sha256"] != files[extraction["cost_receipt"]]
        or any(item.get("bootstrap_operations") != BOOTSTRAP_OPERATIONS for item in (cost, review))
        or any(item.get("session_operation_scope") != SESSION_OPERATION_SCOPE for item in (cost, review))
    ):
        raise ValueError("Reviewed exact bootstrap/dependency/overlay/seal operation costs required")
    calendar = read_bound(project, CALENDAR, CALENDAR_SHA256)
    deadline = datetime.fromisoformat(calendar["new_work_calendar_deadline_utc"])
    return value, deadline


def package(root, declaration, digest, output, identity):
    root = Path(root).resolve()
    declaration = str(Path(declaration).resolve().relative_to(root))
    value, _ = metadata(root, declaration, digest)
    if not re.fullmatch(r"sumit1703/cardiotwin-ecgfounder-session-[0-9]{8}-[a-z0-9-]+", identity):
        raise ValueError("Unique private session identity required")
    helper = load_script(root, "kaggle_ecgfounder_probe")
    excluded = set(value["staging"]) | {CYCLE + "/weights/1_lead_ECGFounder.pth"}
    manifest = {}
    for group in (
        value["source_sha256"],
        {n: pin for n, pin in value["files_sha256"].items() if n not in excluded},
        {declaration: digest},
    ):
        for name, pin in group.items():
            if name in manifest and manifest[name] != pin:
                raise ValueError("Conflicting capsule bindings")
            manifest[name] = pin
    files, total = {}, 0
    for name, pin in manifest.items():
        target = bound_path(root, name)
        if (
            name.startswith("data/")
            or "weights" in target.parts
            or target.suffix not in (".py", ".json", ".md", ".toml", ".lock")
            and target.name != "LICENSE"
        ):
            raise ValueError("Only source and small metadata may enter capsule")
        if target.suffix == ".py" and name not in value["source_sha256"]:
            raise ValueError("Every executed Python member must be source bound")
        if target.stat().st_size > 5000000:
            raise ValueError("Source member exceeds5MB")
        content = target.read_bytes()
        total += len(content)
        if total > 10000000 or hashlib.sha256(content).hexdigest() != pin:
            raise ValueError("Capsule bytes/hash exceed authority")
        files[name] = base64.b64encode(content).decode("ascii")
    payload = dict(
        files=files,
        manifest=manifest,
        identity=helper.manifest_identity(manifest),
        identity_name=identity,
        declaration=declaration,
        declaration_sha256=digest,
    )
    output = Path(output).resolve()
    if not output.is_relative_to(root):
        raise ValueError("Local package output must stay within root")
    output.mkdir(parents=True, exist_ok=False)
    wrapper = (root / ENTRY).read_text()
    packed = base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode()
    (output / "train.py").write_text(
        "import time\nBOOTSTRAP_STARTED=time.monotonic()\nimport base64,json,zlib\nPAYLOAD=json.loads(zlib.decompress(base64.b64decode("
        + repr(packed)
        + ")))\nBOOTSTRAP_NAMESPACE=dict(__name__='ecgfounder_session_bootstrap', BOOTSTRAP_STARTED=BOOTSTRAP_STARTED)\nexec(compile("
        + repr(wrapper)
        + ", 'session-bootstrap.py', 'exec'), BOOTSTRAP_NAMESPACE)\nBOOTSTRAP_NAMESPACE['remote'](PAYLOAD, session_started=BOOTSTRAP_STARTED)\n"
    )
    kernel = helper.kernel_metadata(
        identity.replace("cardiotwin-ecgfounder-session-", "cardiotwin-ecgfounder-probe-")
    )
    kernel.update(id=identity, title=identity.split("/")[1])
    write(output / "kernel-metadata.json", kernel)
    write(
        output / "package-receipt.json",
        dict(
            source_identity=payload["identity"],
            source_files=len(files),
            source_bytes=total,
            train_sha256=sha(output / "train.py"),
            execution_released=False,
            full_run_eligible=False,
        ),
    )
    return payload


def overlay(project, payload):
    manifest, files = payload["manifest"], payload["files"]
    identity = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if not files or len(files) > 512 or set(files) != set(manifest) or identity != payload["identity"]:
        raise ValueError("Capsule membership/identity differs")
    decoded, total = {}, 0
    for name, encoded in files.items():
        target = bound_path(project, name)
        if (
            len(encoded) > 6666668
            or name.startswith("data/")
            or "weights" in target.parts
            or target.suffix not in (".py", ".json", ".md", ".toml", ".lock")
            and target.name != "LICENSE"
        ):
            raise ValueError("Unsafe source capsule member")
        content = base64.b64decode(encoded, validate=True)
        total += len(content)
        if total > 10000000 or hashlib.sha256(content).hexdigest() != manifest[name]:
            raise ValueError("Source capsule bytes/hash differ")
        decoded[name] = content
    project.mkdir(parents=True, exist_ok=False)
    for name, content in decoded.items():
        target = bound_path(project, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    write(project / "execution-source-manifest.json", dict(files_sha256=manifest, source_identity=identity))


def arm_clock(session_started, deadline):
    now = time.monotonic()
    if not math.isfinite(session_started) or not 0 <= session_started <= now:
        raise ValueError("Finite original first-bootstrap clock required")
    remaining = min(
        session_started + 41100 - now, (deadline - datetime.now(timezone.utc)).total_seconds() - 300
    )
    if remaining <= 0:
        raise TimeoutError("Original session/calendar cleanup reserve exhausted")

    def stop(signum, frame):
        raise TimeoutError("Independent bootstrap/seal hard alarm exhausted")

    signal.signal(signal.SIGALRM, stop)
    signal.setitimer(signal.ITIMER_REAL, remaining)


def remote(payload, *, session_started, working=None, inputs=None):
    working, inputs = Path(working or WORKING), Path(inputs or INPUTS)
    if (
        working != WORKING
        or inputs != INPUTS
        or not re.fullmatch(
            r"sumit1703/cardiotwin-ecgfounder-session-[0-9]{8}-[a-z0-9-]+", payload["identity_name"]
        )
    ):
        raise ValueError("Fixed private session roots/identity required")
    calendar_bytes = base64.b64decode(payload["files"][CALENDAR], validate=True)
    if (
        hashlib.sha256(calendar_bytes).hexdigest() != CALENDAR_SHA256
        or payload["manifest"][CALENDAR] != CALENDAR_SHA256
    ):
        raise ValueError("Prospective calendar addendum differs")
    deadline = datetime.fromisoformat(json.loads(calendar_bytes)["new_work_calendar_deadline_utc"])
    arm_clock(session_started, deadline)
    project = working / payload["identity_name"].split("/")[1]
    receipt = dict(
        status="running",
        session_started=session_started,
        source_identity=payload["identity"],
        bootstrap_operations=BOOTSTRAP_OPERATIONS,
        retry=False,
        full_run_eligible=False,
    )
    output = project / "session-bootstrap"
    try:
        overlay(project, payload)
        value, checked_deadline = metadata(project, payload["declaration"], payload["declaration_sha256"])
        if any(payload["manifest"].get(n) != pin for n, pin in value["source_sha256"].items()):
            raise ValueError("Capsule and executed source authority differ")
        engineering = load_script(project, "kaggle_ecgfounder_engineering")
        if checked_deadline != deadline or engineering.DEADLINE != deadline:
            raise ValueError("Uniform reviewed calendar deadline required")
        output.mkdir(exist_ok=False)
        write(output / "bootstrap-receipt.json", receipt)
        source_stamp = sha(project / "execution-source-manifest.json")
        pip_tmp = project / "dependencies/.pip-tmp"
        pip_tmp.mkdir(parents=True)
        env = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1", "TMPDIR": str(pip_tmp)}
        command = [
            sys.executable,
            str(project / ENTRY),
            "--worker-parent-pid",
            str(os.getpid()),
            "--root",
            str(project),
            "--inputs",
            str(inputs),
            "--declaration",
            payload["declaration"],
            "--declaration-sha256",
            payload["declaration_sha256"],
            "--session-start-monotonic",
            str(session_started),
        ]
        supervisor = engineering.supervise(
            command,
            project=project,
            output=output / "supervisor",
            roots=[inputs, working],
            session_started=session_started,
            engineering_started=None,
            phase="staging_identity",
            table_root=bound_path(project, value["output"]) / "extraction/tables",
            env=env,
        )
        if (
            supervisor["status"] != "complete"
            or supervisor["exit_code"] != 0
            or supervisor["session_started"] != session_started
        ):
            raise ValueError("Actual successful independent supervisor required")
        worker_path = bound_path(project, value["output"] + "/session-receipt.json")
        if worker_path.stat().st_size > 1000000:
            raise ValueError("Bounded worker completion receipt required")
        worker_sha = sha(worker_path)
        worker_receipt = json.loads(worker_path.read_text())
        if (
            worker_receipt["status"] != "work_complete_pending_supervisor"
            or worker_receipt["session_started"] != session_started
            or worker_receipt["session_declaration_sha256"] != payload["declaration_sha256"]
            or not worker_receipt["independent_replay_complete"]
        ):
            raise ValueError("Source-bound replayed pending worker receipt required")
        bundle_manifest = bound_path(project, value["output"] + "/bundle/artifact-manifest.json")
        if (
            bundle_manifest.stat().st_size > 1000000
            or sha(bundle_manifest) != worker_receipt["bundle_manifest_sha256"]
        ):
            raise ValueError("Bounded final bundle manifest changed before seal")
        if sha(project / "execution-source-manifest.json") != source_stamp:
            raise ValueError("Post-supervisor source identity stamp changed")
        supervisor_sha = sha(output / "supervisor/receipt.json")
        if json.loads((output / "supervisor/receipt.json").read_text()) != supervisor:
            raise ValueError("External supervisor receipt changed")
        if max(supervisor["diagnostic_storage"].values()) + 1000000 > 250000000:
            raise RuntimeError("Aggregate diagnostic storage must reserve tiny completion seal")
        receipt.update(
            status="complete",
            session_operation_scope=SESSION_OPERATION_SCOPE,
            worker_receipt_sha256=worker_sha,
            supervisor_receipt_sha256=supervisor_sha,
            bundle_manifest_sha256=worker_receipt["bundle_manifest_sha256"],
            session_declaration_sha256=payload["declaration_sha256"],
            elapsed_seconds=time.monotonic() - session_started,
        )
        # The worker publishes/verifies every bounded artifact under the watchdog; this tiny seal attests actual child exit.
        write(output / "bootstrap-receipt.json", receipt)
        write(worker_path.parent / "bundle/completion-seal.json", receipt)
        return receipt
    except BaseException as error:
        receipt.update(
            status="failed",
            original_error=str(error),
            error_type=type(error).__name__,
            traceback=traceback.format_exc(),
        )
        if output.is_dir():
            try:
                write(output / "bootstrap-receipt.json", receipt)
            except BaseException:
                pass
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def worker(args):
    if args.worker_parent_pid != os.getppid():
        raise ValueError("Live independent outer supervisor required")
    project = Path(args.root).resolve()
    if Path(__file__).resolve() != bound_path(project, ENTRY):
        raise ValueError("Actual staged bootstrap worker required")
    value, _ = metadata(project, args.declaration, args.declaration_sha256)
    for name, pin in value["source_sha256"].items():
        if sha(bound_path(project, name)) != pin:
            raise ValueError("Executed worker source changed")
    dependencies = tomllib.loads((project / "pyproject.toml").read_text())["project"]["dependencies"]
    pip_tmp = project / "dependencies/.pip-tmp"
    if os.environ.get("TMPDIR") != str(pip_tmp) or os.environ.get("OMP_NUM_THREADS") != "1":
        raise ValueError("Bounded dependency scratch and OMP_NUM_THREADS=1 required")
    command = [
        sys.executable,
        "-m",
        "pip",
        "--isolated",
        "install",
        "--quiet",
        "--no-cache-dir",
        "--target",
        str(project / "dependencies"),
        "--index-url",
        "https://pypi.org/simple",
        *dependencies,
        "h5py>=3.14,<4",
    ]
    started = time.monotonic()
    print(
        json.dumps(
            dict(
                stage="dependency_bootstrap",
                utc=datetime.now(timezone.utc).isoformat(),
                requested_packages=len(dependencies) + 1,
                elapsed_seconds=started - args.session_start_monotonic,
            )
        ),
        flush=True,
    )
    subprocess.run(command, cwd=project, check=True)
    write(
        project / "session-bootstrap/dependency-receipt.json",
        dict(
            status="complete",
            session_started=args.session_start_monotonic,
            requested_packages=len(dependencies) + 1,
            elapsed_seconds=time.monotonic() - started,
            dependency_download_bytes=None,
        ),
    )
    scratch = project / "scratch"
    scratch.mkdir(exist_ok=False)
    env = {
        **os.environ,
        "TMPDIR": str(scratch),
        "PYTHONPATH": str(project / "src") + os.pathsep + str(project / "dependencies"),
    }
    caller = [
        sys.executable,
        str(project / CALLER),
        "--root",
        str(project),
        "--input-root",
        args.inputs,
        "--data-root",
        str(bound_path(Path(args.inputs), value["input_data_root"])),
        "--declaration",
        str(bound_path(project, args.declaration)),
        "--declaration-sha256",
        args.declaration_sha256,
        "--session-start-monotonic",
        str(args.session_start_monotonic),
        "--worker-parent-pid",
        str(args.worker_parent_pid),
    ]
    print(
        json.dumps(
            dict(
                stage="direct_session_caller",
                utc=datetime.now(timezone.utc).isoformat(),
                elapsed_seconds=time.monotonic() - args.session_start_monotonic,
            )
        ),
        flush=True,
    )
    os.execve(sys.executable, caller, env)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-parent-pid", type=int, required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--declaration", required=True)
    parser.add_argument("--declaration-sha256", required=True)
    parser.add_argument("--session-start-monotonic", type=float, required=True)
    return worker(parser.parse_args(argv))


if __name__ == "__main__":
    main()
