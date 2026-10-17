#!/usr/bin/env python3
"""Source-only packaging and independently guarded private engineering bootstrap; never submit."""

import time

BOOTSTRAP_STARTED = globals().get("BOOTSTRAP_STARTED", time.monotonic())
import argparse  # noqa: E402
import base64  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import tomllib  # noqa: E402
import traceback  # noqa: E402
import zlib  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

SESSION_SECONDS, CLEANUP_SECONDS, ENGINEERING_SECONDS = 41400, 300, 3600
DIAGNOSTIC_BYTES, TABLE_BYTES, TABLE_RECEIPT_BYTES = 250000000, 2619117060, 250000
PROJECT_BYTES, RSS_BYTES = 25000000000, 10000000000
DEADLINE = datetime(2026, 10, 18, 3, 30, tzinfo=timezone.utc)
PHASE_SECONDS = dict(staging_identity=41100, synthetic_cpu=1800, synthetic_table=900, dense_native=600)


def safe_text(value):
    value = re.sub(r"https?://[^\s'\"<>]+", "<URL redacted>", str(value))
    value = re.sub(r"(?i)authorization\s*[:=]\s*(?:bearer\s+)?[^\s,;]+", "authorization=<redacted>", value)
    return re.sub(
        r"(?i)(token|authorization|password|secret|api_key)\s*[:=]\s*[^\s,;]+", r"\1=<redacted>", value
    )


def safe_lines(chunks):
    pending = ""
    discarding = False
    for chunk in chunks:
        pending += chunk.decode("utf-8", errors="replace")
        while "\n" in pending:
            line, pending = pending.split("\n", 1)
            if not discarding:
                yield safe_text(line) + "\n" if len(line) <= 65536 else "<oversized output line redacted>\n"
            discarding = False
        if len(pending) > 65536:
            if not discarding:
                yield "<oversized output line redacted>\n"
            pending = ""
            discarding = True
    if pending and not discarding:
        yield safe_text(pending)


def load_script(project, name):
    spec = importlib.util.spec_from_file_location(name, Path(project) / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def remaining_seconds(
    *, session_started, engineering_started, phase_started, phase_limit, now=None, calendar_remaining=None
):
    now = time.monotonic() if now is None else now
    values = (session_started, phase_started, phase_limit, now)
    if (
        any(not math.isfinite(v) or v < 0 for v in values)
        or phase_started < session_started
        or now < phase_started
        or phase_limit <= 0
    ):
        raise ValueError("Nonrenewable finite clock values required")
    limits = [
        SESSION_SECONDS - CLEANUP_SECONDS - (now - session_started),
        phase_limit - (now - phase_started),
    ]
    if engineering_started is not None:
        if not math.isfinite(engineering_started) or not session_started <= engineering_started <= now:
            raise ValueError("Engineering clock must follow first bootstrap")
        limits.append(ENGINEERING_SECONDS - (now - engineering_started))
    limits.append(
        (DEADLINE - datetime.now(timezone.utc)).total_seconds()
        if calendar_remaining is None
        else calendar_remaining
    )
    return min(limits)


def parse_group_rss(stdout, group):
    total, found = 0, False
    for line in stdout.splitlines():
        fields = line.split()
        if len(fields) != 3:
            raise RuntimeError("Unavailable process-group RSS inventory")
        pid, pgid, rss = map(int, fields)
        if min(pid, pgid, rss) < 0:
            raise RuntimeError("Invalid actual process RSS")
        if pgid == group:
            total += rss * 1024
            found = True
    if not found:
        raise RuntimeError("Unavailable actual child process-group RSS")
    return total


def supervise(
    command, *, project, output, roots, session_started, engineering_started, phase, table_root=None, env=None
):
    """Parent measures the whole child group independently of cooperative worker checks."""
    cycle = load_script(project, "kaggle_cycle")
    helper = load_script(project, "kaggle_ecgfounder_probe")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(
        status="running",
        phase=phase,
        session_started=session_started,
        phase_started=time.monotonic(),
        engineering_started=engineering_started,
        peak_process_group_rss_bytes=0,
        full_run_eligible=False,
        actual_outcome_fits=0,
        retry=False,
    )
    helper.write_json(output / "receipt.json", receipt)
    process = None
    errors = []
    reader = None

    def drain():
        try:
            with (output / "child.log").open("xb") as stream:
                for text in safe_lines(iter(lambda: os.read(process.stdout.fileno(), 65536), b"")):
                    block = text.encode("utf-8")
                    if stream.tell() + len(block) > DIAGNOSTIC_BYTES:
                        raise RuntimeError("Bounded diagnostic log exceeded")
                    stream.write(block)
                    stream.flush()
                    sys.stdout.write(text)
                    sys.stdout.flush()
        except BaseException as error:
            errors.append(helper.sanitized(error))

    def resources():
        occupied = {
            key: cycle.dataset_bytes(roots, allocated=allocated)
            for key, allocated in (("apparent", False), ("allocated", True))
        }
        receipt["storage"] = occupied
        receipt["resource_snapshot_utc"] = datetime.now(timezone.utc).isoformat()
        if max(occupied.values()) + DIAGNOSTIC_BYTES > PROJECT_BYTES:
            raise RuntimeError("Global storage cap/reserve exceeded")
        exclusions = [
            Path(project) / "data",
            Path(project) / "dependencies",
            Path(project) / "reports/ml-investigation/next-cycle/ecgfounder-cycle-20261013/weights",
        ]
        diagnostic_usage = receipt["diagnostic_storage"] = {}
        table_usage_snapshot = receipt["table_storage"] = {}
        for metric, allocated in (("apparent", False), ("allocated", True)):
            table_usage = (
                cycle.dataset_bytes([table_root], allocated=allocated) if table_root is not None else 0
            )
            table_usage_snapshot[metric] = table_usage
            if table_usage > TABLE_BYTES + TABLE_RECEIPT_BYTES + 65536:
                raise RuntimeError("Declared one-table allocation exceeded")
            diagnostic_usage[metric] = (
                cycle.dataset_bytes([project], allocated=allocated)
                - cycle.dataset_bytes(exclusions, allocated=allocated)
                - table_usage
            )
        if max(diagnostic_usage.values()) > DIAGNOSTIC_BYTES:
            raise RuntimeError("Diagnostic/source output allowance exceeded")
        return occupied

    try:
        resources()
        if (
            remaining_seconds(
                session_started=session_started,
                engineering_started=engineering_started,
                phase_started=receipt["phase_started"],
                phase_limit=PHASE_SECONDS[phase],
            )
            <= 0
        ):
            raise TimeoutError("Session/calendar/engineering phase deadline exhausted before launch")
        process = subprocess.Popen(
            command,
            cwd=project,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        last_storage = last_save = 0.0
        while process.poll() is None:
            if errors:
                raise RuntimeError("Child output drain failed: " + errors[0])
            remaining = remaining_seconds(
                session_started=session_started,
                engineering_started=engineering_started,
                phase_started=receipt["phase_started"],
                phase_limit=PHASE_SECONDS[phase],
            )
            if remaining <= 0:
                raise TimeoutError("Nonrenewable session/engineering/phase/calendar ceiling exhausted")
            measured = subprocess.run(
                ["ps", "-eo", "pid=,pgid=,rss="],
                capture_output=True,
                text=True,
                timeout=min(2, max(0.01, remaining)),
                check=True,
            )
            try:
                rss = parse_group_rss(measured.stdout, process.pid)
            except RuntimeError:
                if process.poll() is not None:
                    break
                raise
            receipt["peak_process_group_rss_bytes"] = max(receipt["peak_process_group_rss_bytes"], rss)
            if rss > RSS_BYTES:
                raise RuntimeError("Actual child process-group RSS cap exceeded")
            now = time.monotonic()
            if now - last_storage >= 5:
                receipt["storage"] = resources()
                last_storage = now
            if now - last_save >= 10:
                receipt["elapsed_seconds"] = now - receipt["phase_started"]
                helper.write_json(output / "receipt.json", receipt)
                print(
                    json.dumps(
                        dict(
                            utc=datetime.now(timezone.utc).isoformat(),
                            phase=phase,
                            pid=process.pid,
                            rss_bytes=rss,
                            elapsed_seconds=receipt["elapsed_seconds"],
                        ),
                        allow_nan=False,
                    ),
                    flush=True,
                )
                last_save = now
            try:
                process.wait(timeout=min(0.25, max(0.01, remaining)))
            except subprocess.TimeoutExpired:
                pass
        if process.returncode != 0:
            raise RuntimeError(f"Engineering child exited {process.returncode}")
        if (
            remaining_seconds(
                session_started=session_started,
                engineering_started=engineering_started,
                phase_started=receipt["phase_started"],
                phase_limit=PHASE_SECONDS[phase],
            )
            <= 0
        ):
            raise TimeoutError("Deadline exceeded at child completion")
        receipt["storage"] = resources()
        receipt["status"] = "complete"
    except BaseException as error:
        receipt.update(
            status="failed",
            original_error=safe_text(error),
            error_type=type(error).__name__,
            sanitized_traceback=safe_text(traceback.format_exc()),
        )
        raise (TimeoutError if isinstance(error, TimeoutError) else RuntimeError)(safe_text(error)) from None
    finally:
        if process is not None:
            try:
                cycle._stop_group(process)
            except BaseException as error:
                errors.append("Process-group cleanup failed: " + helper.sanitized(error))
            if reader is not None:
                reader.join(timeout=2)
            receipt["exit_code"] = process.returncode
        if errors or reader is not None and reader.is_alive():
            receipt.update(
                status="failed",
                output_errors=errors,
                original_error=receipt.get("original_error", "Child output stream incomplete"),
            )
        receipt["elapsed_seconds"] = time.monotonic() - receipt["phase_started"]
        helper.write_json(output / "receipt.json", receipt)
    if receipt["status"] != "complete":
        raise RuntimeError(receipt["original_error"])
    return receipt


def package(root, declaration, expected_sha256, output, identity, release, release_sha256):
    root = Path(root).resolve()
    declaration = Path(declaration).resolve()
    output = Path(output)
    if not re.fullmatch(
        r"sumit1703/cardiotwin-ecgfounder-engineering-[0-9]{8}-[a-z0-9-]+", identity
    ) or not re.fullmatch("[a-f0-9]{64}", release_sha256):
        raise ValueError("Unique engineering identity/detached release digest required")
    if not output.resolve().is_relative_to(root):
        raise ValueError("Package output must remain within bound root")
    helper = load_script(root, "run_ecgfounder_engineering")
    old = load_script(root, "kaggle_ecgfounder_probe")
    if not declaration.is_relative_to(root) or helper.sha(declaration) != expected_sha256:
        raise ValueError("Reviewed declaration digest/path differs")
    value = json.loads(declaration.read_text())
    if value.get("version") != "ecgfounder-intended-worker-engineering-v1":
        raise ValueError("Fixed engineering declaration required")
    release = Path(release).resolve()
    if not release.is_relative_to(root) or helper.sha(release) != release_sha256:
        raise ValueError("Detached reviewed release path/hash differs")
    release_name = str(release.relative_to(root))
    receipts = helper.release_receipts(root, json.loads(release.read_text()))
    sources = value["source_sha256"]
    manifest = {}
    for group in (
        sources,
        value.get("package_files_sha256", {}),
        value.get("scientific_files_sha256", {}),
        {str(declaration.relative_to(root)): expected_sha256, release_name: release_sha256},
        receipts,
    ):
        for name, pin in group.items():
            if name in manifest and manifest[name] != pin:
                raise ValueError("Conflicting package pins")
            manifest[name] = pin
    payload = dict(
        files={},
        manifest=manifest,
        identity=old.manifest_identity(manifest),
        identity_name=identity,
        declaration=str(declaration.relative_to(root)),
        declaration_sha256=expected_sha256,
        release=release_name,
        release_sha256=release_sha256,
    )
    total = 0
    for name, pin in manifest.items():
        path = helper.bound_path(root, name)
        if path.suffix == ".py" and name not in sources:
            raise ValueError("Every staged Python source requires outer binding")
        if path.suffix not in (".py", ".json", ".md", ".toml", ".lock") and path.name != "LICENSE":
            raise ValueError("Only bounded source/metadata in capsule")
        if name.startswith("data/") or "weights" in path.parts:
            raise ValueError("No cache/raw/checkpoint content in source capsule")
        if path.stat().st_size > 5000000:
            raise ValueError("Source member exceeds bound")
        content = path.read_bytes()
        total += len(content)
        if hashlib.sha256(content).hexdigest() != pin:
            raise ValueError("Package source differs")
        payload["files"][name] = base64.b64encode(content).decode("ascii")
    if total > 10000000:
        raise ValueError("Source capsule exceeds10MB")
    if output.exists():
        raise FileExistsError("Fresh package directory required")
    # Submission remains an explicit later external action; this function writes files only.
    metadata = old.kernel_metadata(
        identity.replace("cardiotwin-ecgfounder-engineering-", "cardiotwin-ecgfounder-probe-")
    )
    metadata.update(id=identity, title=identity.split("/")[1])
    wrapper = (root / "scripts/kaggle_ecgfounder_engineering.py").read_text()
    bootstrap = (
        "import time\nBOOTSTRAP_STARTED=time.monotonic()\nimport base64,json,zlib\nPAYLOAD=json.loads(zlib.decompress(base64.b64decode("
        + repr(base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode())
        + ")))\nexec(compile("
        + repr(wrapper)
        + ", 'engineering-bootstrap.py', 'exec'), globals())\nbootstrap(PAYLOAD, session_started=BOOTSTRAP_STARTED)\n"
    )
    output.mkdir(parents=True)
    (output / "train.py").write_text(bootstrap)
    old.write_json(output / "kernel-metadata.json", metadata)
    old.write_json(
        output / "package-receipt.json",
        dict(
            source_identity=payload["identity"],
            source_files=len(manifest),
            source_bytes=total,
            train_sha256=helper.sha(output / "train.py"),
            execution_released=False,
            full_run_eligible=False,
        ),
    )
    return payload


def remote(payload, *, session_started, working=Path("/kaggle/working"), inputs=Path("/kaggle/input")):
    """Concrete detached release is source-pinned before any real data access."""
    working, inputs = Path(working), Path(inputs)
    if working != Path("/kaggle/working") or inputs != Path("/kaggle/input"):
        raise ValueError("Fixed intended-worker roots required")
    if not re.fullmatch(
        r"sumit1703/cardiotwin-ecgfounder-engineering-[0-9]{8}-[a-z0-9-]+", payload["identity_name"]
    ):
        raise ValueError("Unsafe attempt identity")
    name = payload["identity_name"].split("/")[1]
    project = working / name
    if project.exists():
        raise FileExistsError("Existing attempt; no automatic retry")
    # Decode and validate the entire source map before creating the attempt.
    decoded = {}
    for name, encoded in payload["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or str(relative) != name:
            raise ValueError("Unsafe capsule path")
        content = base64.b64decode(encoded, validate=True)
        if hashlib.sha256(content).hexdigest() != payload["manifest"][name]:
            raise ValueError("Capsule source checksum differs")
        decoded[name] = content
    if set(decoded) != set(payload["manifest"]):
        raise ValueError("Capsule source membership differs")
    if (
        hashlib.sha256(
            json.dumps(payload["manifest"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        != payload["identity"]
    ):
        raise ValueError("Capsule identity differs")
    project.mkdir()
    for name, content in decoded.items():
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    sys.path.insert(0, str(project / "src"))
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PYTHONPATH"] = str(project / "src")
    scratch = project / "scratch"
    scratch.mkdir()
    os.environ["TMPDIR"] = str(scratch)
    driver = load_script(project, "run_ecgfounder_engineering")
    old = load_script(project, "kaggle_ecgfounder_probe")
    declaration = driver.bound_path(project, payload["declaration"])
    value = json.loads(declaration.read_text())
    value["_sha256"] = payload["declaration_sha256"]
    release = driver.bound_path(project, payload["release"])
    release_sha = payload["release_sha256"]
    if driver.sha(release) != release_sha:
        raise ValueError("Concrete release digest differs")
    driver.validate_release(value, json.loads(release.read_text()), root=project)
    roots = [inputs, working]
    output = project / "engineering-results"
    output.mkdir()
    receipt = dict(
        status="running",
        source_identity=payload["identity"],
        phases={},
        full_run_eligible=False,
        actual_outcome_fits=0,
        accepted_remote_upper_seconds=dict.fromkeys(driver.COMPONENTS),
        session_started=session_started,
        retry=False,
    )

    def save():
        receipt["bootstrap_checksum_calls"] = driver.HASH_ACCOUNTING
        receipt["source_overlay_bytes"] = sum(len(content) for content in decoded.values())
        receipt["dependency_download_bytes"] = None
        receipt["elapsed_seconds"] = time.monotonic() - session_started
        old.write_json(output / "bootstrap-receipt.json", receipt)

    def stop(signum, frame):
        raise TimeoutError("Original bootstrap work/calendar ceiling exhausted")

    signal.signal(signal.SIGALRM, stop)
    signal.setitimer(
        signal.ITIMER_REAL,
        max(
            0.001,
            min(
                41100 - (time.monotonic() - session_started),
                (DEADLINE - datetime.now(timezone.utc)).total_seconds(),
            ),
        ),
    )
    engineering_started = None
    try:
        save()
        # Dependency traffic is inside the original bootstrap clock and storage ledger.
        dependencies = tomllib.loads((project / "pyproject.toml").read_text())["project"]["dependencies"]
        dependency_scratch = project / "dependencies" / ".pip-tmp"
        dependency_scratch.mkdir(parents=True)
        receipt["dependency_bootstrap"] = supervise(
            [
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
            ],
            project=project,
            output=output / "dependencies-guard",
            roots=roots,
            session_started=session_started,
            engineering_started=None,
            phase="staging_identity",
            env={**os.environ, "TMPDIR": str(dependency_scratch)},
        )
        sys.path.insert(0, str(project / "dependencies"))
        os.environ["PYTHONPATH"] = str(project / "src") + os.pathsep + str(project / "dependencies")
        for phase in ("staging_identity", "synthetic_cpu", "synthetic_table", "dense_native"):
            if phase == "synthetic_cpu":
                checkpoint = driver.bound_path(project, driver.CYCLE + "/weights/1_lead_ECGFounder.pth")
                old.reserve_storage(
                    roots, checkpoint_bytes=old.CHECKPOINT_BYTES + TABLE_BYTES + TABLE_RECEIPT_BYTES
                )
                receipt["checkpoint_transfer"] = old.acquire_checkpoint(
                    "https://huggingface.co/PKUDigitalHealth/ECGFounder/resolve/"
                    + old.REVISION
                    + "/1_lead_ECGFounder.pth",
                    checkpoint,
                    old.CHECKPOINT_BYTES,
                    old.CHECKPOINT_SHA,
                    lambda event: print(json.dumps(event, allow_nan=False), flush=True),
                )
                engineering_started = time.monotonic()
                receipt["engineering_started"] = engineering_started
                save()
                benchmark = load_script(project, "benchmark_ecgfounder_cost")
                command = [
                    sys.executable,
                    str(project / "scripts/benchmark_ecgfounder_cost.py"),
                    "--output",
                    str((project / benchmark.RECIPE_PATH).parent / "benchmarks" / value["benchmark_attempt"]),
                    "--requirements",
                    str(project / benchmark.RECIPE_PATH),
                    "--source-manifest",
                    str(driver.bound_path(project, value["benchmark_source_manifest"])),
                ]
            else:
                command = [
                    sys.executable,
                    str(project / "scripts/run_ecgfounder_engineering.py"),
                    "--phase",
                    phase,
                    "--declaration",
                    str(declaration),
                    "--declaration-sha256",
                    payload["declaration_sha256"],
                    "--release",
                    str(release),
                    "--release-sha256",
                    release_sha,
                    "--output",
                    str(output / phase),
                ]
            receipt["phases"][phase] = supervise(
                command,
                project=project,
                output=output / (phase + "-guard"),
                roots=roots,
                session_started=session_started,
                engineering_started=engineering_started,
                phase=phase,
                table_root=output / "synthetic_table/fixture/tables",
            )
            save()
        receipt["status"] = "complete_engineering_measurements_only"
    except BaseException as error:
        receipt.update(
            status="failed",
            error_type=type(error).__name__,
            original_error=safe_text(error),
            sanitized_traceback=safe_text(traceback.format_exc()),
        )
        raise RuntimeError(safe_text(error)) from None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        save()


def bootstrap(payload, *, session_started):
    try:
        return remote(payload, session_started=session_started)
    except BaseException as error:
        print(
            json.dumps(
                dict(status="failed", original_error_type=type(error).__name__, safe_error=safe_text(error))
            ),
            flush=True,
        )
        raise SystemExit(1) from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--declaration-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--release-sha256", required=True)
    args = parser.parse_args(argv)
    package(
        args.root,
        args.declaration,
        args.declaration_sha256,
        args.output,
        args.identity,
        args.release,
        args.release_sha256,
    )


if __name__ == "__main__" and "PAYLOAD" not in globals():
    try:
        main()
    except Exception as error:
        print(safe_text(error), file=sys.stderr, flush=True)
        raise SystemExit(1) from None
