"""Run one declared offline study stage under an independent nonrenewable watchdog."""
# ruff: noqa: E402

import time

BOOTSTRAP_STARTED = time.monotonic()

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"

from cardiotwin.data import file_hash
from cardiotwin.ecgfounder_probe import read_bound_manifest
from cardiotwin.ecgfounder_runner import DEADLINE, ExecutionGuard, _storage, storage_measurement
from cardiotwin.ecgfounder_study import (
    _artifact_bytes,
    _verify,
    audit_common_support,
    execute_study,
    prepare_study,
    replay_study,
)
from cardiotwin.investigation import _write
from cardiotwin.precursor import fresh_output

SCRIPT = Path(__file__).resolve()


def work_deadline(session_started):
    current = time.monotonic()
    if not math.isfinite(session_started) or not 0 <= session_started <= current:
        raise ValueError("Actual finite first-bootstrap monotonic start required")
    deadline = min(
        session_started + 41100, current + (DEADLINE - datetime.now(timezone.utc)).total_seconds() - 300
    )
    if current >= deadline:
        raise TimeoutError("Original session/calendar work budget exhausted;300 seconds reserved")
    return deadline


def source_authority(root, declaration, digest, guard):
    root = Path(root).resolve()
    payload = read_bound_manifest(Path(declaration), digest)
    names = ("scripts/run_ecgfounder_study.py", "scripts/kaggle_cycle.py")
    if SCRIPT != root / names[0] or any(
        payload["source_sha256"].get(n) != file_hash(root / n) for n in names
    ):
        raise ValueError("Actual CLI/supervisor executed source must be prospectively bound")
    _verify(payload["source_sha256"], root, guard)


def parse_group_rss(stdout, group):
    total, found = 0, False
    for line in stdout.splitlines():
        fields = line.split()
        if len(fields) != 3:
            raise RuntimeError("Unavailable process-group RSS inventory")
        pid, pgid, rss = map(int, fields)
        if min(pid, pgid, rss) < 0:
            raise RuntimeError("Invalid actual process-group RSS")
        if pgid == group:
            total += rss * 1024
            found = True
    if not found:
        raise RuntimeError("Unavailable actual child process-group RSS")
    return total


def group_rss(process, remaining):
    measured = subprocess.run(
        ["ps", "-eo", "pid=,pgid=,rss="],
        capture_output=True,
        text=True,
        timeout=min(2, max(0.01, remaining)),
        check=True,
    )
    return parse_group_rss(measured.stdout, process.pid)


def supervise(command, output, *, root, session_started, deadline, storage_check):
    """Keep a parent alive outside the worker's Python callbacks and blocking native calls."""
    output = fresh_output(output)
    spec = importlib.util.spec_from_file_location(
        "study_cycle_supervisor", Path(root) / "scripts/kaggle_cycle.py"
    )
    cycle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cycle)
    receipt = dict(
        status="running",
        session_started=session_started,
        work_deadline=deadline,
        session_deadline=session_started + 41400,
        cleanup_reserve_seconds=300,
        child_pid=None,
        peak_process_group_rss_bytes=0,
        full_run_eligible=False,
        automatic_retry=False,
    )
    _write(output / "supervisor.json", receipt)
    process = None
    reader = None
    output_errors = []
    guard = ExecutionGuard(session_started=session_started, stage_seconds=41100)

    def drain():
        try:
            with (output / "child.log").open("xb") as stream:
                count = 0
                while block := os.read(process.stdout.fileno(), 65536):
                    count += len(block)
                    if count > 250000000:
                        raise RuntimeError("Study child log allocation exceeded")
                    stream.write(block)
                    stream.flush()
        except BaseException as error:
            output_errors.append(str(error))

    try:
        guard()
        if not math.isfinite(deadline) or deadline > work_deadline(session_started):
            raise ValueError("Watchdog cannot extend original first-bootstrap budget")
        receipt["storage"] = _storage(storage_check)
        process = subprocess.Popen(
            command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True
        )
        receipt["child_pid"] = process.pid
        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        last_storage = 0.0
        while process.poll() is None:
            guard()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Original whole-session/calendar work deadline exhausted")
            if output_errors:
                raise RuntimeError(output_errors[0])
            try:
                rss = group_rss(process, remaining)
            except BaseException:
                if process.poll() is not None:
                    break
                raise
            receipt["peak_process_group_rss_bytes"] = max(receipt["peak_process_group_rss_bytes"], rss)
            if rss > 10000000000:
                raise RuntimeError("Actual child process-group RSS cap exceeded")
            if time.monotonic() - last_storage >= 5:
                receipt["storage"] = _storage(storage_check)
                if _artifact_bytes(output) > 250000000:
                    raise RuntimeError("Study outputs/logs exceed250MB allocation")
                _write(output / "supervisor.json", receipt)
                last_storage = time.monotonic()
            try:
                process.wait(timeout=min(0.25, max(0.001, remaining)))
            except subprocess.TimeoutExpired:
                pass
        guard()
        if time.monotonic() >= deadline:
            raise TimeoutError("Child exceeded original work deadline")
        if process.returncode != 0:
            detail = output / "worker-error.json"
            original = (
                json.loads(detail.read_text())["original_error"]
                if detail.exists()
                else f"Child exited {process.returncode}"
            )
            raise RuntimeError(original)
        receipt["storage"] = _storage(storage_check)
        if _artifact_bytes(output) > 250000000:
            raise RuntimeError("Study outputs/logs exceed250MB allocation")
        receipt["status"] = "complete"
    except BaseException as error:
        receipt.update(status="failed", original_error=str(error), error_type=type(error).__name__)
        try:
            (output / "supervisor-original-error.txt").write_text(traceback.format_exc())
        except BaseException:
            pass
        try:
            _write(output / "supervisor.json", receipt)
        except BaseException:
            pass
    finally:
        if process is not None:
            try:
                cycle._stop_group(process)
                receipt["child_exit_code"] = process.returncode
                receipt["process_group_terminated"] = True
            except BaseException as error:
                receipt.update(
                    status="failed",
                    original_error=receipt.get("original_error", str(error)),
                    termination_error=str(error),
                    process_group_terminated=False,
                )
        if reader is not None:
            reader.join(timeout=2)
            if reader.is_alive() or output_errors:
                receipt.update(
                    status="failed",
                    original_error=receipt.get(
                        "original_error",
                        output_errors[0] if output_errors else "Child output reader did not finish",
                    ),
                )
        receipt["elapsed_since_first_bootstrap_seconds"] = time.monotonic() - session_started
        try:
            _write(output / "supervisor.json", receipt)
        except BaseException as error:
            receipt.update(status="failed", original_error=receipt.get("original_error", str(error)))
    return receipt


def worker(args):
    if args.worker_parent_pid != os.getppid():
        raise ValueError("Study worker requires its live independent supervisor")
    output = Path(args.output)
    guard = ExecutionGuard(session_started=args.session_start_monotonic, stage_seconds=41100)
    deadline = work_deadline(args.session_start_monotonic)

    def check():
        guard()
        if time.monotonic() >= deadline:
            raise TimeoutError("Original work deadline exhausted")

    def storage():
        return storage_measurement(args.root)

    prepared = None
    failed = False
    try:
        check()
        source_authority(args.root, args.declaration, args.declaration_sha256, check)
        prepared = prepare_study(
            args.declaration, args.declaration_sha256, root=args.root, storage_check=storage, guard=check
        )
        target = output / "study"
        if args.mode == "preflight":
            fresh_output(target)
            _write(target / "receipt.json", prepared.receipt)
        elif args.mode == "support":
            support = audit_common_support(prepared, guard=check)
            fresh_output(target)
            _write(target / "support.json", support)
            _write(target / "receipt.json", support)
            return 0 if support["status"] == "passed" else 1
        elif args.mode == "run":
            support = audit_common_support(prepared, guard=check)
            execute_study(prepared, support, target, guard=check, storage_check=storage, deadline=deadline)
        else:
            replay_study(
                prepared,
                args.run_dir,
                target,
                expected_receipt_sha256=args.run_receipt_sha256,
                guard=check,
                storage_check=storage,
                deadline=deadline,
            )
        check()
        return 0
    except BaseException as error:
        failed = True
        original = traceback.format_exc()
        # Preserve the triggering exception even when storage or cleanup has also failed.
        try:
            print(original, file=sys.stderr, flush=True)
        except BaseException:
            pass
        try:
            (output / "original-error.txt").write_text(original)
        except BaseException:
            pass
        try:
            _write(
                output / "worker-error.json",
                dict(
                    status="failed",
                    original_error=str(error),
                    error_type=type(error).__name__,
                    full_run_eligible=False,
                ),
            )
        except BaseException:
            pass
        raise
    finally:
        if prepared is not None:
            try:
                prepared.close()
            except BaseException:
                if not failed:
                    raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "declaration", "declaration-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--mode", required=True, choices=("preflight", "support", "run", "replay"))
    parser.add_argument(
        "--session-start-monotonic",
        required=True,
        type=float,
        help="Original first-bootstrap monotonic start; never reset between stages",
    )
    parser.add_argument("--run-dir")
    parser.add_argument(
        "--run-receipt-sha256",
        help="Externally trusted complete run receipt digest; required for replay before binary loading",
    )
    parser.add_argument("--worker-parent-pid", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.mode == "replay" and (not args.run_dir or not args.run_receipt_sha256):
        parser.error("replay requires --run-dir and externally trusted --run-receipt-sha256")
    if args.session_start_monotonic > BOOTSTRAP_STARTED:
        raise ValueError("Session clock must precede this script's first bootstrap")
    deadline = work_deadline(args.session_start_monotonic)
    if args.worker_parent_pid is not None:
        return worker(args)
    guard = ExecutionGuard(session_started=args.session_start_monotonic, stage_seconds=41100)
    source_authority(args.root, args.declaration, args.declaration_sha256, guard)
    command = [
        sys.executable,
        str(SCRIPT),
        *(sys.argv[1:] if argv is None else argv),
        "--worker-parent-pid",
        str(os.getpid()),
    ]
    result = supervise(
        command,
        args.output,
        root=args.root,
        session_started=args.session_start_monotonic,
        deadline=deadline,
        storage_check=lambda: storage_measurement(args.root),
    )
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
