#!/usr/bin/env python3
"""Supervise the development-only matched RR16 stopping-objective cycle."""

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from cardiotwin.gpu_telemetry import GpuTelemetry
from cardiotwin.progress import Progress

SESSION_SECONDS = 41_400
ARM_SECONDS = 20_160
RECEIPT_RESERVE_SECONDS = 300
DATASET_BYTES = 25_000_000_000
RR_CACHE_BYTES = 4_000_000_000
OUTPUT_RESERVE_BYTES = 100_000_000
RUNNER = Path(__file__).with_name("run_investigation.py")
LAUNCHED = time.monotonic()


def _termination_handler(signum, frame):
    raise RuntimeError(f"Supervisor received {signal.Signals(signum).name}")


def dataset_bytes(roots, *, allocated=False):
    roots = {Path(root).resolve() for root in roots}
    roots = {
        root for root in roots if not any(root != other and root.is_relative_to(other) for other in roots)
    }
    seen, total = set(), 0
    for root in roots:
        for path in root.rglob("*"):
            try:
                if not path.is_file():
                    continue
                info = path.stat()
            except FileNotFoundError:
                continue
            identity = (info.st_dev, info.st_ino)
            if identity not in seen:
                seen.add(identity)
                total += info.st_blocks * 512 if allocated else info.st_size
    return total


def _stop_group(process):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        if sig == signal.SIGTERM:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
    process.wait(timeout=2)


def run_child(command, log_path, timeout_seconds, storage_roots=(), *, check_allocated=False, output_limit=None):
    """Drain output in a daemon thread so output cannot block the timeout guard."""
    started = time.monotonic()
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True
    )
    output_errors = []

    def copy_output():
        try:
            with Path(log_path).open("wb") as log:
                while chunk := os.read(process.stdout.fileno(), 65_536):
                    log.write(chunk)
                    log.flush()
                    try:
                        sys.stdout.write(chunk.decode("utf-8", errors="replace"))
                        sys.stdout.flush()
                    except (OSError, ValueError):
                        pass
        except OSError as error:
            output_errors.append(str(error))

    reader = threading.Thread(target=copy_output, name="arm-output", daemon=True)
    reader.start()
    status, error = None, None
    def occupied(roots):
        apparent = dataset_bytes(roots)
        return max(apparent, dataset_bytes(roots, allocated=True)) if check_allocated else apparent

    def storage_error():
        if storage_roots and occupied(storage_roots) > DATASET_BYTES:
            return "Dataset storage cap exceeded"
        if output_limit and occupied(output_limit[0] if isinstance(output_limit[0], (list, tuple)) else [output_limit[0]]) > output_limit[1]:
            return "Experiment output reservation exceeded"
        return None

    next_storage_check = started
    with Progress("kaggle_cycle.child", total=1, pid=process.pid, cap_seconds=timeout_seconds) as progress:
        try:
            while process.poll() is None:
                remaining = timeout_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    status, error = "timed_out", "Supervisor arm wall-time limit exhausted"
                    break
                if (storage_roots or output_limit) and time.monotonic() >= next_storage_check:
                    measured = occupied(storage_roots)
                    next_storage_check = time.monotonic() + 30
                    problem = storage_error()
                    if problem:
                        status, error = "failed", problem
                        break
                    progress.update(dataset_bytes=measured)
                try:
                    process.wait(timeout=min(1, remaining))
                except subprocess.TimeoutExpired:
                    progress.update()
            if status is None:
                status = "finished" if process.returncode == 0 else "failed"
                error = None if process.returncode == 0 else f"Child exited with code {process.returncode}"
            if status == "finished" and (problem := storage_error()):
                status, error = "failed", problem + " at child completion"
        finally:
            _stop_group(process)
            reader.join(timeout=2)
            progress.finish("done" if status == "finished" else "failed", exit_code=process.returncode)
    if output_errors or reader.is_alive():
        status, error = "failed", "; ".join(output_errors) or "Child output reader did not finish"
    tail = None
    if status != "finished" and Path(log_path).exists():
        with Path(log_path).open("rb") as log:
            log.seek(max(0, log.seek(0, os.SEEK_END) - 8192))
            tail = log.read().decode("utf-8", errors="replace")
    return {
        "status": status,
        "exit_code": process.returncode,
        "original_error": error,
        "child_output_tail": tail,
        "output_errors": output_errors,
        "output_reader_finished": not reader.is_alive(),
        "wall_seconds": time.monotonic() - started,
        "log": str(log_path),
    }


def classify_arm(summary_path, mode):
    try:
        summary = json.loads(Path(summary_path).read_text())
    except (OSError, ValueError) as error:
        return "partial", str(error)
    matched = (
        summary.get("seed") == 42
        and summary.get("specs") == [["rr", 16]]
        and summary.get("screening_epochs") == 8
        and summary.get("screening_max_training_windows") == 10_000
        and summary.get("early_stop_weighting") == mode
    )
    if not matched:
        return "failed", "Child summary does not match the predeclared recipe"
    complete = (
        summary.get("status") == "complete"
        and not summary.get("failures")
        and summary.get("combined", {}).get("rr_tcn_16", {}).get("complete_outer_folds") == 3
    )
    return ("complete", None) if complete else ("partial", "Child did not complete all three RR16 folds")


def run_cycle(data, cache, split, output, session_seconds_remaining, arm_seconds=ARM_SECONDS):
    started = time.monotonic()
    if (
        not math.isfinite(session_seconds_remaining)
        or not RECEIPT_RESERVE_SECONDS < session_seconds_remaining <= SESSION_SECONDS
    ):
        raise ValueError("Session seconds remaining must be >300 and <=41400")
    if not math.isfinite(arm_seconds) or not 0 < arm_seconds <= ARM_SECONDS:
        raise ValueError("Arm seconds must be positive and <=20160")
    if json.loads(Path(split).read_text()).get("seed") != 42:
        raise ValueError("The matched cycle requires the frozen seed42 development split")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Fresh cycle output required: {output}")
    roots = (data, cache, output)
    if Path("/kaggle/input").exists():
        roots += (Path("/kaggle/input"),)
    if not Path(data).is_dir() or not Path(cache).is_dir():
        raise ValueError("Source and development cache directories must exist")
    occupied = dataset_bytes(roots)
    if occupied + RR_CACHE_BYTES + OUTPUT_RESERVE_BYTES > DATASET_BYTES:
        raise ValueError("Insufficient 25 GB headroom for the 4 GB RR cache and 100 MB output reserve")
    output.mkdir(parents=True, exist_ok=False)
    deadline = started + session_seconds_remaining - RECEIPT_RESERVE_SECONDS
    receipt = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "session_seconds_remaining_at_start": session_seconds_remaining,
        "receipt_reserve_seconds": RECEIPT_RESERVE_SECONDS,
        "per_arm_cap_seconds": arm_seconds,
        "initial_dataset_bytes": occupied,
        "rr_cache_reserve_bytes": RR_CACHE_BYTES,
        "output_reserve_bytes": OUTPUT_RESERVE_BYTES,
        "recipe": {"model": "rr16", "seed": 42, "epochs": 8, "training_windows": 10_000},
        "telemetry_scope": "NVIDIA metrics cover the device; process_cpu_core_percent covers the supervisor only, not the training child",
        "scope": "Original development groups only; examined primary/LTAF cohorts are historical references",
        "automatic_retry": False,
        "arms": [],
    }

    def save():
        receipt["wall_seconds"] = time.monotonic() - started
        receipt["updated_utc"] = datetime.now(timezone.utc).isoformat()
        temporary = output / "receipt.tmp"
        temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        temporary.replace(output / "receipt.json")

    with Progress("kaggle_cycle", total=2, **receipt["recipe"]) as progress:
        try:
            for mode in ("patient_only", "patient_class"):
                cap = min(arm_seconds, deadline - time.monotonic())
                arm_output = output / mode
                arm = {"mode": mode, "cap_seconds": max(0, cap), "output": str(arm_output)}
                receipt["arms"].append(arm)
                if cap <= 0:
                    arm.update(status="skipped", original_error="Session wall-time budget exhausted")
                    break
                command = [
                    sys.executable,
                    "-u",
                    str(RUNNER),
                    "neural",
                    "--data",
                    str(data),
                    "--cache",
                    str(cache),
                    "--split",
                    str(split),
                    "--output",
                    str(arm_output),
                    "--sequence-cache",
                    str(output / "rr-sequences.sqlite"),
                    "--models",
                    "rr16",
                    "--epochs",
                    "8",
                    "--training-windows",
                    "10000",
                    "--early-stop-weighting",
                    mode,
                    "--max-hours",
                    str(cap / 3600),
                ]
                arm.update(
                    status="running", started_utc=datetime.now(timezone.utc).isoformat(), command=command
                )
                save()
                arm.update(run_child(command, output / f"{mode}.log", cap, storage_roots=roots))
                if arm["status"] == "finished":
                    arm["status"], arm["original_error"] = classify_arm(arm_output / "summary.json", mode)
                save()
                progress.update(len(receipt["arms"]), force=True, arm_mode=mode, arm_status=arm["status"])
                if arm["status"] != "complete":
                    break
            receipt["status"] = (
                "complete"
                if len(receipt["arms"]) == 2 and all(a["status"] == "complete" for a in receipt["arms"])
                else "incomplete"
            )
        except BaseException as error:
            receipt.update(status="failed", original_error=str(error), error_type=type(error).__name__)
            if receipt["arms"] and receipt["arms"][-1]["status"] == "running":
                receipt["arms"][-1].update(status="failed", original_error=str(error))
            raise
        finally:
            save()
        progress.finish("done" if receipt["status"] == "complete" else "failed")
    return receipt


def main():
    signal.signal(signal.SIGTERM, _termination_handler)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "cache", "split", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--session-seconds-remaining", required=True, type=float)
    parser.add_argument("--arm-seconds", type=float, default=ARM_SECONDS)
    args = parser.parse_args()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU unavailable; refusing a CPU fallback for this cycle")
    with GpuTelemetry():
        args.session_seconds_remaining -= time.monotonic() - LAUNCHED
        receipt = run_cycle(**vars(args))
    if receipt["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
