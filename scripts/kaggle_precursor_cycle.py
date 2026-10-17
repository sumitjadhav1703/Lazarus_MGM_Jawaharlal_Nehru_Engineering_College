"""Local/private execution guard for a predeclared precursor command; never submits jobs."""

import time

LAUNCHED = time.monotonic()

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import signal  # noqa: E402
from pathlib import Path  # noqa: E402

from cardiotwin.investigation import _write  # noqa: E402
from cardiotwin.precursor import fresh_output  # noqa: E402

_spec = importlib.util.spec_from_file_location("precursor_supervisor", Path(__file__).with_name("kaggle_cycle.py"))
_cycle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cycle)
dataset_bytes, run_child = _cycle.dataset_bytes, _cycle.run_child


def run_guarded(command, output, storage_roots, *, session_started, session_seconds=41400,
                reserve_seconds=300, output_reserve_bytes=250_000_000, experiment_roots=()):
    output = fresh_output(output)
    before, after = {}, {}
    state = dict(stage="time_preflight", command=command, storage_before=before, storage_after=after,
                 session_seconds=session_seconds, receipt_reserve_seconds=reserve_seconds,
                 output_reserve_bytes=output_reserve_bytes, retry=False)
    try:
        remaining = session_seconds - (time.monotonic() - session_started) - reserve_seconds
        state["remaining_seconds"] = remaining
        if remaining <= 0:
            raise TimeoutError("No fitting allowance remains after bootstrap and receipt reserve")
        state["stage"] = "storage_measurement"
        for name, allocated in (("apparent_bytes", False), ("allocated_bytes", True)):
            before[name] = dataset_bytes(storage_roots, allocated=allocated)
        state["stage"] = "storage_reservation"
        if max(before.values()) + output_reserve_bytes > _cycle.DATASET_BYTES:
            raise RuntimeError("Output reservation does not fit the project storage cap")
        _write(output / "preflight.json", state)
        state["stage"] = "child_execution"
        receipt = run_child(command, output / "child.log", remaining, storage_roots=storage_roots,
                            check_allocated=True, output_limit=([output, *experiment_roots], output_reserve_bytes))
        state["stage"] = "storage_after_measurement"
        for name, allocated in (("apparent_bytes", False), ("allocated_bytes", True)):
            after[name] = dataset_bytes(storage_roots, allocated=allocated)
    except BaseException as error:
        _write(output / "receipt.json", dict(state, status="failed", original_error=str(error),
                                             error_type=type(error).__name__, session_elapsed_seconds=time.monotonic()-session_started))
        raise
    receipt.update(storage_before=before, storage_after=after,
                   session_elapsed_seconds=time.monotonic() - session_started,
                   retry=False, output_reserve_bytes=output_reserve_bytes)
    _write(output / "receipt.json", receipt)
    return receipt


def main():
    started = LAUNCHED
    signal.signal(signal.SIGTERM, _cycle._termination_handler)
    signal.signal(signal.SIGINT, _cycle._termination_handler)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--storage-root", action="append", required=True)
    parser.add_argument("--session-seconds", type=float, default=41400)
    parser.add_argument("--experiment-root", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not 300 < args.session_seconds <= 41400:
        parser.error("a command and whole-session allowance in (300,41400] are required")
    result = run_guarded(command, args.output, args.storage_root, session_started=started,
                         session_seconds=args.session_seconds, experiment_roots=args.experiment_root)
    return 0 if result["status"] == "finished" else 1


if __name__ == "__main__":
    raise SystemExit(main())
