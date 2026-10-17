"""Preflight or execute one separately authorized outcome-blind extraction; never fit/submit."""

import argparse
import os
import signal
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from cardiotwin.ecgfounder_runner import (
    CHECKPOINT_PATH,
    ExecutionGuard,
    extract,
    preflight,
    storage_measurement,
)
from cardiotwin.investigation import _write
from cardiotwin.precursor import fresh_output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "declaration", "declaration-sha256", "data-root", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--mode", choices=("preflight", "extract"), required=True)
    parser.add_argument(
        "--checkpoint", default=CHECKPOINT_PATH, help="Must resolve to the fixed bound checkpoint"
    )
    parser.add_argument(
        "--session-start-monotonic",
        type=float,
        help="Actual bootstrap process monotonic start; required for extraction",
    )
    args = parser.parse_args(argv)
    if args.mode == "extract" and args.session_start_monotonic is None:
        parser.error("extract requires actual bootstrap --session-start-monotonic")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Fresh output required; no retry/resume")
    session_started = (
        args.session_start_monotonic if args.session_start_monotonic is not None else time.monotonic()
    )
    guard = ExecutionGuard(session_started=session_started, stage_seconds=41100.0)

    def storage():
        return storage_measurement(args.root)

    prior = None
    cleanup_deadline = None

    def cleanup_expired(signum, frame):
        os._exit(124)  # Nested best-effort cleanup catches BaseException; the hard ceiling cannot raise.

    def preserve_failure(error):
        if not output.exists():
            fresh_output(output)
        if not (output / "receipt.json").exists():
            _write(
                output / "receipt.json",
                dict(
                    status="failed",
                    original_error=str(error),
                    error_type=type(error).__name__,
                    fits_allowed=False,
                    full_run_eligible=False,
                    finished_utc=datetime.now(timezone.utc).isoformat(),
                ),
            )

    def arm_cleanup(deadline=None):
        nonlocal prior, cleanup_deadline
        current = time.monotonic()
        cleanup_deadline = min(
            current + 300,
            session_started + 41400,
            deadline if deadline is not None else float("inf"),
            cleanup_deadline if cleanup_deadline is not None else float("inf"),
        )
        handler = signal.signal(signal.SIGALRM, cleanup_expired)
        if prior is None:
            prior = handler
        signal.setitimer(signal.ITIMER_REAL, max(0.001, cleanup_deadline - current))

    def expired(signum, frame):
        arm_cleanup()
        error = TimeoutError("Whole session41100-second work deadline reached;300-second cleanup reserved")
        try:
            preserve_failure(error)
        except BaseException:
            pass
        raise error

    try:
        guard()
        if args.mode == "extract":
            prior = signal.signal(signal.SIGALRM, expired)
            signal.setitimer(signal.ITIMER_REAL, max(0.001, 41100 - (time.monotonic() - session_started)))
        prepared = preflight(
            args.declaration,
            args.declaration_sha256,
            root=args.root,
            data_root=args.data_root,
            storage_check=storage,
            guard=guard,
            checkpoint_path=args.checkpoint,
        )
        if args.mode == "preflight":
            fresh_output(output)
            _write(output / "receipt.json", prepared["receipt"])
            return 0
        result = extract(
            prepared,
            output,
            session_started=session_started,
            storage_check=storage,
            cleanup_watchdog=arm_cleanup,
        )
        return 0 if result["status"] == "complete" else 1
    except BaseException as error:
        try:
            if args.mode == "extract":
                arm_cleanup()
            preserve_failure(error)
            (output / "original-error.txt").write_text(traceback.format_exc())
        except BaseException:
            pass
        raise
    finally:
        if prior is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, prior)


if __name__ == "__main__":
    raise SystemExit(main())
