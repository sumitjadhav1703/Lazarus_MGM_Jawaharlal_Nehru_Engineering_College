"""Flushed, rate-limited JSON progress; callers supply aggregate metadata only."""

import json
import threading
import time
from datetime import datetime, timezone

_OUTPUT_LOCK = threading.Lock()


class Progress:
    def __init__(self, stage, total=None, interval_seconds=30, **context):
        self.stage, self.total = stage, total
        self.interval_seconds, self.context = interval_seconds, context
        self.started = self.last_emitted = time.monotonic()
        self.completed, self.finished = 0, False
        self._emit("started", self.started)

    def _emit(self, status, now):
        payload = json.dumps(
            {
                **self.context,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "stage": self.stage,
                "status": status,
                "completed": self.completed,
                "total": self.total,
                "elapsed_seconds": max(0, now - self.started),
            }
        )
        with _OUTPUT_LOCK:
            try:
                print(payload, flush=True)
            except (OSError, ValueError):
                # A closed diagnostic sink must not abort work or replace its exception.
                pass
        self.last_emitted = now

    def update(self, completed=None, force=False, **details):
        if self.finished:
            return
        if completed is not None:
            self.completed = completed
        self.context.update(details)
        now = time.monotonic()
        if force or now - self.last_emitted >= self.interval_seconds:
            self._emit("running", now)

    def finish(self, status="done", **details):
        if not self.finished:
            self.context.update(details)
            self._emit(status, time.monotonic())
            self.finished = True

    def __enter__(self):
        return self

    def __exit__(self, error_type, error, traceback):
        if error_type is None:
            self.finish()
        else:
            self.finish("failed", error_type=error_type.__name__)
