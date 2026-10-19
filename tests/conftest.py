"""Configure native runtimes before optional neural tests are collected."""

import os
import sys
from datetime import datetime, timezone

import pytest

if sys.platform == "darwin":
    # Mixed Torch/XGBoost OpenMP runtimes can crash; verified both import orders.
    # https://github.com/pytorch/pytorch/issues/155201
    os.environ.setdefault("OMP_NUM_THREADS", "1")


class HistoricalECGFounderDatetime(datetime):
    """Synthetic UTC clock before the frozen experiment's calendar cutoff."""

    @classmethod
    def now(cls, tz=None):
        fixed = cls(2026, 10, 15, 12, tzinfo=timezone.utc)
        return fixed.astimezone(tz) if tz is not None else fixed.replace(tzinfo=None)


@pytest.fixture
def historical_ecgfounder_clock(monkeypatch):
    from cardiotwin import ecgfounder_runner, ecgfounder_study

    # Patch module bindings only; pytest restores them after each synthetic test.
    monkeypatch.setattr(ecgfounder_runner, "datetime", HistoricalECGFounderDatetime)
    monkeypatch.setattr(ecgfounder_study, "datetime", HistoricalECGFounderDatetime)
