"""State copy optimization keeps the original alert state machine and isolation."""

import copy
import importlib.util
import random
from pathlib import Path

import numpy as np
import pytest

from cardiotwin import alerts
from cardiotwin.config import load_config

BASELINE = Path(__file__).parents[1] / "council/work/ecgfounder-resume-20261021/source-before/alerts.py"


def original():
    spec = importlib.util.spec_from_file_location("original_alert_copy", BASELINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config(finite):
    c = load_config()
    if finite:
        c["alert"].update(policy_version="finite-validity-v1", max_duration_seconds=180.0, rearm_seconds=60.0)
    return c


def test_scalar_state_fastpath_skips_recursive_copy_and_isolates_warning(monkeypatch):
    state = {
        "active": True,
        "last_timestamp": 60.0,
        "active_warning": {"timestamp": 30.0, "risk": 0.9, "end_time": None},
    }
    monkeypatch.setattr(alerts, "deepcopy", lambda value: pytest.fail("Scalar state should not recurse"))
    result = alerts._copy_state(state)
    assert result == state and result is not state
    result["active_warning"]["risk"] = 0.1
    assert state["active_warning"]["risk"] == 0.9


@pytest.mark.parametrize("finite", [False, True])
def test_original_transition_and_interval_parity(finite):
    old = original()
    cfg = config(finite)
    rng = random.Random(42)
    before, after, timestamp = {}, {}, 0.0
    stream = [(30.0, 0.99, True)] * 12 + [(30.0, 0.1, True)] * 5 + [(90.0, None, False)]
    stream += [
        (rng.choice([30.0, 30.0, 90.0]), rng.choice([0.01, 0.7, 0.99]), rng.choice([True, True, False]))
        for _ in range(300)
    ]
    for delta, risk, available in stream:
        timestamp += delta
        saved = copy.deepcopy(after)
        expected, warning = old.advance_alert(before, timestamp, risk, available, cfg)
        actual, actual_warning = alerts.advance_alert(after, timestamp, risk, available, cfg)
        assert after == saved
        assert actual == expected and actual_warning == warning
        if finite:
            assert alerts.alert_interval_seconds(
                actual, timestamp, timestamp + delta, cfg
            ) == old.alert_interval_seconds(expected, timestamp, timestamp + delta, cfg)
        if actual_warning:
            isolated = copy.deepcopy(actual)
            actual_warning["risk"] = -10
            assert actual == isolated
        before, after = expected, actual


@pytest.mark.parametrize("kind", ["nested", "cycle", "alias", "subclass", "numpy", "custom"])
def test_complex_state_uses_original_deepcopy(kind, monkeypatch):
    class DictSubclass(dict):
        pass

    class Custom:
        def __init__(self):
            self.items = []

    state = {"extra": [1]}
    if kind == "cycle":
        state["extra"] = state
    elif kind == "alias":
        state = {"first": [], "second": None}
        state["second"] = state["first"]
    elif kind == "subclass":
        state = DictSubclass(state)
    elif kind == "numpy":
        state = {"extra": np.array([1.0])}
    elif kind == "custom":
        state = {"extra": Custom()}
    calls = []

    def recursive(value):
        calls.append(value)
        return copy.deepcopy(value)

    monkeypatch.setattr(alerts, "deepcopy", recursive)
    cloned = alerts._copy_state(state)
    assert calls == [state]
    assert cloned is not state and type(cloned) is type(state)
    if kind == "cycle":
        assert cloned["extra"] is cloned
    elif kind == "alias":
        assert cloned["first"] is cloned["second"] and cloned["first"] is not state["first"]
    else:
        assert cloned["extra"] is not state["extra"]


@pytest.mark.parametrize("pattern", [0, 1])
def test_complete_generated_policy_and_candidate_reports_match_original(pattern, monkeypatch):
    import ast
    import contextlib
    import io
    import time

    from cardiotwin.precursor import freeze_policy_frontiers

    path = Path(__file__).parents[1] / "scripts/benchmark_ecgfounder_cost.py"
    chronology = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "chronology"
    )
    scope = {}
    exec(compile(ast.Module(body=[chronology], type_ignores=[]), "<generated chronology>", "exec"), scope)
    rows, events = scope["chronology"](3000, 2900, pattern)
    cfg = config(True)
    cfg["alert"].update(max_burden_occupancy=0.05, max_eligible_negative_burden_occupancy=0.05)
    current = alerts.advance_alert
    outputs = []
    for advance in (original().advance_alert, current):
        monkeypatch.setattr(alerts, "advance_alert", advance)
        with contextlib.redirect_stdout(io.StringIO()):
            outputs.append(
                freeze_policy_frontiers(rows, events, copy.deepcopy(cfg), deadline=time.monotonic() + 30)
            )
    assert outputs[0] == outputs[1]
