"""Pure persistence/hysteresis/cooldown policy shared by replay and evaluation."""

import math
from copy import deepcopy

_SCALAR_TYPES = (str, int, float, bool, type(None))


def _copy_state(state):
    # Ordinary runtime states contain scalars and one scalar-only warning dictionary.
    if type(state) is dict and all(
        type(key) is str and (
            type(value) in _SCALAR_TYPES
            or key == "active_warning" and type(value) is dict and all(
                type(k) is str and type(v) in _SCALAR_TYPES for k, v in value.items()
            )
        )
        for key, value in state.items()
    ):
        result = state.copy()
        if type(result.get("active_warning")) is dict:
            result["active_warning"] = result["active_warning"].copy()
        return result
    return deepcopy(state)


def advance_alert(state, timestamp, risk, available, config):
    version = config["alert"].get("policy_version")
    if state and state.get("policy_version") != version:
        raise ValueError("Alert state/config policy version mismatch")
    if version == "finite-validity-v1":
        return _advance_finite(state, timestamp, risk, available, config)
    if version is not None:
        raise ValueError("Unknown alert policy version")
    s = _copy_state(state)
    a = config["alert"]
    previous = s.get("last_timestamp")
    if not math.isfinite(timestamp) or (previous is not None and timestamp <= previous):
        raise ValueError("Alert timestamps must increase and be finite")
    if available and (risk is None or not math.isfinite(risk) or not 0 <= risk <= 1):
        raise ValueError("Available risk must be finite in [0,1]")
    gap = previous is not None and timestamp - previous > config["target"]["cadence_seconds"] + 1e-6
    s.update(last_timestamp=timestamp)
    s.setdefault("active", False)
    s.setdefault("suppressed_count", 0)
    warning = None
    if not available or gap or risk < a["release_threshold"]:
        s["high_since"] = None
        if s["active"]:
            warning = deepcopy(s["active_warning"])
            warning["end_time"] = timestamp
            s["active"] = False
            s["active_warning"] = None
        if not available or gap:
            if available and risk >= a["threshold"]:
                s["high_since"] = timestamp
            return s, warning
    if risk >= a["threshold"]:
        if s.get("high_since") is None:
            s["high_since"] = timestamp
        persisted = timestamp - s["high_since"] >= a["persistence_seconds"]
        if persisted and not s["active"]:
            if timestamp - s.get("last_issue", -float("inf")) >= a["cooldown_seconds"]:
                warning = {
                    "timestamp": timestamp,
                    "end_time": None,
                    "risk": float(risk),
                    "persistence_seconds": timestamp - s["high_since"],
                }
                s.update(active=True, last_issue=timestamp, active_warning=deepcopy(warning))
            else:
                s["suppressed_count"] += 1
    elif not s["active"]:
        s["high_since"] = None
    return s, warning


def _advance_finite(state, timestamp, risk, available, config):
    s, a = _copy_state(state), config["alert"]
    previous = s.get("last_timestamp")
    if not math.isfinite(timestamp) or (previous is not None and timestamp <= previous):
        raise ValueError("Alert timestamps must increase and be finite")
    if available and (risk is None or not math.isfinite(risk) or not 0 <= risk <= 1):
        raise ValueError("Available risk must be finite in [0,1]")
    cadence = config["target"]["cadence_seconds"]
    gap = previous is not None and timestamp > previous + cadence + 1e-6
    s.setdefault("active", False)
    s.setdefault("awaiting_rearm", False)
    s.setdefault("low_since", None)
    s.setdefault("suppressed_count", 0)
    warning = None
    if s["active"]:
        expiry = s["active_warning"]["timestamp"] + a["max_duration_seconds"]
        boundary = min(expiry, previous + cadence) if gap else expiry
        close = gap or not available or risk < a["release_threshold"] or timestamp >= expiry
        if close:
            warning = {**s["active_warning"], "end_time": min(timestamp, boundary)}
            s.update(active=False, active_warning=None, awaiting_rearm=True, low_since=None, high_since=None)
    s.update(policy_version="finite-validity-v1", last_timestamp=timestamp,
             last_available=bool(available), last_risk=float(risk) if available else None)
    if gap or not available:
        s.update(low_since=None, high_since=None)
    if s["awaiting_rearm"]:
        s["high_since"] = None
        if available and risk < a["release_threshold"]:
            if s["low_since"] is None:
                s["low_since"] = timestamp
            if timestamp - s["low_since"] >= a["rearm_seconds"]:
                s.update(awaiting_rearm=False, low_since=None)
        else:
            s["low_since"] = None
        return s, warning
    if warning or not available:
        return s, warning
    if risk >= a["threshold"]:
        if s.get("high_since") is None:
            s["high_since"] = timestamp
        if timestamp - s["high_since"] >= a["persistence_seconds"] and not s["active"]:
            last_issue = s.get("last_issue")
            if last_issue is None or timestamp - last_issue >= a["cooldown_seconds"]:
                warning = dict(timestamp=timestamp, end_time=None, risk=float(risk),
                               persistence_seconds=timestamp - s["high_since"])
                s.update(active=True, active_warning=deepcopy(warning), last_issue=timestamp)
            else:
                s["suppressed_count"] += 1
    elif not s["active"]:
        s["high_since"] = None
    return s, warning


def alert_interval_seconds(state, start, end, config):
    """Integrate one forecast's supported interval; an absent update is unavailable."""
    if config["alert"].get("policy_version") != "finite-validity-v1" or state.get("policy_version") != "finite-validity-v1":
        raise ValueError("Interval accounting requires the finite policy version")
    if not math.isfinite(start) or not math.isfinite(end) or end < start or state["last_timestamp"] != start:
        raise ValueError("Invalid interval or state timestamp")
    cadence, a = config["target"]["cadence_seconds"], config["alert"]
    supported = min(end - start, cadence)
    available = bool(state["last_available"])
    active = bool(state["active"])
    waiting = bool(state["awaiting_rearm"])
    warning = min(supported, max(0., state["active_warning"]["timestamp"] + a["max_duration_seconds"] - start)) if active else 0.
    remaining = supported - warning
    elevated = available and state["last_risk"] >= a["release_threshold"]
    unresolved = remaining if elevated and (waiting or active) else 0.
    low_wait = remaining if available and not elevated and waiting else 0.
    unavailable = supported if not available and waiting else 0.
    if waiting or active:
        unavailable += end - start - supported
    result = dict(warning_seconds=warning, unresolved_elevated_seconds=unresolved,
                  rearm_wait_seconds=low_wait, unavailable_rearm_seconds=unavailable,
                  burden_seconds=warning + unresolved, available_seconds=supported if available else 0.)
    assert all(v >= 0 for v in result.values()) and result["burden_seconds"] <= end - start
    return result
