"""Versioned experimental forecasting contract and validation."""

import json
from importlib.resources import files
from pathlib import Path


def load_config(path=None):
    source = Path(path) if path else Path(__file__).parents[2] / "configs/default.json"
    if not source.exists() and path is None:
        source = files("cardiotwin").joinpath("default.json")
    config = json.loads(source.read_text())
    validate_config(config)
    return config


def validate_config(config):
    from .features import FEATURE_VERSION

    if config.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Feature extraction version differs; re-audit and retrain with the current config")
    training = config.get("training", {})
    if training.get("sample_weighting") not in {"patient_class", "patient_only", "window_class"}:
        raise ValueError("Unknown training sample weighting")
    if not isinstance(training.get("calibration_c"), (int, float)) or not 0 < training[
        "calibration_c"
    ] < float("inf"):
        raise ValueError("calibration_c must be finite and positive")
    t, q, b, a = (config[k] for k in ("target", "quality", "baseline", "alert"))
    finite_fields = {"policy_version", "max_duration_seconds", "rearm_seconds", "max_burden_occupancy",
                     "max_eligible_negative_burden_occupancy"}
    if finite_fields & a.keys():
        if not finite_fields <= a.keys() or a["policy_version"] != "finite-validity-v1":
            raise ValueError("Finite alert policy requires a recognized complete version block")
        for key in ("max_duration_seconds", "rearm_seconds"):
            if not isinstance(a[key], (float, int)) or not 0 < a[key] < float("inf"):
                raise ValueError(f"{key} must be finite and positive")
        if a["max_duration_seconds"] < t["cadence_seconds"]:
            raise ValueError("max_duration_seconds must be at least cadence")
        for key in ("max_burden_occupancy", "max_eligible_negative_burden_occupancy"):
            if not isinstance(a[key], (float, int)) or not 0 <= a[key] <= 1:
                raise ValueError(f"{key} must be finite and in [0,1]")
    for key in (
        "history_seconds",
        "gap_seconds",
        "horizon_seconds",
        "cadence_seconds",
        "min_episode_seconds",
    ):
        if not isinstance(t[key], (int, float)) or not 0 < t[key] < float("inf"):
            raise ValueError(f"{key} must be finite and positive")
    if t["gap_seconds"] >= t["horizon_seconds"] or t["washout_seconds"] < 0:
        raise ValueError("gap must be below horizon and washout nonnegative")
    if not 0 <= a["release_threshold"] <= a["threshold"] <= 1:
        raise ValueError("Require 0 <= release threshold <= warning threshold <= 1")
    for key in ("persistence_seconds", "cooldown_seconds"):
        if not 0 <= a[key] < float("inf"):
            raise ValueError(f"{key} must be finite and nonnegative")
    if not 1 <= b["minimum_updates"] <= b["max_updates"]:
        raise ValueError("Invalid baseline update limits")
    for key in ("max_missing_fraction", "max_flatline_fraction", "max_lead_disagreement"):
        if not 0 <= q[key] <= 1:
            raise ValueError(f"{key} must be in [0,1]")
    if q["min_peaks"] < 2:
        raise ValueError("min_peaks must be at least two")
    if "min_qrs_kurtosis" in q and (
        not isinstance(q["min_qrs_kurtosis"], (int, float)) or not 0 < q["min_qrs_kurtosis"] < float("inf")
    ):
        raise ValueError("min_qrs_kurtosis must be finite and positive")
    if "max_saturation_fraction" in q and (
        not isinstance(q["max_saturation_fraction"], (int, float))
        or not 0 <= q["max_saturation_fraction"] <= 1
    ):
        raise ValueError("max_saturation_fraction must be in [0,1]")
    budget = a.get("max_false_warnings_per_hour")
    sensitivity = a.get("min_event_sensitivity")
    if budget is not None and not 0 <= budget < float("inf"):
        raise ValueError("False warning budget must be finite and nonnegative")
    if sensitivity is not None and not 0 <= sensitivity <= 1:
        raise ValueError("Sensitivity target must be in [0,1]")
    for key in ("min_event_subjects", "min_test_event_subjects"):
        if config["audit"][key] < 1:
            raise ValueError(f"{key} must be positive")
