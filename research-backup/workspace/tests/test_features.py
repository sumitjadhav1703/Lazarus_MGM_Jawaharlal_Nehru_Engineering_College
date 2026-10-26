import math

import numpy as np
import pytest

from cardiotwin.config import load_config
from cardiotwin.features import (
    CORE_ECG_FIELDS,
    ENGINEERED_ECG_FIELDS,
    FEATURE_VERSION,
    _lead_disagreement,
    _rr_features,
    _sample_entropy,
    assess_quality,
    extract_features,
    feature_row,
    update_baseline,
)


def ecg(duration=360, fs=200, inverted=False):
    x = np.random.default_rng(19).normal(0, 0.006, (duration * fs, 2))
    for center in np.arange(0.5, duration - 0.1, 0.8) * fs:
        offsets = np.arange(-8, 9)
        pulse = np.exp(-0.5 * (offsets / 1.6) ** 2)
        x[int(center) + offsets, 0] += pulse
        x[int(center) + offsets, 1] += (-0.9 if inverted else 0.9) * pulse
    return x


def test_isolated_large_artifact_does_not_erase_other_beats():
    signal = ecg()
    signal[170 * 200, 1] += 50
    features = extract_features(signal, 200)
    assert features["quality_peak_count"] >= 440
    assert features["quality_lead_disagreement"] < 0.02
    assert assess_quality(features, load_config())["accepted"]


def test_polarity_and_independent_noise_quality():
    inverted = extract_features(ecg(inverted=True), 200)
    assert inverted["quality_lead_disagreement"] < 0.02
    noise = np.random.default_rng(5).normal(0, 0.01, (60000, 2))
    assert not assess_quality(extract_features(noise, 200), load_config())["accepted"]


def test_lead_matching_is_symmetric_one_to_one():
    a, b = np.array([1.0, 1.05]), np.array([1.02, 4.0])
    assert _lead_disagreement(a, b) == 0.5
    assert _lead_disagreement(b, a) == 0.5
    assert _lead_disagreement(np.array([]), a) == 1
    assert _lead_disagreement(np.array([1.0]), np.array([1.08])) == 0


def test_engineered_constant_rr_and_recent_window_contrast():
    peaks = np.arange(0, 361, dtype=float)
    result = _rr_features(peaks, 360)
    assert result["ecg_rr_median"] == 1
    assert result["ecg_rr_iqr"] == result["ecg_rr_mad"] == result["ecg_rr_cv"] == 0
    assert result["ecg_sd1"] == result["ecg_sd2"] == 0
    assert result["ecg_hr_60s"] == result["ecg_hr_300s"] == 60
    assert result["ecg_rr_sample_entropy"] == 0
    assert result["ecg_rr_lag1_corr"] is None
    varying = np.r_[np.arange(0, 301, dtype=float), np.arange(300.5, 360.1, 0.5)]
    values = _rr_features(varying, 360)
    assert values["ecg_hr_60s"] == 120
    assert values["ecg_hr_300s"] < 120
    assert values["ecg_hr_60s_vs_300s_delta"] > 0


def test_invalid_rr_never_joins_successive_pairs_or_entropy_patterns():
    rr = np.r_[np.ones(20), 0.1, np.full(20, 1.2)]
    result = _rr_features(np.r_[0, np.cumsum(rr)], 600)
    assert result["ecg_rmssd"] == pytest.approx(0, abs=1e-12)
    assert result["ecg_sd1"] == pytest.approx(0, abs=1e-12)
    assert result["ecg_rr_lag1_corr"] == pytest.approx(1)
    assert result["ecg_rr_sample_entropy"] is None
    assert result["quality_entropy_rr_count"] == 20
    short_tail = np.r_[np.ones(40), 0.1, np.full(3, 1.1)]
    assert _rr_features(np.r_[0, np.cumsum(short_tail)], 600)["ecg_rr_sample_entropy"] is None


def test_sample_entropy_math_and_missing_matches():
    assert _sample_entropy(np.ones(20)) == 0
    assert _sample_entropy(np.r_[np.arange(100), np.ones(256)]) == 0
    assert _sample_entropy(np.array([1.0, 2.0, 3.0])) is None
    assert _sample_entropy(np.arange(5, dtype=float)) is None
    # Six m=2 matches, three m=3 matches among same five template starts => -log(3/6).
    assert _sample_entropy(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0])) == pytest.approx(math.log(2))


def test_baseline_dispersion_uses_only_prior_per_field_updates():
    config = load_config()
    baseline = {"count": 0, "means": {}, "counts": {}}
    for value in [60.0, 70.0, 80.0, 90.0, 100.0]:
        baseline = update_baseline(baseline, {"ecg_hr": value}, config)
    assert baseline["variances"]["ecg_hr"] == pytest.approx(200)
    signal = ecg()
    row, _ = feature_row(signal, 200, {}, baseline, config)
    assert row["baseline_ecg_hr_zscore"] == pytest.approx((row["ecg_hr"] - 80) / math.sqrt(200))
    assert baseline["means"]["ecg_hr"] == 80  # feature extraction never changes its prior state
    cold = {"count": 100, "means": {"ecg_hr": 80}, "counts": {"ecg_hr": 4}, "variances": {"ecg_hr": 200}}
    assert feature_row(signal, 200, {}, cold, config)[0]["baseline_ecg_hr_zscore"] is None


def test_future_prefix_invariance_and_versioned_exports():
    config = load_config()
    signal = ecg()
    changed = signal.copy()
    changed[300 * 200 :] = 999
    assert extract_features(signal[: 300 * 200], 200) == extract_features(changed[: 300 * 200], 200)
    row = extract_features(signal, 200)
    assert all(name in row for name in CORE_ECG_FIELDS + ENGINEERED_ECG_FIELDS)
    assert FEATURE_VERSION == config["feature_version"]
    assert all(value is None or np.isfinite(value) for value in row.values())


def test_missing_samples_in_plausible_rr_are_excluded():
    from cardiotwin.features import _peaks

    fs = 200
    signal = ecg()
    signal[int(100.85 * fs) : int(100.87 * fs), 0] = np.nan  # Between two ordinary .8s beats.
    peaks = _peaks(signal[:, 0], fs)
    rr = np.diff(peaks)
    assert np.all((rr >= 0.25) & (rr <= 2.5))
    row = extract_features(signal, fs)
    assert row["quality_missing"] < 0.001
    assert row["quality_peak_reliability"] < 1
    assert row["quality_peak_reliability"] > 0.99
    assert row["quality_entropy_rr_count"] == 256


def test_entropy_short_production_tail_and_small_baseline_variance_are_unavailable():
    rr = np.r_[np.ones(80), 0.1, np.ones(49)]
    row = _rr_features(np.r_[0, np.cumsum(rr)], 600)
    assert row["quality_entropy_rr_count"] == 49
    assert row["ecg_rr_sample_entropy"] is None
    baseline = {
        "count": 100,
        "counts": {"ecg_hr": 100},
        "means": {"ecg_hr": 80},
        "variances": {"ecg_hr": 1e-28},
    }
    assert feature_row(ecg(), 200, {}, baseline, load_config())[0]["baseline_ecg_hr_zscore"] is None
