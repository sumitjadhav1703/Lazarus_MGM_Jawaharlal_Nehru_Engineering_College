"""Annotation-free trailing ECG features and prior-only patient baselines.

RR variability is not certified normal-to-normal HRV: beat selection is
signal-derived, and arrhythmia or ectopy can invalidate physiological HRV.
"""

import numpy as np
from scipy.signal import butter, find_peaks, sosfilt
from scipy.stats import kurtosis

FEATURE_VERSION = "causal-rr-v2"
MIN_ENTROPY_RR = 50  # Provisional engineering minimum; short sequences have unstable estimates.
BASELINE_VARIANCE_FLOOR = 1e-12  # Squared feature units; avoids floating-point pseudo-confidence.
CORE_ECG_FIELDS = ("ecg_hr", "ecg_rr_mean", "ecg_sdnn", "ecg_rmssd", "ecg_pnn50", "ecg_hr_trend")
ENGINEERED_ECG_FIELDS = (
    "ecg_rr_median",
    "ecg_rr_iqr",
    "ecg_rr_mad",
    "ecg_rr_cv",
    "ecg_sd1",
    "ecg_sd2",
    "ecg_rr_lag1_corr",
    "ecg_rr_sample_entropy",
    "ecg_hr_60s",
    "ecg_hr_300s",
    "ecg_rr_cv_60s",
    "ecg_rr_cv_300s",
    "ecg_rmssd_60s",
    "ecg_rmssd_300s",
    "ecg_hr_60s_vs_300s_delta",
    "ecg_rr_cv_60s_vs_300s_delta",
    "ecg_rmssd_60s_vs_300s_delta",
)

CLINICAL_FIELDS = (
    "age",
    "sex_male",
    "bmi",
    "hypertension",
    "diabetes",
    "heart_failure",
    "prior_stroke",
    "vascular_disease",
    "prior_af_ablation",
    "prior_afl",
    "pacemaker",
    "beta_blocker",
    "antiarrhythmic",
    "anticoagulation",
    "echo_lad",
    "echo_lvef",
)
BASELINE_FIELDS = ("ecg_hr", "ecg_rr_mean", "ecg_sdnn", "ecg_rmssd", "ecg_hr_trend") + ENGINEERED_ECG_FIELDS


def _finite(value):
    return float(value) if value is not None and np.isfinite(value) else None


def _qrs_band(lead, fs):
    # Forward fill missing samples; interpolation would use later samples.
    present = np.isfinite(lead)
    indices = np.maximum.accumulate(np.where(present, np.arange(len(lead)), 0))
    clean = np.where(present[indices], lead[indices], 0.0)
    sos = butter(2, [5, min(20, fs * 0.4)], btype="bandpass", fs=fs, output="sos")
    return sosfilt(sos, clean)


def _peaks(lead, fs, filtered=None):
    filtered = _qrs_band(lead, fs) if filtered is None else filtered
    energy = filtered**2
    median = np.median(energy)
    amplitude = np.percentile(energy, 99)  # Isolated motion spikes must not erase the entire history.
    scale = np.median(np.abs(energy - median))
    peaks, _ = find_peaks(
        energy,
        height=max(median + 8 * scale, amplitude * 0.10, 1e-10),
        distance=max(1, int(0.25 * fs)),
        prominence=max(scale * 4, amplitude * 0.04),
    )
    # Ignore filter startup and unresolved trailing edge.
    return peaks[(peaks >= fs) & (peaks < len(lead) - int(0.1 * fs))] / fs


def _lead_disagreement(a, b):
    """Symmetric unmatched fraction, with chronological one-to-one 80 ms matching."""
    if not len(a) or not len(b):
        return 1.0
    i = j = matches = 0
    while i < len(a) and j < len(b):
        if abs(a[i] - b[j]) <= 0.08 + 1e-12:
            matches += 1
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return max(1 - matches / len(a), 1 - matches / len(b))


def _sample_entropy(values):
    """SampEn(m=2,r=.2*std), excluding self matches; non-estimable values stay null."""
    values = np.asarray(values, dtype=float)[-256:]
    if len(values) < 4 or not np.isfinite(values).all():
        return None
    if np.ptp(values) <= 1e-12:
        return 0.0
    radius = 0.2 * np.std(values)
    templates = np.lib.stride_tricks.sliding_window_view(values, 3)
    # ponytail: exact O(n²) comparisons capped at 256 RR; chunk pairs if a longer window is required.
    differences = np.abs(templates[:, None, :] - templates[None, :, :])
    pairs = np.triu(np.ones((len(templates), len(templates)), dtype=bool), k=1)
    short_matches = np.count_nonzero(pairs & (differences[:, :, :2].max(axis=2) <= radius))
    long_matches = np.count_nonzero(pairs & (differences.max(axis=2) <= radius))
    if not short_matches or not long_matches:
        return None
    return _finite(-np.log(long_matches / short_matches))


def _rr_features(peaks, duration, valid_mask=None):
    """Descriptive RR statistics only: no claim of validated normal-to-normal HRV."""
    result = dict.fromkeys(CORE_ECG_FIELDS + ENGINEERED_ECG_FIELDS)
    rr = np.diff(peaks)
    valid = (rr >= 0.25 - 1e-12) & (rr <= 2.5 + 1e-12)
    if valid_mask is not None:
        signal_valid = np.asarray(valid_mask, dtype=bool)
        if signal_valid.shape != valid.shape:
            raise ValueError("RR signal-valid mask must match interval count")
        valid &= signal_valid
    good = rr[valid]
    # Most recent contiguous valid run; never create templates across invalid/missing gaps.
    last_invalid = np.flatnonzero(~valid)
    tail = rr[last_invalid[-1] + 1 :] if len(last_invalid) else rr
    reliability = float(valid.mean()) if len(rr) else 0.0
    result.update(
        quality_peak_count=len(peaks),
        quality_peak_reliability=reliability,
        quality_entropy_rr_count=min(256, len(tail)),
    )
    if len(good) < 2:
        return result
    result["ecg_rr_mean"] = _finite(good.mean())
    result["ecg_hr"] = _finite(60 / good.mean())
    pairs = valid[:-1] & valid[1:]
    left, right = rr[:-1][pairs], rr[1:][pairs]
    differences = right - left
    if len(good) >= 10 and reliability >= 0.8:
        deviation = np.std(good, ddof=1)
        result.update(
            ecg_rr_median=_finite(np.median(good)),
            ecg_rr_iqr=_finite(np.percentile(good, 75) - np.percentile(good, 25)),
            ecg_rr_mad=_finite(np.median(np.abs(good - np.median(good)))),
            ecg_rr_cv=_finite(deviation / good.mean()),
        )
        if duration >= 300:
            result["ecg_sdnn"] = _finite(deviation)
            if len(differences):
                result["ecg_rmssd"] = _finite(np.sqrt(np.mean(differences**2)))
                result["ecg_pnn50"] = _finite(np.mean(np.abs(differences) > 0.05))
        if len(differences) >= 3:
            result["ecg_sd1"] = _finite(np.std(differences, ddof=1) / np.sqrt(2))
            result["ecg_sd2"] = _finite(np.std(right + left, ddof=1) / np.sqrt(2))
            if np.std(left) > 1e-12 and np.std(right) > 1e-12:
                result["ecg_rr_lag1_corr"] = _finite(np.corrcoef(left, right)[0, 1])
        if len(tail) >= MIN_ENTROPY_RR:
            result["ecg_rr_sample_entropy"] = _sample_entropy(tail)
    early = rr[(peaks[1:] < peaks[0] + 60) & valid]
    late = rr[(peaks[1:] > peaks[-1] - 60) & valid]
    if len(early) > 2 and len(late) > 2:
        result["ecg_hr_trend"] = _finite(60 / late.mean() - 60 / early.mean())
    for seconds in [60, 300]:
        if duration < seconds:
            continue
        recent = peaks[1:] > duration - seconds
        window_valid = valid & recent
        values = rr[window_valid]
        if len(values) < 10 or not recent.any() or window_valid.sum() / recent.sum() < 0.8:
            continue
        result[f"ecg_hr_{seconds}s"] = _finite(60 / values.mean())
        result[f"ecg_rr_cv_{seconds}s"] = _finite(np.std(values, ddof=1) / values.mean())
        recent_pairs = window_valid[:-1] & window_valid[1:]
        if recent_pairs.any():
            result[f"ecg_rmssd_{seconds}s"] = _finite(np.sqrt(np.mean(np.diff(rr)[recent_pairs] ** 2)))
    for key in ["ecg_hr", "ecg_rr_cv", "ecg_rmssd"]:
        short, long = result[f"{key}_60s"], result[f"{key}_300s"]
        if short is not None and long is not None:
            result[f"{key}_60s_vs_300s_delta"] = _finite(short - long)
    return result


def extract_features(signal, fs):
    x = np.asarray(signal, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[1] < 1 or not np.isfinite(fs) or fs < 30:
        raise ValueError("ECG must have channels and a finite sampling rate >=30 Hz")
    missing = float(np.mean(~np.isfinite(x))) if x.size else 1.0
    flat = []
    block = max(1, int(fs))
    for lead in x.T:
        for start in range(0, len(lead), block):
            values = lead[start : start + block]
            values = values[np.isfinite(values)]
            flat.append(len(values) == 0 or np.ptp(values) < 1e-5)
    saturation = []
    for lead in x.T:
        finite = lead[np.isfinite(lead)]
        saturation.append(
            float(np.mean((finite == finite.min()) | (finite == finite.max()))) if len(finite) else 1.0
        )
    bands = [_qrs_band(lead, fs) if len(lead) > 2 * fs else np.array([]) for lead in x.T]
    peaks = [
        _peaks(lead, fs, filtered=band) if len(lead) > 2 * fs and np.any(np.isfinite(lead)) else np.array([])
        for lead, band in zip(x.T, bands, strict=True)
    ]
    # RR bounds alone cannot exclude short missing spans: inspect every sample between beats.
    beat_samples = np.rint(peaks[0] * fs).astype(int)
    missing_prefix = np.r_[0, np.cumsum(~np.isfinite(x[:, 0]))]
    interval_valid = missing_prefix[beat_samples[1:] + 1] == missing_prefix[beat_samples[:-1]]
    result = _rr_features(peaks[0], len(x) / fs, valid_mask=interval_valid)
    disagreement = max(_lead_disagreement(peaks[0], other) for other in peaks[1:]) if len(peaks) > 1 else None
    result.update(
        quality_missing=missing,
        quality_saturation=max(saturation),
        quality_flatline=float(np.mean(flat)) if flat else 1.0,
        quality_lead_disagreement=disagreement,
        quality_qrs_kurtosis=_finite(kurtosis(bands[0], fisher=False))
        if len(bands[0]) and np.std(bands[0]) > 0
        else None,
    )
    return result


def assess_quality(row, config):
    """One quality gate shared by cohort audit, training and live replay."""
    q = config["quality"]
    reason = None
    if row["quality_missing"] > q["max_missing_fraction"]:
        reason = "missing_signal"
    elif row["quality_flatline"] > q["max_flatline_fraction"]:
        reason = "flatline"
    elif row["quality_peak_count"] < q["min_peaks"] or row["quality_peak_reliability"] < 0.8:
        reason = "unreliable_peaks"
    elif q.get("min_qrs_kurtosis") is not None and (
        row.get("quality_qrs_kurtosis") is None or row["quality_qrs_kurtosis"] < q["min_qrs_kurtosis"]
    ):
        reason = "non_qrs_signal"
    elif (
        q.get("max_saturation_fraction") is not None
        and row["quality_saturation"] > q["max_saturation_fraction"]
    ):
        reason = "saturated_signal"
    elif (
        row["quality_lead_disagreement"] is not None
        and row["quality_lead_disagreement"] > q["max_lead_disagreement"]
    ):
        reason = "lead_disagreement"
    return {
        "accepted": reason is None,
        "reason": reason,
        **{k: v for k, v in row.items() if k.startswith("quality_")},
    }


def feature_row(signal, fs, clinical, baseline, config):
    row = extract_features(signal, fs)
    quality = assess_quality(row, config)
    for key in CLINICAL_FIELDS:
        value = clinical.get(key)
        if isinstance(value, dict):
            value = value.get("value") if value.get("available") else None
        row["clinical_" + key] = _finite(value) if isinstance(value, (int, float)) else None
        row["clinical_" + key + "_available"] = int(row["clinical_" + key] is not None)
    count = baseline.get("count", 0)
    row["baseline_count"] = count
    row["baseline_warmup"] = int(count < config["baseline"]["minimum_updates"])
    for key in BASELINE_FIELDS:
        mean = baseline.get("means", {}).get(key)
        value = row.get(key)
        deviation = _finite(value - mean) if value is not None and mean is not None else None
        row["baseline_" + key + "_deviation"] = deviation
        variance = baseline.get("variances", {}).get(key)
        row["baseline_" + key + "_zscore"] = (
            _finite(deviation / np.sqrt(variance))
            if deviation is not None
            and variance is not None
            and variance > BASELINE_VARIANCE_FLOOR
            and baseline.get("counts", {}).get(key, 0) >= 5
            else None
        )
    return row, quality


def update_baseline(baseline, features, config):
    count = baseline.get("count", 0) + 1
    means = dict(baseline.get("means", {}))
    counts = dict(baseline.get("counts", {}))
    variances = dict(baseline.get("variances", {}))
    cap = config["baseline"]["max_updates"]
    for key in BASELINE_FIELDS:
        value = _finite(features.get(key))
        if value is not None:
            n = min(counts.get(key, 0) + 1, cap)
            delta = value - means.get(key, value)
            alpha = 1 / n
            means[key] = means.get(key, value) + alpha * delta
            variances[key] = (1 - alpha) * (variances.get(key, 0) + alpha * delta**2)
            counts[key] = n
    return {"count": min(count, cap), "means": means, "counts": counts, "variances": variances}
