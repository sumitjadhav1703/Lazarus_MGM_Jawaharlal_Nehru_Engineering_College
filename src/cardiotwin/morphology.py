"""Fixed native-rate beat-shape proxies; no adjudicated atrial/ectopic labels."""

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .features import _finite, _peaks
from .progress import Progress

SUFFIXES = ("pre_dispersion_recent", "anchor_dispersion_recent", "pre_dispersion_change", "pre_template_change")
MORPHOLOGY_SCHEMA = [f"morph_c{c}_{name}" for c in range(2) for name in SUFFIXES]


def _channel(values, *, overlays, anchors=None, quality=False):
    result = dict(available=False, anchor_samples=[], candidates=[0, 0], valid_beats=[0, 0],
                  rejected={}, features=dict.fromkeys(SUFFIXES), overlays=[])
    if quality:
        result.update(quality_halves=[], original_reference_count=0)
    finite = np.isfinite(values)
    result["finite_fraction"] = float(finite.mean())
    if not finite.any():
        result["rejected"] = {"no_finite_samples": 1}
        return result
    q5, median, q95 = np.percentile(values[finite], [5, 50, 95])
    history_scale = q95-q5
    result.update(history_scale=_finite(history_scale), history_median=_finite(median),
                  flat_fraction=float(np.mean(np.diff(values[finite]) == 0)) if finite.sum() > 1 else None)
    if not np.isfinite(history_scale) or history_scale <= 0:
        result["rejected"] = {"history_scale": 1}
        return result
    if anchors is None:
        detected = np.rint(_peaks((values-median)/history_scale, 200)*200).astype(int)
        anchors = []
        for peak in detected:
            start, end = max(0, peak-20), min(len(values), peak+11)
            local = values[start:end]
            present = np.isfinite(local)
            if not present.any():
                continue
            deviations = np.where(present, np.abs(local-np.median(local[present])), -np.inf)
            anchors.append(int(start)+int(np.argmax(deviations)))
        refined = []
        for anchor in sorted(set(anchors)):
            if not refined or anchor-refined[-1] >= 50:
                refined.append(anchor)
        if quality:
            result["original_reference_count"] = len(detected)
    else:
        refined = list(anchors)
        result["original_reference_count"] = None  # No independent secondary detection.
    result["anchor_samples"] = refined
    halves, positions, rejected = [[], []], [[], []], Counter()
    scales, concentrations = [[], []], [[], []]
    relative = np.arange(-60, 40)/200
    midpoints = (np.arange(99)-59.5)/200
    central = (midpoints >= -.06) & (midpoints < .08)
    for anchor in refined:
        half = int(anchor >= 60000)
        result["candidates"][half] += 1
        start, end = anchor-60, anchor+40
        if start < 0 or end > len(values):
            rejected["history_boundary"] += 1
            continue
        raw = values[start:end]
        if not np.isfinite(raw).all():
            rejected["missing_segment"] += 1
            continue
        left, right = np.median(raw[:8]), np.median(raw[92:100])
        detrended = raw - (left+(right-left)*(relative+.280)/.460)
        scale = float(np.diff(np.percentile(detrended[48:76], [5, 95]))[0])
        if not np.isfinite(scale) or scale <= 0 or scale < 1e-6*history_scale:
            rejected["segment_scale"] += 1
            continue
        halves[half].append(detrended/scale)
        positions[half].append(anchor)
        if quality:
            energy = np.diff(detrended)**2
            denominator = float(energy.sum())
            concentrations[half].append(float(energy[central].sum()/denominator)
                if np.isfinite(denominator) and denominator > 0 else None)
            scales[half].append(scale)
    result["valid_beats"] = [len(x) for x in halves]
    result["rejected"] = dict(rejected)
    sufficient = all(count >= 60 and count >= .8*result["candidates"][n]
                     for n, count in enumerate(result["valid_beats"]))
    if quality:
        for half in range(2):
            median_scale = float(np.median(scales[half])) if scales[half] else None
            ratios = [a/median_scale for a in scales[half]]
            fraction_c = float(np.mean([c is not None and c >= .5 for c in concentrations[half]])) if ratios else 0.
            fraction_a = float(np.mean(np.asarray(ratios) >= .25)) if ratios else 0.
            result["quality_halves"].append(dict(concentrations=concentrations[half], scale_ratios=ratios,
                median_scale=median_scale, concentrated_fraction=fraction_c, scaled_fraction=fraction_a))
            if fraction_c < .9:
                rejected[f"half_{half}_concentration"] += 1
            if fraction_a < .95:
                rejected[f"half_{half}_amplitude"] += 1
        result["rejected"] = dict(rejected)
        sufficient &= all(h["concentrated_fraction"] >= .9 and h["scaled_fraction"] >= .95
                          for h in result["quality_halves"])
    if overlays and (sufficient or quality):
        for half, beats in enumerate(halves):
            if not beats:
                continue
            chosen = [(j, "deterministic") for j in sorted(set([0, len(beats)//2, len(beats)-1]))]
            if quality:
                worst = min(range(len(beats)), key=lambda j: (
                    concentrations[half][j] if concentrations[half][j] is not None else -1, positions[half][j]))
                chosen.append((worst, "lowest_concentration"))
            for j, selection in chosen:
                anchor = positions[half][j]
                entry = dict(half=half, anchor_sample=anchor, raw=values[anchor-60:anchor+40].tolist(),
                             normalized=beats[j].tolist())
                if quality:
                    entry.update(selection=selection, concentration=concentrations[half][j])
                result["overlays"].append(entry)
    if not sufficient:
        return result
    early, recent = (np.asarray(x) for x in halves)
    te, tr = np.median(early, axis=0), np.median(recent, axis=0)

    def dispersion(beats, template, region):
        return float(np.median(np.sqrt(np.mean((beats[:, region]-template[region])**2, axis=1))))

    pre, anchor_region = slice(10, 44), slice(48, 76)
    ve, vr = dispersion(early, te, pre), dispersion(recent, tr, pre)
    result["features"] = dict(zip(SUFFIXES, [vr, dispersion(recent, tr, anchor_region), vr-ve,
                                               float(np.sqrt(np.mean((tr[pre]-te[pre])**2)))], strict=True))
    result["available"] = True
    return result


def extract_morphology(signal, fs, *, decision_sample=None, validity=None, overlays=False,
                       anchor_mode="independent"):
    """All processing follows the decision slice; masks survive caller zero filling."""
    if anchor_mode not in ("independent", "shared-channel0-v1"):
        raise ValueError("Unknown declared anchor mode")
    signal = np.asarray(signal, dtype=float)
    if fs != 200 or signal.ndim != 2 or signal.shape[1] != 2:
        raise ValueError("Declared morphology requires native 200-Hz two-channel ECG")
    end = len(signal) if decision_sample is None else decision_sample
    if not np.isfinite(end) or int(end) != end or not 120000 <= end <= len(signal):
        raise ValueError("A complete 600-second history and integer decision sample are required")
    end = int(end)
    history = signal[end-120000:end].copy()
    if validity is not None:
        mask = np.asarray(validity, dtype=bool)
        if mask.shape != signal.shape:
            raise ValueError("Validity mask must match the signal")
        history[~mask[end-120000:end]] = np.nan
    if anchor_mode == "shared-channel0-v1":
        first = _channel(history[:, 0], overlays=overlays, quality=True)
        second = _channel(history[:, 1], overlays=overlays, anchors=first["anchor_samples"], quality=True)
        if not first["available"]:
            second["available"] = False
            second["features"] = dict.fromkeys(SUFFIXES)
            second["rejected"]["unavailable_reference"] = 1
        channels = [first, second]
    else:
        channels = [_channel(history[:, c], overlays=overlays) for c in range(2)]
    features = {f"morph_c{c}_{name}": value for c, channel in enumerate(channels)
                for name, value in channel["features"].items()}
    return dict(available=all(c["available"] for c in channels), features=features, channels=channels)


def select_audit_rows(rows, roles):
    """Outcome-blind fixed identities; never replace a failed measurement."""
    selected = {}
    for role in roles:
        people = sorted(role["fit"], key=lambda g: hashlib.sha256(f"42|{g}".encode()).digest())[:12]
        if len(people) != 12 or len(set(people)) != 12:
            raise ValueError("Each fitting role needs twelve distinct audit people")
        selected[role["fold"]] = set(people)
    best = {}
    for i, row in enumerate(rows):
        if not row["accepted"] or row["timestamp"] < 600:
            continue
        group = row["group_id"]
        memberships = [fold for fold, people in selected.items() if group in people]
        if not memberships:
            continue
        rank = hashlib.sha256(f"42|{row['record_id']}|{float(row['timestamp']):.9f}".encode()).digest()
        for fold in memberships:
            key = (fold, group)
            if key not in best or rank < best[key][0]:
                best[key] = (rank, dict(fold=fold, row_index=i, group_id=group, patient_id=row["patient_id"],
                                       record_id=row["record_id"], timestamp=row["timestamp"]))
    if len(best) != 36:
        raise ValueError("Selected audit person has no originally accepted complete history")
    return [item for _, item in sorted(best.values(), key=lambda x: (x[1]["fold"], x[1]["group_id"]))]


def measurement_gate(valid_by_fold):
    counts = list(valid_by_fold)
    if len(counts) != 3 or any(int(n) != n or not 0 <= n <= 12 for n in counts):
        raise ValueError("Three fold counts in [0,12] are required")
    passed = sum(counts) >= 33 and all(n >= 10 for n in counts)
    return dict(status="passed" if passed else "failed", valid_by_fold=counts,
                valid_total=sum(counts), required_total=33, required_per_fold=10)


def measure_histories(selected, records, reader, output, *, deadline=float("inf")):
    """Bounded audit only; identities must be durable before any signal read."""
    output = Path(output)
    if json.loads((output / "selected.json").read_text()) != selected:
        raise ValueError("Saved selection differs")
    if len(selected) != 36 or [sum(r["fold"] == f for r in selected) for f in range(3)] != [12]*3:
        raise ValueError("Exactly twelve requests per original fold required")
    counts = [0, 0, 0]
    with (output / "measurements.jsonl").open("x") as handle, Progress(
        "morphology.measurement", total=36, device="cpu", sampling_hz=200
    ) as progress:
        for completed, item in enumerate(selected, 1):
            if time.monotonic() >= deadline:
                raise TimeoutError("Morphology audit deadline reached")
            record = records[item["record_id"]]
            signal = reader(record, item["timestamp"]-600, item["timestamp"])
            measurement = extract_morphology(signal, record.fs, overlays=True)
            counts[item["fold"]] += int(measurement["available"])
            handle.write(json.dumps({**item, "measurement": measurement}, allow_nan=False)+"\n")
            handle.flush()
            progress.update(completed=completed, force=True, valid_by_fold=counts.copy())
    return measurement_gate(counts)
