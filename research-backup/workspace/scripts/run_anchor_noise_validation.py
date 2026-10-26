"""One fixed known-anchor numerical check; no real signals, fitting or submission."""

import argparse
import json
import time
import traceback
from collections import defaultdict

import numpy as np

from cardiotwin.data import file_hash
from cardiotwin.investigation import _digest, _write
from cardiotwin.morphology import extract_morphology
from cardiotwin.precursor import fresh_output, source_manifest
from cardiotwin.progress import Progress

MODE = "shared-channel0-v1"


def synthetic_signal():
    t = np.arange(120000)/200

    def phase(center):
        return (t-center+.5) % 1-.5

    pulses = np.column_stack((np.exp(-.5*(phase(0)/.015)**2), -.5*np.exp(-.5*(phase(.02)/.015)**2)))
    broad = .6*np.exp(-.5*(phase(.3)/.08)**2)+.4*np.exp(-.5*(phase(.55)/.08)**2)
    return pulses+broad[:, None]


def known_reference(signal):
    """Independent vector calculation at generating times, without detector helpers."""
    signal = np.asarray(signal)
    if signal.shape != (120000, 2) or not np.isfinite(signal).all():
        raise ValueError("Oracle requires the fixed finite synthetic history")
    times = np.arange(1, 600)
    offsets = np.arange(-60, 40)
    segments = signal[(times*200)[:, None]+offsets]
    left = np.median(segments[:, :8], axis=1)
    right = np.median(segments[:, 92:], axis=1)
    baseline = left[:, None, :]+(right-left)[:, None, :]*((offsets/200+.280)/.460)[None, :, None]
    detrended = segments-baseline
    limits = np.percentile(detrended[:, 48:76], [5, 95], axis=1)
    amplitude = limits[1]-limits[0]
    normalized = detrended/amplitude[:, None, :]
    if not np.isfinite(normalized).all() or np.any(amplitude <= 0):
        raise ValueError("Invalid synthetic oracle scale")
    early, recent = normalized[times < 300], normalized[times >= 300]
    te, tr = np.median(early, axis=0), np.median(recent, axis=0)
    pre_early = np.median(np.sqrt(np.mean((early[:, 10:44]-te[10:44])**2, axis=1)), axis=0)
    pre_recent = np.median(np.sqrt(np.mean((recent[:, 10:44]-tr[10:44])**2, axis=1)), axis=0)
    q_recent = np.median(np.sqrt(np.mean((recent[:, 48:76]-tr[48:76])**2, axis=1)), axis=0)
    template_delta = np.sqrt(np.mean((tr[10:44]-te[10:44])**2, axis=0))
    features = {}
    for c in range(2):
        for name, values in [("pre_dispersion_recent", pre_recent), ("anchor_dispersion_recent", q_recent),
                             ("pre_dispersion_change", pre_recent-pre_early), ("pre_template_change", template_delta)]:
            features[f"morph_c{c}_{name}"] = float(values[c])
    return dict(features=features, clean_scales=np.median(amplitude, axis=0).tolist())


def anchor_accuracy(samples):
    actual, expected = np.sort(np.asarray(samples)/200), np.arange(1, 600)
    i = j = matches = 0
    while i < len(actual) and j < len(expected):
        if abs(actual[i]-expected[j]) <= .02+1e-12:
            matches += 1
            i += 1
            j += 1
        elif actual[i] < expected[j]:
            i += 1
        else:
            j += 1
    return dict(precision=matches/len(actual) if len(actual) else 0., recall=matches/599)


def shape_signal(clean, scales, *, alternating):
    t = np.arange(120000)/200
    beat = np.rint(t).astype(int)
    eligible = (beat >= 300) & (beat <= 599)
    if alternating:
        eligible &= beat % 2 == 0
    triangle = np.maximum(0, 1-np.abs(t-beat+.16)/.04)*eligible
    return clean+triangle[:, None]*(.2*np.asarray(scales)*[1, -1])


def run_cases(handle):
    clean = synthetic_signal()
    scales = known_reference(clean)["clean_scales"]
    conditions = dict(baseline=clean, coherent=shape_signal(clean, scales, alternating=False),
                      alternating=shape_signal(clean, scales, alternating=True))
    failures, baseline_values, means, completed = [], {}, defaultdict(list), 0
    with Progress("anchor.synthetic_noise_reference", total=135, device="cpu") as progress:
        for seed in range(42, 47):
            unit_noise = np.random.default_rng(seed).normal(size=clean.shape)
            for sigma in (0., .005, .01):
                for condition, deterministic in conditions.items():
                    signal = deterministic+sigma*unit_noise
                    actual = extract_morphology(signal, 200, anchor_mode=MODE)
                    oracle = known_reference(signal)
                    accuracy = anchor_accuracy(actual["channels"][0]["anchor_samples"])
                    issues = []
                    if not actual["available"] or min(accuracy.values()) < .95:
                        issues.append("availability_or_anchor_accuracy")
                    differences = {name: value-oracle["features"][name] if value is not None else None
                                   for name, value in actual["features"].items()}
                    if any(v is None or abs(v) > .02 for v in differences.values()):
                        issues.append("oracle_agreement")
                    paired = {}
                    if condition == "baseline":
                        baseline_values[seed, sigma] = (actual["features"], oracle["features"])
                    else:
                        field = "pre_template_change" if condition == "coherent" else "pre_dispersion_recent"
                        for source, features, base in zip(("extractor", "oracle"),
                                (actual["features"], oracle["features"]), baseline_values[seed, sigma], strict=True):
                            for c in range(2):
                                key = f"morph_c{c}_{field}"
                                delta = features[key]-base[key] if features[key] is not None and base[key] is not None else None
                                paired[f"{source}_{key}"] = delta
                                if delta is None or delta < .02:
                                    issues.append(f"{source}_channel{c}_shape_response")
                    entry = dict(seed=seed, sigma=sigma, condition=condition, accuracy=accuracy,
                                 available=actual["available"], extractor=actual["features"], oracle=oracle["features"],
                                 differences=differences, paired_increments=paired, failed_checks=issues)
                    handle.write(json.dumps(entry, allow_nan=False)+"\n")
                    handle.flush()
                    failures.extend(dict(seed=seed, sigma=sigma, condition=condition, check=issue) for issue in issues)
                    for source in ("extractor", "oracle"):
                        for name, value in entry[source].items():
                            if value is not None:
                                means[sigma, condition, source, name].append(value)
                    completed += 1
                    progress.update(completed=completed)
                signal = clean+sigma*unit_noise
                base = baseline_values[seed, sigma][0]
                history_scale = np.percentile(signal, 95, axis=0)-np.percentile(signal, 5, axis=0)
                transforms = {"gain_half": .5*signal, "gain_double": 2*signal, "polarity": -signal,
                              "offset_positive": signal+5*history_scale, "offset_negative": signal-5*history_scale,
                              "linear_drift": signal+np.linspace(0, .5, len(signal))[:, None]*history_scale}
                for name, transformed in transforms.items():
                    actual = extract_morphology(transformed, 200, anchor_mode=MODE)
                    accuracy = anchor_accuracy(actual["channels"][0]["anchor_samples"])
                    differences = {key: value-base[key] if value is not None and base[key] is not None else None
                                   for key, value in actual["features"].items()}
                    passed = actual["available"] and min(accuracy.values()) >= .95 and all(
                        v is not None and abs(v) <= .02 for v in differences.values())
                    handle.write(json.dumps(dict(seed=seed, sigma=sigma, transform=name, passed=passed,
                        accuracy=accuracy, extractor=actual["features"], differences=differences), allow_nan=False)+"\n")
                    handle.flush()
                    if not passed:
                        failures.append(dict(seed=seed, sigma=sigma, transform=name, check="nuisance_invariance"))
                    completed += 1
                    progress.update(completed=completed)
    return dict(status="passed" if not failures else "failed", cases=completed, failed_checks=failures,
                clean_scales=scales, seed_means=[dict(sigma=k[0], condition=k[1], source=k[2], field=k[3],
                    mean=float(np.mean(v))) for k, v in sorted(means.items())])


def run(args):
    output = fresh_output(args.output)
    started = time.monotonic()
    try:
        source = source_manifest()
        for path in ("scripts/run_anchor_noise_validation.py", "tests/test_morphology_validation.py", "tests/test_morphology.py",
                     "reports/ml-investigation/next-cycle/anchor-repair-20261013/spec.md",
                     "reports/ml-investigation/next-cycle/morphology-cycle-20261012/spec.md"):
            source["files"][path] = file_hash(path)
        source["sha256"] = _digest(source["files"])
        spec_hash = file_hash(args.spec)
        _write(output / "predeclaration.json", dict(source=source, spec_sha256=spec_hash, mode=MODE,
            sigmas=[0, .005, .01], seeds=list(range(42, 47)), cases=135, real_data=False, fits=0))
        with (output / "cases.jsonl").open("x") as handle:
            result = run_cases(handle)
        for name, checksum in source["files"].items():
            if file_hash(name) != checksum:
                raise ValueError(f"Source changed during validation: {name}")
        if file_hash(args.spec) != spec_hash or result["cases"] != 135:
            raise ValueError("Declaration changed or incomplete cases")
        result.update(elapsed_seconds=time.monotonic()-started, cases_sha256=file_hash(output / "cases.jsonl"),
                      predeclaration_sha256=file_hash(output / "predeclaration.json"), real_reads=0, fits=0, remote_jobs=0)
        _write(output / "receipt.json", result)
        print(json.dumps({k: v for k, v in result.items() if k != "seed_means"}), flush=True)
        return 0 if result["status"] == "passed" else 2
    except BaseException as error:
        _write(output / "receipt.json", dict(status="error", error_type=type(error).__name__, error=str(error),
               traceback=traceback.format_exc(), elapsed_seconds=time.monotonic()-started))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
