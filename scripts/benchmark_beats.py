"""Separate beat-localization benchmark; never AF forecasting training data."""

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import wfdb

from cardiotwin.config import load_config
from cardiotwin.features import FEATURE_VERSION, _peaks, assess_quality, extract_features

BEAT_SYMBOLS = set("NLRaAVFSjeEJ/QfnB")


def match_beats(reference, predicted, tolerance=0.15):
    reference, predicted = sorted(reference), sorted(predicted)
    i = j = matched = 0
    errors, matched_pairs = [], []
    while i < len(reference) and j < len(predicted):
        error = predicted[j] - reference[i]
        if abs(error) <= tolerance:
            matched += 1
            errors.append(abs(error))
            matched_pairs.append((i, j, reference[i], predicted[j]))
            i += 1
            j += 1
        elif error < -tolerance:
            j += 1
        else:
            i += 1
    rr_errors = [
        abs((b[3] - a[3]) - (b[2] - a[2]))
        for a, b in zip(matched_pairs, matched_pairs[1:])
        if b[0] == a[0] + 1 and b[1] == a[1] + 1
    ]
    return {
        "reference_beats": len(reference),
        "detected_beats": len(predicted),
        "matched_beats": matched,
        "missed_beats": len(reference) - matched,
        "extra_beats": len(predicted) - matched,
        "sensitivity": matched / len(reference) if reference else None,
        "precision": matched / len(predicted) if predicted else None,
        "median_absolute_timing_error_ms": float(np.median(errors) * 1000) if errors else None,
        "matched_adjacent_rr_intervals": len(rr_errors),
        "median_absolute_rr_error_ms": float(np.median(rr_errors) * 1000) if rr_errors else None,
        "p95_absolute_rr_error_ms": float(np.percentile(rr_errors, 95) * 1000) if rr_errors else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=("mitdb", "nstdb"), default="mitdb")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", nargs="+", default=["100", "201"])
    parser.add_argument("--legacy-source", type=Path, help="Trusted local previous features.py snapshot")
    parser.add_argument("--seconds", type=float, default=600, help="0 benchmarks the complete recording")
    parser.add_argument("--xqrs", action="store_true", help="Compare installed WFDB XQRS on the same signals")
    parser.add_argument(
        "--chunk-seconds", type=float, default=600, help="Trailing-window detector reset length"
    )
    args = parser.parse_args()
    if args.seconds < 0 or args.chunk_seconds <= 2:
        raise ValueError("seconds must be nonnegative and chunks must exceed two seconds")
    detectors = {FEATURE_VERSION: _peaks}
    if args.xqrs:

        def xqrs(lead, fs):
            samples = processing.xqrs_detect(lead, fs=fs, verbose=False)
            return samples[(samples >= fs) & (samples < len(lead) - int(0.1 * fs))] / fs

        from wfdb import processing

        detectors["wfdb-xqrs"] = xqrs
    if args.legacy_source:
        spec = importlib.util.spec_from_file_location("legacy_features", args.legacy_source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        detectors["causal-rr-v1"] = module._peaks
    rows = []
    hashes = {}
    quality_rows = []
    quality_config = load_config()
    quality_config["quality"].update(min_qrs_kurtosis=5.0, max_saturation_fraction=0.05)
    for name in args.records:
        if not name.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Invalid benchmark record ID")
        stem = args.data_dir / name
        header = wfdb.rdheader(str(stem))
        stop = min(header.sig_len, int(args.seconds * header.fs)) if args.seconds else header.sig_len
        recording = wfdb.rdrecord(str(stem), sampfrom=0, sampto=stop)
        annotations = wfdb.rdann(str(stem), "atr", sampfrom=0, sampto=stop)
        chunk_samples = int(args.chunk_seconds * header.fs)
        reference = [
            sample / header.fs
            for sample, symbol in zip(annotations.sample, annotations.symbol)
            if symbol in BEAT_SYMBOLS
            and 1
            <= (sample % chunk_samples) / header.fs
            < min(args.chunk_seconds, (stop - (sample // chunk_samples) * chunk_samples) / header.fs) - 0.1
        ]
        for start in range(0, stop, chunk_samples):
            chunk = recording.p_signal[start : min(start + chunk_samples, stop)]
            if len(chunk) > 2 * header.fs:
                features = extract_features(chunk, header.fs)
                quality_rows.append(
                    {
                        "record": name,
                        "start_seconds": start / header.fs,
                        "legacy_quality": assess_quality(features, load_config()),
                        "candidate_quality": assess_quality(features, quality_config),
                    }
                )
        for extension in ["hea", "dat", "atr"]:
            path = stem.with_suffix("." + extension)
            with path.open("rb") as stream:
                hashes[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
        for version, detector in detectors.items():
            for lead in range(recording.p_signal.shape[1]):
                started = time.perf_counter()
                predicted = np.concatenate(
                    [
                        start / recording.fs
                        + detector(
                            recording.p_signal[start : min(start + chunk_samples, stop), lead], recording.fs
                        )
                        for start in range(0, stop, chunk_samples)
                        if min(chunk_samples, stop - start) > 2 * recording.fs
                    ]
                )
                elapsed = time.perf_counter() - started
                rows.append(
                    {
                        "record": name,
                        "lead": lead + 1,
                        "version": version,
                        "duration_seconds": stop / recording.fs,
                        "processing_seconds": elapsed,
                        **match_beats(reference, predicted),
                    }
                )
                if args.dataset == "nstdb":

                    def noisy(timestamp):
                        return timestamp >= 300 and (timestamp - 300) % 240 < 120

                    for condition in ("clean", "noise"):
                        selected = condition == "noise"
                        ref = [t for t in reference if noisy(t) == selected]
                        pred = [t for t in predicted if noisy(t) == selected]
                        rows[-1][condition] = match_beats(ref, pred)
                    suffix = name.split("e", 1)[1]
                    rows[-1]["noise_snr_db"] = -6 if suffix == "_6" else int(suffix)
    report = {
        "dataset": (
            "MIT-BIH Arrhythmia Database" if args.dataset == "mitdb" else "MIT-BIH Noise Stress Test Database"
        )
        + " v1.0.0",
        "source": f"https://physionet.org/content/{args.dataset}/1.0.0/",
        "license": f"https://physionet.org/content/{args.dataset}/view-license/1.0.0/",
        "attribution": "Moody GB, Mark RG. IEEE Eng Med Biol20(3):45-50(2001). DOI10.13026/C2F305.",
        "requested_duration_seconds": args.seconds or "complete",
        "detector_chunk_seconds": args.chunk_seconds,
        "quality_candidate_config": quality_config["quality"],
        "quality_results": quality_rows,
        "matching_tolerance_seconds": 0.15,
        "noise_protocol": "NST: clean first 300s, then alternating 120s noise/120s clean; separate stratum RR errors can span masked boundaries"
        if args.dataset == "nstdb"
        else None,
        "reference": "Reviewed atr beat annotations, declared beat symbols, startup/edge exclusions",
        "files_sha256": hashes,
        "results": rows,
        "limitations": [
            "Detector comparison; record selection and reviewed annotations must be interpreted separately from forecasting validation.",
            "No AF forecasting labels, clinical histories or training examples are joined.",
            "Recorded hashes establish local reproducibility, not independently published checksums.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    for row in rows:
        print(
            row["record"],
            row["lead"],
            row["version"],
            "sensitivity",
            row["sensitivity"],
            "precision",
            row["precision"],
        )


if __name__ == "__main__":
    main()
