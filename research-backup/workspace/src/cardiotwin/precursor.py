"""Development-only precursor experiment; future labels are offline supervision."""

import csv
import gzip
import hashlib
import json
import math
import pickle
import shutil
import time
import warnings
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from .data import file_hash
from .features import CORE_ECG_FIELDS, ENGINEERED_ECG_FIELDS
from .investigation import _digest, _extraction_contract, _group, _partitions, _write
from .modeling import build_pipeline, matrix, raw_score
from .progress import Progress

ROLE_NAMES = ("fit", "stop", "calibration", "policy", "outer")
LINEAR_ARMS = ("pointwise_pair_control", "within_person_rank")
LINEAR_SCHEMA = sorted(CORE_ECG_FIELDS + ENGINEERED_ECG_FIELDS)


@dataclass(frozen=True)
class EventPair:
    group_id: str
    patient_id: str
    record_id: str
    onset: float
    positive_index: int
    negative_index: int


def _identities(rows, events):
    patients, records, seen_events = {}, {}, set()
    for item in (*events,):
        key = (item["record_id"], item["onset"])
        if key in seen_events:
            raise ValueError("duplicate record/onset event")
        seen_events.add(key)
    for items in (rows, events):
        for item in items:
            patient, record, group = item["patient_id"], item["record_id"], _group(item)
            if patients.setdefault(patient, group) != group or records.setdefault(record, (patient, group)) != (patient, group):
                raise ValueError("inconsistent patient/record/group identity")
    return patients, records


def make_roles(rows, events, index, split, *, seed=42):
    if seed != 42 or split.get("seed") != seed:
        raise ValueError("original seed 42 is required")
    patients, _ = _identities(rows, events)
    allowed = set(index["allowed_groups"])
    groups = sorted({_group(r) for r in rows})
    if (set(groups) != allowed or allowed != set(split["development_groups"])
            or allowed & set(split["test_groups"])):
        raise ValueError("development membership differs from original split")
    for patient, group in patients.items():
        if index["group_ids"].get(patient) != group or split["group_ids"].get(patient) != group:
            raise ValueError("patient alias differs from bound identity")
    # _partitions depends only on group membership and available classes, not row multiplicity.
    classes = defaultdict(set)
    for r in rows:
        if r["accepted"] and r["label"] in (0, 1):
            classes[_group(r)].add(r["label"])
    summaries = [dict(patient_id=g, group_id=g, accepted=c is not None, label=c)
                 for g in groups for c in sorted(classes[g]) or [None]]
    values = np.array(groups)
    positive = {_group(e) for e in events}
    strata = np.array([int(g in positive) for g in groups])
    result = []
    for fold, (training, outer) in enumerate(StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(
            np.zeros(len(groups)), strata, values)):
        fit, cal, policy = _partitions([r for r in summaries if _group(r) in set(values[training])], seed + fold)
        _, stop, _ = _partitions([r for r in summaries if _group(r) in set(fit)], seed + 100 + fold)
        result.append(dict(fold=fold, fit=sorted(set(fit) - set(stop)), stop=sorted(stop),
                           calibration=sorted(cal), policy=sorted(policy), outer=sorted(values[outer])))
    return result


def build_event_pairs(rows, events, groups, config):
    _identities(rows, events)
    groups = set(groups)
    target = config["target"]
    by_record, event_records = defaultdict(list), defaultdict(list)
    for i, r in enumerate(rows):
        if _group(r) in groups:
            by_record[(r["patient_id"], r["record_id"])].append((float(r["timestamp"]), i))
    ordered = sorted((e for e in events if _group(e) in groups),
                     key=lambda e: (_group(e), e["patient_id"], e["record_id"], e["onset"]))
    for e in ordered:
        event_records[(e["patient_id"], e["record_id"])].append(e["onset"])
    for key in by_record:
        by_record[key].sort()
        times = [t for t, _ in by_record[key]]
        if len(times) != len(set(times)):
            raise ValueError("duplicate record/timestamp row")
    pairs, exclusions, used, observable, recurrence = [], [], set(), 0, Counter()
    import bisect

    for e in ordered:
        key, onset = (e["patient_id"], e["record_id"]), e["onset"]
        items = by_record[key]
        times = [t for t, _ in items]
        def interval(a, b):
            return [i for _, i in items[bisect.bisect_left(times, a):bisect.bisect_right(times, b)]]
        opportunities = interval(onset - target["horizon_seconds"], onset - target["gap_seconds"])
        positives = [i for i in opportunities if rows[i]["accepted"] and rows[i]["label"] == 1]
        observable += bool(positives)
        unique = [i for i in positives if sum(
            rows[i]["timestamp"] + target["gap_seconds"] <= other <= rows[i]["timestamp"] + target["horizon_seconds"]
            for other in event_records[key]) == 1]
        reason = ("no_cadence_opportunity" if not opportunities else "no_accepted_positive" if not positives
                  else "ambiguous_positive_event" if not unique else None)
        if reason is None:
            positive = min(unique, key=lambda i: (abs(rows[i]["timestamp"] - (onset - 360)), rows[i]["timestamp"]))
            negatives = [i for i in interval(onset - 3600, onset - 1800)
                         if rows[i]["accepted"] and rows[i]["label"] == 0 and i not in used
                         and rows[i]["timestamp"] <= rows[positive]["timestamp"] - target["history_seconds"]]
            if negatives:
                negative = min(negatives, key=lambda i: (abs(rows[i]["timestamp"] - (onset - 2700)), rows[i]["timestamp"]))
                used.add(negative)
                pairs.append(EventPair(_group(e), *key, float(onset), positive, negative))
                recurrence["first" if onset == event_records[key][0] else "recurrent"] += 1
            else:
                reason = "no_eligible_unused_negative"
        if reason:
            exclusions.append({**e, "reason": reason, "cadence_opportunities": len(opportunities),
                               "accepted_positives": len(positives),
                               "opportunity_label_reasons": dict(Counter(str(rows[i].get("label_reason")) for i in opportunities)),
                               "opportunity_rejection_reasons": dict(Counter(str(rows[i].get("reason")) for i in opportunities if not rows[i]["accepted"]))})
    people = Counter(p.group_id for p in pairs)
    positive_people = {_group(e) for e in ordered}
    report = dict(eligible_events=len(ordered), observable_events=observable, positive_people=len(positive_people),
                  paired_events=len(pairs), paired_people=len(people),
                  observable_event_fraction=len(pairs) / observable if observable else 0.,
                  positive_person_fraction=len(people) / len(positive_people) if positive_people else 0.,
                  pair_counts_by_person=dict(people), recurrence=dict(recurrence), exclusions=exclusions,
                  person_total_mass={g: 1 / len(people) for g in people})
    return pairs, report


def reference_intervals(annotations, daily_samples, fs):
    """Decode sample CSV coordinates directly; no production interval/label helpers."""
    if not daily_samples or any(int(n) != n or n <= 0 for n in daily_samples) or not math.isfinite(fs) or fs <= 0:
        raise ValueError("invalid independent sample metadata")
    offsets = [0]
    for n in daily_samples:
        offsets.append(offsets[-1] + n)
    af = []
    for r in annotations:
        span = []
        for side in ("start", "end"):
            day, sample = int(r[f"{side}_file_index"]), int(r[f"{side}_qrs_index"])
            if not 0 <= day < len(daily_samples) or not 0 <= sample <= daily_samples[day]:
                raise ValueError("annotation sample coordinate outside acquisition")
            span.append((offsets[day] + sample) / fs)
        if span[1] <= span[0]:
            raise ValueError("nonpositive annotation span")
        af.append(span)
    af.sort()
    if any(a[1] > b[0] for a, b in zip(af, af[1:])):
        raise ValueError("overlapping annotation spans")
    masks = [(start / fs, min(start + 6000, end) / fs) for start, end in zip(offsets, offsets[1:])]
    ends = sorted({0., offsets[-1] / fs, *(t for span in af + masks for t in span)})
    intervals = []
    for a, b in zip(ends, ends[1:]):
        midpoint = (a + b) / 2
        rhythm = ("unknown" if any(lo <= midpoint < hi for lo, hi in masks)
                  else "AF" if any(lo <= midpoint < hi for lo, hi in af) else "other")
        if intervals and intervals[-1][2] == rhythm:
            a = intervals.pop()[0]
        intervals.append((float(a), float(b), rhythm))
    return intervals


def _reference_covered(intervals, lo, hi, kinds):
    length = sum(max(0., min(hi, b) - max(lo, a)) for a, b, kind in intervals if kind in kinds)
    return math.isclose(length, hi - lo, abs_tol=1e-8, rel_tol=0.)


def reference_events(intervals, duration, config):
    target, output, last_end = config["target"], [], -math.inf
    for j, (a, b, kind) in enumerate(intervals):
        if kind != "AF":
            continue
        if j and intervals[j - 1][2] == "other" and intervals[j - 1][1] == a:
            first = max(intervals[j - 1][0] + target["history_seconds"], a - target["horizon_seconds"],
                        last_end + target["washout_seconds"])
            last = min(a - target["gap_seconds"], duration - target["horizon_seconds"])
            if (b - a >= target["min_episode_seconds"] and first <= last
                    and _reference_covered(intervals, first - target["history_seconds"], a, {"other"})
                    and _reference_covered(intervals, first, first + target["horizon_seconds"], {"other", "AF"})):
                output.append(a)
        last_end = b
    return output


def reference_label(t, intervals, duration, config, *, onsets=None):
    target = config["target"]
    af = [(a, b) for a, b, kind in intervals if kind == "AF"]
    if any(a <= t < b for a, b in af):
        return None, "current_AF"
    lo, hi = t - target["history_seconds"], t + target["horizon_seconds"]
    if lo < 0:
        return None, "history"
    if hi > duration:
        return None, "future"
    if any(b <= t < b + target["washout_seconds"] for _, b in af):
        return None, "washout"
    if not _reference_covered(intervals, lo, hi, {"other", "AF"}):
        return None, "unknown"
    if not _reference_covered(intervals, lo, t, {"other"}):
        return None, "history"
    if any(t < a < t + target["gap_seconds"] for a, _ in af):
        return None, "gap"
    upcoming = [a for a, _ in af if t + target["gap_seconds"] <= a <= hi]
    if upcoming:
        eligible = reference_events(intervals, duration, config) if onsets is None else onsets
        return (1, None) if any(a in eligible for a in upcoming) else (None, "ineligible_event")
    return 0, None


def audit_metadata(rows, events, index, metadata_root, config, *, deadline=math.inf):
    root = Path(metadata_root)
    allowed_records = {r["record"]["record_id"] for r in index["records"]}
    with (root / "metadata.csv").open(newline="") as handle:
        metadata = {r["record_id"]: r for r in csv.DictReader(handle) if r["record_id"] in allowed_records}
    continuity_path = root / "continuity-diagnostics.json"
    continuity = {r["record_id"]: r for r in json.loads(continuity_path.read_text())["records"]
                  if r["record_id"] in allowed_records} if continuity_path.exists() else {}
    by_record, event_map = defaultdict(list), defaultdict(list)
    for i, r in enumerate(rows):
        by_record[r["record_id"]].append(i)
    for e in events:
        event_map[e["record_id"]].append(float(e["onset"]))
    report = dict(status="passed", records=[], label_disagreements=[], event_disagreements=[],
                  cadence_disagreements=[], metadata_errors=[], incomplete_records=[], checked_rows=0,
                  reference_events=0, metadata_sha256={"metadata.csv": file_hash(root / "metadata.csv")})
    if continuity_path.exists():
        report["metadata_sha256"][continuity_path.name] = file_hash(continuity_path)
    with Progress("precursor.metadata", total=len(index["records"])) as progress:
        for n, item in enumerate(index["records"], 1):
            if time.monotonic() >= deadline:
                raise TimeoutError("metadata audit deadline exceeded")
            record = item["record"]
            rid, fs = record["record_id"], record["fs"]
            if Path(rid).name != rid or rid in {".", ".."}:
                raise ValueError("invalid allowed record directory")
            source = metadata.get(rid)
            counts = continuity.get(rid, {}).get("daily_samples")
            if counts is None and source and int(source["record_files"]) == 1 and len(record["signal_parts"]) == 1:
                counts = [int(source["record_samples"])]
            if counts is None:
                report["incomplete_records"].append(rid)
                continue
            if (not source or sum(counts) != int(source["record_samples"])
                    or len(counts) != int(source["record_files"])
                    or counts != [p["samples"] for p in record["signal_parts"]]
                    or "iridia-af:" + source["patient_id"] != record["patient_id"]
                    or fs != 200. or sum(counts) / fs != record["duration_seconds"]):
                report["metadata_errors"].append({"record_id": rid, "reason": "acquisition_index_disagreement"})
                continue
            path = root / "records" / rid / f"{rid}_ecg_labels.csv"
            checksum = file_hash(path)
            if checksum != item["input_contract"]["annotation_sha256"]:
                report["metadata_errors"].append({"record_id": rid, "reason": "annotation_checksum"})
                continue
            report["metadata_sha256"][str(path.relative_to(root))] = checksum
            with path.open(newline="") as handle:
                intervals = reference_intervals(list(csv.DictReader(handle)), counts, fs)
            onsets = reference_events(intervals, record["duration_seconds"], config)
            report["reference_events"] += len(onsets)
            if sorted(onsets) != sorted(event_map[rid]):
                report["event_disagreements"].append(dict(record_id=rid, reference=onsets, cached=event_map[rid]))
            indices = by_record[rid]
            expected_times = np.arange(0, record["duration_seconds"] + 1e-9, config["target"]["cadence_seconds"])
            if expected_times[-1] < record["duration_seconds"]:
                expected_times = np.append(expected_times, record["duration_seconds"])
            actual_times = [rows[i]["timestamp"] for i in indices]
            if len(actual_times) != len(expected_times) or not np.array_equal(actual_times, expected_times):
                report["cadence_disagreements"].append(dict(record_id=rid, expected=len(expected_times), actual=len(actual_times)))
            for i in indices:
                r = rows[i]
                expected, reason = reference_label(r["timestamp"], intervals, record["duration_seconds"], config, onsets=onsets)
                if expected != r["label"] or reason != r.get("label_reason"):
                    report["label_disagreements"].append(dict(record_id=rid, row_index=i, timestamp=r["timestamp"],
                                                             cached_label=r["label"], cached_reason=r.get("label_reason"),
                                                             expected_label=expected, expected_reason=reason))
                report["checked_rows"] += 1
            report["records"].append(dict(record_id=rid, days=len(counts), samples=sum(counts), eligible_events=len(onsets),
                                          annotation_sha256=checksum, first_sample_masks_seconds=30., prior_af_ends=[b for _, b, k in intervals if k == "AF"]))
            progress.update(completed=n)
    if any(report[k] for k in ("label_disagreements", "event_disagreements", "cadence_disagreements", "metadata_errors")):
        report["status"] = "failed"
    if report["incomplete_records"]:
        report["status"] = "independent_reference_incomplete"
    return report


def fresh_output(path):
    path = Path(path)
    if path.exists():
        raise ValueError(f"output already exists: {path}")
    path.mkdir(parents=True)
    return path


def source_manifest():
    root = Path(__file__).resolve().parents[2]
    paths = [*sorted((root / "src/cardiotwin").glob("*.py")), *sorted((root / "scripts").glob("*precursor*.py")),
             root / "scripts/kaggle_cycle.py", root / "pyproject.toml"]
    files = {str(p.relative_to(root)): file_hash(p) for p in paths}
    return {"files": files, "sha256": _digest(files)}


def pair_support(rows, events, roles, config):
    result, all_pairs, failures = [], [], []
    for fold in roles:
        entry = {"fold": fold["fold"], "roles": {}}
        for role in ROLE_NAMES:
            pairs, report = build_event_pairs(rows, events, fold[role], config)
            entry["roles"][role] = report
            all_pairs.extend({"fold": fold["fold"], "role": role, **asdict(p)} for p in pairs)
            if role == "fit" and (report["observable_event_fraction"] < .8 or report["positive_person_fraction"] < .8):
                failures.append(f"fold_{fold['fold']}_fit_pair_support")
            if role == "stop" and report["paired_people"] < 2:
                failures.append(f"fold_{fold['fold']}_stop_pair_people")
            if role in {"calibration", "policy"}:
                groups = set(fold[role])
                counts = Counter(r["label"] for r in rows if _group(r) in groups and r["accepted"] and r["label"] in (0, 1))
                report["natural_label_counts"] = dict(counts)
                if set(counts) != {0, 1}:
                    failures.append(f"fold_{fold['fold']}_{role}_classes")
        result.append(entry)
    return {"status": "passed" if not failures else "feasibility_failed", "failed_criteria": failures,
            "folds": result}, all_pairs


def score_pair_diagnostics(streams, rows, pairs, allowed_groups, *, deadline=math.inf):
    wanted = {(rows[i]["record_id"], rows[i]["timestamp"]): i
              for p in pairs for i in (p.positive_index, p.negative_index)}
    scores, people, folds = {}, {}, []
    for stream in streams:
        path = Path(stream["path"])
        if file_hash(path) != stream["sha256"]:
            raise ValueError("saved prediction checksum mismatch")
        count, labeled, positive, risk_sum, minimum, maximum = 0, 0, 0, 0., math.inf, -math.inf
        previous = {}
        with gzip.open(path, "rt") as handle, Progress("precursor.saved_scores", fold=stream["fold"]) as progress:
            for line in handle:
                r = json.loads(line)
                group = _group(r)
                if group not in allowed_groups:
                    raise ValueError("saved score outside development membership")
                rid, t = r["record_id"], r["timestamp"]
                if rid in previous and t <= previous[rid]:
                    raise ValueError("saved score duplicate or nonchronological row")
                previous[rid] = t
                count += 1
                if r["accepted"] and r["risk"] is not None:
                    risk = float(r["risk"])
                    if not math.isfinite(risk) or not 0 <= risk <= 1:
                        raise ValueError("nonfinite or unbounded saved probability")
                    if (rid, t) in wanted:
                        scores[wanted[(rid, t)]] = risk
                    if r["label"] in (0, 1):
                        labeled += 1
                        positive += r["label"]
                        risk_sum += risk
                        minimum, maximum = min(minimum, risk), max(maximum, risk)
                        cell = people.setdefault(group, dict(rows=0, positives=0, score_sum=0., score_square_sum=0.,
                                                              score_min=risk, score_max=risk))
                        cell["rows"] += 1
                        cell["positives"] += r["label"]
                        cell["score_sum"] += risk
                        cell["score_square_sum"] += risk * risk
                        cell["score_min"], cell["score_max"] = min(cell["score_min"], risk), max(cell["score_max"], risk)
                if count % 10000 == 0:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("saved score diagnostic deadline exceeded")
                    progress.update(completed=count)
        folds.append(dict(fold=stream["fold"], rows=count, labeled_rows=labeled, positive_rows=positive,
                          prevalence=positive / labeled if labeled else None,
                          mean_risk=risk_sum / labeled if labeled else None,
                          risk_min=minimum if labeled else None, risk_max=maximum if labeled else None,
                          sha256=stream["sha256"]))
    differences, by_person, pair_details = [], defaultdict(list), []
    for p in pairs:
        if p.positive_index not in scores or p.negative_index not in scores:
            raise ValueError("saved stream is missing a selected accepted pair row")
        d = scores[p.positive_index] - scores[p.negative_index]
        differences.append(d)
        by_person[p.group_id].append(d)
        pair_details.append({**asdict(p), "score_difference": d, "positive_risk": scores[p.positive_index],
                             "negative_risk": scores[p.negative_index]})
    for cell in people.values():
        cell.update(prevalence=cell["positives"] / cell["rows"], mean_risk=cell["score_sum"] / cell["rows"],
                    within_person_variance=max(0., cell["score_square_sum"] / cell["rows"] - (cell["score_sum"] / cell["rows"]) ** 2))
    def concordance(values):
        return float(np.mean([1. if d > 0 else .5 if d == 0 else 0. for d in values])) if values else None
    return dict(paired_events=len(pairs), paired_people=len(by_person), pair_concordance=concordance(differences),
                person_macro_concordance=float(np.mean([concordance(v) for v in by_person.values()])) if by_person else None,
                person_macro_difference=float(np.mean([np.mean(v) for v in by_person.values()])) if by_person else None,
                within_person_variance=float(np.mean([v["within_person_variance"] for v in people.values()])) if people else None,
                between_person_mean_variance=float(np.var([v["mean_risk"] for v in people.values()])) if people else None,
                raw_pair_analysis="unavailable_unsaved_logits", folds=folds, people=people, pair_details=pair_details)


def saved_diagnostics(rows, events, roles, index, config, *, deadline):
    root = Path(__file__).resolve().parents[2]
    base = root / "reports/ml-investigation/next-cycle"
    bindings = json.loads((base / "astra-research-20261011/source-evidence.json").read_text())["local_evidence_sha256"]
    pairs, report = build_event_pairs(rows, events, index["allowed_groups"], config)
    results = {"pair_support": report, "arms": {}}
    sources = [("waveform", "waveform-screen-20261011", "outputs/cardiotwin-waveform-screen/model", None),
               ("rr_patient_only", "recovery-review-20261011", "outputs/cardiotwin-checkpoint-cycle/patient_only", "patient_only"),
               ("rr_patient_class", "recovery-review-20261011", "outputs/cardiotwin-checkpoint-cycle/patient_class", "patient_class")]
    for name, directory, model_directory, arm in sources:
        review_path = base / directory / "review.json"
        if file_hash(review_path) != bindings[str(review_path.relative_to(root))]:
            raise ValueError("historical review binding changed")
        review = json.loads(review_path.read_text())
        reviewed_folds = [f for f in review["folds"] if arm is None or f["arm"] == arm]
        model_root = base / directory / model_directory
        trials = [json.loads(line) for line in (model_root / "experiments.jsonl").read_text().splitlines()]
        streams, misses = [], []
        for f in reviewed_folds:
            trial = next(t for t in trials if t["fold"] == f["fold"] and t["status"] == "complete")
            role = roles[f["fold"]]
            for key, saved_key in (("fit", "fit_groups"), ("stop", "early_stop_groups"), ("calibration", "calibration_groups"),
                                   ("policy", "policy_groups"), ("outer", "outer_groups")):
                if sorted(trial[saved_key]) != role[key]:
                    raise ValueError("original outer receipt role mismatch")
            streams.append(dict(path=model_root / Path(trial["predictions_artifact"]).name,
                                sha256=f["stream"]["gzip_byte_sha256"], fold=f["fold"]))
            for miss in trial["report"]["missed_event_details"]:
                onset = miss["onset"]
                prior = [w for w in trial["report"]["warning_details"]
                         if w["record_id"] == miss["record_id"] and w["timestamp"] < onset - config["target"]["horizon_seconds"]
                         and w["end_time"] > onset - config["target"]["horizon_seconds"]]
                misses.append({**miss, "prior_warning_active_in_target": bool(prior)})
        result = score_pair_diagnostics(streams, rows, pairs, set(index["allowed_groups"]), deadline=deadline)
        result.update(source_identity=review["source_identity"], review_sha256=file_hash(review_path),
                      missed_events=len(misses), missed_while_prior_warning_active=sum(m["prior_warning_active_in_target"] for m in misses),
                      missed_event_details=misses,
                      saved_raw_calibration_diagnostics=[dict(fold=t["fold"], **t["calibration_diagnostics"]) for t in trials])
        results["arms"][name] = result
    fields = sorted({k for p in pairs for k in rows[p.positive_index]["features"]
                     if k.startswith(("ecg_", "personal_", "baseline_", "quality_"))})
    results["feature_strata"] = {}
    for field in fields:
        differences, missing = defaultdict(list), 0
        for p in pairs:
            a, b = rows[p.positive_index]["features"].get(field), rows[p.negative_index]["features"].get(field)
            if a is None or b is None or not math.isfinite(a) or not math.isfinite(b):
                missing += 1
            else:
                differences[p.group_id].append(a - b)
        results["feature_strata"][field] = dict(pairs=sum(map(len, differences.values())), people=len(differences),
                                               missing_pairs=missing, person_macro_difference=float(np.mean([np.mean(v) for v in differences.values()])) if differences else None)
    return results


def compact_cache_binding(cache, *, expected=None):
    """Bind the mapping contract and every array consumed by CompactRows."""
    cache = Path(cache)
    index = json.loads((cache / "index.json").read_text())
    contract = json.loads((cache / "compact.json").read_text())
    schema = contract["features"]
    if (schema != index.get("feature_fields") or len(set(schema)) != len(schema)
            or contract["rows_sha256"] != index["rows_sha256"]):
        raise ValueError("compact schema/source differs from bound index")
    arrays = {}
    for name in ("features.npy", "metadata.npy"):
        checksum = file_hash(cache / name)
        if checksum != contract["checksums"][name]:
            raise ValueError(f"compact array checksum mismatch: {name}")
        values = np.load(cache / name, mmap_mode="r", allow_pickle=False)
        if len(values) != index["row_count"] or (name == "features.npy" and values.shape != (index["row_count"], len(schema))):
            raise ValueError("compact array shape mismatch")
        arrays[name] = dict(sha256=checksum, shape=list(values.shape), dtype=values.dtype.descr)
    binding = dict(version="compact-source-v1", index_sha256=file_hash(cache / "index.json"),
                   rows_sha256=index["rows_sha256"], compact_sha256=file_hash(cache / "compact.json"),
                   feature_schema=schema, arrays=arrays)
    binding = json.loads(json.dumps(binding))
    # Canonical JSON also normalizes numpy dtype descriptor tuples after receipt serialization.
    if expected is not None and _digest(binding) != _digest(expected):
        raise ValueError("compact mapping/array binding changed")
    return binding


def validate_compact_source(cache, rows, index, *, roles=(), deadline=float("inf")):
    """Stream the index-bound original rows; compare actual loader values by feature name."""
    cache = Path(cache)
    binding = compact_cache_binding(cache)
    if file_hash(cache / "rows.jsonl") != index["rows_sha256"]:
        raise ValueError("original source rows hash changed")
    schema = binding["feature_schema"]
    role_counts = {str(fold["fold"]): dict.fromkeys(ROLE_NAMES, 0) for fold in roles}
    role_sets = [{name: set(fold[name]) for name in ROLE_NAMES} for fold in roles]
    checked, missing = 0, 0
    with Progress("precursor.source_feature_validation", total=len(rows), device="cpu") as progress:
        with (cache / "rows.jsonl").open() as handle:
            for number, line in enumerate(handle):
                if number >= len(rows):
                    raise ValueError("original source has extra rows")
                original, loaded = json.loads(line), rows[number]
                expected_meta = {key: original.get(key) for key in loaded if key != "features"}
                if {key: loaded[key] for key in loaded if key != "features"} != expected_meta:
                    raise ValueError(f"source metadata/order mismatch at row {number}")
                if set(original["features"]) - set(schema):
                    raise ValueError(f"source feature schema mismatch at row {number}")
                for name in schema:
                    value, actual = original["features"].get(name), loaded["features"].get(name)
                    missing += value is None
                    if value != actual:
                        raise ValueError(f"source feature/missingness mismatch at row {number}: {name}")
                for fold, groups in zip(roles, role_sets, strict=True):
                    for role, people in groups.items():
                        role_counts[str(fold["fold"])][role] += _group(original) in people
                checked += 1
                if checked % 4096 == 0:
                    _check_deadline(deadline)
                    progress.update(completed=checked)
        if checked != len(rows) or checked != index["row_count"]:
            raise ValueError("original source row count mismatch")
        progress.update(completed=checked)
    if _digest(compact_cache_binding(cache)) != _digest(binding):
        raise ValueError("compact binding changed during source validation")
    return dict(status="passed", binding=binding, rows_checked=checked, feature_values_checked=checked*len(schema),
                missing_values_checked=missing, role_rows_checked=role_counts,
                historical_fit_time_identity_verified=False,
                scope="current actual loader equals index-bound original rows by name, identity/order, missingness and metadata; no historical fit-time identity recovery")


def run_diagnostic(cache, split_path, metadata_root, config, output, *, deadline):
    from .investigation import read_cache

    output = fresh_output(output)
    started = time.monotonic()
    receipt = dict(status="running", stage="audit", config_sha256=_digest(config), source=source_manifest())
    _write(output / "source-manifest.json", receipt["source"])
    root = Path(__file__).resolve().parents[2]
    for name in receipt["source"]["files"]:
        destination = output / "source-package" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, destination)
    _write(output / "predeclaration.json", dict(scope="original development only; metadata, pairs, saved scores; no fitting",
                                                seed=42, config=config, fixed_pairs=dict(positive_seconds=-360, negative_seconds=-2700,
                                                negative_interval=[-3600, -1800]), fit_support=.8, stop_people=2))
    _write(output / "receipt.json", receipt)
    try:
        cache = Path(cache)
        if not (cache / "compact.json").is_file():
            raise ValueError("read-only audit requires existing compact cache")
        with Progress("precursor.cache_integrity", total=1) as progress:
            rows, events, index = read_cache(cache)
            progress.update(completed=1)
        if _extraction_contract(config) != _extraction_contract(index["config"]):
            raise ValueError("extraction config changed")
        split = json.loads(Path(split_path).read_text())
        if (len(index["allowed_groups"]), len(index["records"]), len(events), len(rows)) != (121, 132, 241, 629444):
            raise ValueError("original development cohort counts changed")
        roles = make_roles(rows, events, index, split)
        compact_validation = validate_compact_source(cache, rows, index, roles=roles, deadline=deadline)
        _write(output / "compact-source-validation.json", compact_validation)
        receipt["compact_binding"] = compact_validation["binding"]
        root = Path(__file__).resolve().parents[2]
        prior = root / "reports/ml-investigation/next-cycle/gap-study-20261011/exposure-audit.json"
        expected = json.loads(prior.read_text())
        if expected["cache_sha256"] != index["sha256"] or any(
                {r: f[r] for r in ROLE_NAMES} != old["role_groups"] for f, old in zip(roles, expected["folds"])):
            raise ValueError("roles differ from original exposure audit")
        receipt.update(cache_index_sha256=file_hash(cache / "index.json"), cache_identity=index["sha256"],
                       split_sha256=file_hash(split_path), roles_sha256=_digest(roles), rows=len(rows), events=len(events), people=121)
        _write(output / "roles.json", dict(cache_index_sha256=receipt["cache_index_sha256"], roles=roles))
        metadata = audit_metadata(rows, events, index, metadata_root, config, deadline=deadline)
        _write(output / "metadata-audit.json", metadata)
        with Progress("precursor.pair_support", total=3) as progress:
            support, pairs = pair_support(rows, events, roles, config)
            progress.update(completed=3)
        _write(output / "pair-support.json", support)
        with (output / "pairs.jsonl").open("w") as handle:
            for pair in pairs:
                handle.write(json.dumps(pair, allow_nan=False) + "\n")
        receipt.update(pairs_sha256=file_hash(output / "pairs.jsonl"), metadata_status=metadata["status"],
                       pair_support_status=support["status"], failed_criteria=support["failed_criteria"][:])
        if metadata["status"] != "passed":
            receipt["failed_criteria"].append("independent_metadata_reference")
        receipt["status"] = "feasibility_failed" if receipt["failed_criteria"] else "passed"
        _write(output / "receipt.json", receipt)
        diagnostics = saved_diagnostics(rows, events, roles, index, config, deadline=deadline)
        _write(output / "saved-score-diagnostics.json", diagnostics)
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        _write(output / "receipt.json", receipt)
    return receipt


def pair_weights(pairs):
    if not pairs:
        raise ValueError("No fit pairs")
    counts = Counter(p.group_id for p in pairs)
    values = np.array([1 / (len(counts) * counts[p.group_id]) for p in pairs])
    return values / values.mean()


def _pair_matrices(rows, pairs, schema):
    return tuple(matrix([rows[getattr(p, side)]["features"] for p in pairs], schema)
                 for side in ("positive_index", "negative_index"))


def _fit_fixed_logistic(x, y, weights, seed):
    model = build_pipeline("logistic", {"C": .1}, seed)["model"]
    model.set_params(fit_intercept=False, class_weight=None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y, sample_weight=weights)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught) or not np.isfinite(model.coef_).all():
        raise ValueError("Fixed linear fit did not converge to finite coefficients")
    return model


def fit_linear_pair_arms(rows, pairs, schema, *, seed=42):
    if seed != 42 or list(schema) != LINEAR_SCHEMA:
        raise ValueError("Fixed seed/schema required for matched linear arms")
    weights = pair_weights(pairs)
    row_mass, negative_ids, event_ids = Counter(), set(), set()
    for p, weight in zip(pairs, weights):
        key = (p.record_id, p.onset)
        if p.negative_index in negative_ids or key in event_ids:
            raise ValueError("duplicate fit event or negative row")
        negative_ids.add(p.negative_index)
        event_ids.add(key)
        for position, label in ((p.positive_index, 1), (p.negative_index, 0)):
            r = rows[position]
            if (_group(r), r["patient_id"], r["record_id"]) != (p.group_id, p.patient_id, p.record_id) or not r["accepted"] or r["label"] != label:
                raise ValueError("pair source identity/label mismatch")
            if not set(schema) <= r["features"].keys():
                raise ValueError("required schema missing from fit cache")
            row_mass[position] += weight / 2
    indices = sorted(row_mass)
    x = matrix([rows[i]["features"] for i in indices], schema)
    template = build_pipeline("logistic", {"C": .1}, seed)
    imputer = template["impute"].fit(x)
    scaler = template["scale"].fit(imputer.transform(x), sample_weight=[row_mass[i] for i in indices])
    positive, negative = _pair_matrices(rows, pairs, schema)
    zp, zn = (scaler.transform(imputer.transform(v)) for v in (positive, negative))
    difference = zp - zn
    labels = np.r_[np.ones(len(pairs)), np.zeros(len(pairs))]
    repeated_weights = np.r_[weights, weights]
    arms, fitted = {}, {}
    for name, values in ((LINEAR_ARMS[0], np.vstack((zp, zn))), (LINEAR_ARMS[1], np.vstack((difference, -difference)))):
        model = _fit_fixed_logistic(values, labels, repeated_weights, seed)
        arms[name] = Pipeline([("impute", deepcopy(imputer)), ("scale", deepcopy(scaler)), ("model", model)])
        fitted[name] = dict(coefficient_norm=float(np.linalg.norm(model.coef_)), iterations=int(model.n_iter_[0]),
                            converged=True, training_rows=len(labels))
    return arms, dict(training_events=len(pairs), training_people=len({p.group_id for p in pairs}),
                      unique_fit_rows=len(indices), transformed_columns=x.shape[1], difference_rank=int(np.linalg.matrix_rank(difference)),
                      nonzero_variance_columns=int(np.count_nonzero(np.var(difference, axis=0))),
                      person_total_mass={g: float(sum(w for p, w in zip(pairs, weights) if p.group_id == g) / weights.sum()) for g in sorted({p.group_id for p in pairs})},
                      preprocessing_sha256=hashlib.sha256(pickle.dumps((imputer, scaler), protocol=5)).hexdigest(), arms=fitted)


def pair_metrics(pipeline, rows, pairs, schema):
    if not pairs:
        return dict(pairs=0, people=0, person_macro_concordance=None, person_macro_loss=None)
    positive, negative = _pair_matrices(rows, pairs, schema)
    difference = raw_score(pipeline, positive) - raw_score(pipeline, negative)
    groups = defaultdict(list)
    for p, value in zip(pairs, difference):
        groups[p.group_id].append(float(value))
    concordance = [float(np.mean([1. if v > 0 else .5 if v == 0 else 0. for v in values])) for values in groups.values()]
    loss = [float(np.mean(np.logaddexp(0., -np.array(values)))) for values in groups.values()]
    return dict(pairs=len(pairs), people=len(groups), person_macro_concordance=float(np.mean(concordance)),
                person_macro_loss=float(np.mean(loss)), person_macro_difference=float(np.mean([np.mean(v) for v in groups.values()])),
                person_differences=dict(groups))


def negative_drift_pairs(rows, groups):
    lookup = {(r["patient_id"], r["record_id"], r["timestamp"]): i for i, r in enumerate(rows)
              if _group(r) in groups and r["accepted"] and r["label"] == 0}
    pairs, used = [], set()
    for (patient, record, t), later in sorted(lookup.items()):
        earlier = lookup.get((patient, record, t - 2340))
        if t % 3600 == 0 and earlier is not None and later not in used and earlier not in used:
            pairs.append(EventPair(_group(rows[later]), patient, record, float(t), later, earlier))
            used.update((earlier, later))
    return pairs


def diagnostic_controls(rows, fit_pairs, stop_pairs, schema, fitted_arms, *, seed=42):
    rank = fitted_arms["within_person_rank"]
    positive, negative = _pair_matrices(rows, fit_pairs, schema)
    difference = rank[:2].transform(positive) - rank[:2].transform(negative)
    groups = sorted({p.group_id for p in fit_pairs})
    weights = np.r_[pair_weights(fit_pairs), pair_weights(fit_pairs)]
    labels = np.r_[np.ones(len(fit_pairs)), np.zeros(len(fit_pairs))]
    nulls = []
    for null_seed in range(101, 120):
        orientation = dict(zip(groups, np.random.default_rng(null_seed).choice([-1., 1.], len(groups))))
        null_d = difference * np.array([orientation[p.group_id] for p in fit_pairs])[:, None]
        model = _fit_fixed_logistic(np.vstack((null_d, -null_d)), labels, weights, seed)
        null = Pipeline([("impute", rank["impute"]), ("scale", rank["scale"]), ("model", model)])
        nulls.append(dict(seed=null_seed, **pair_metrics(null, rows, stop_pairs, schema)))
    return dict(scope="fit-only person orientation fits; stop-pair diagnostics only; no significance or model selection",
                stop_people=len({p.group_id for p in stop_pairs}), person_orientation_nulls=nulls,
                real={name: pair_metrics(model, rows, stop_pairs, schema) for name, model in fitted_arms.items()})


PRIMARY_POLICY = "far_0.25_occupancy_0.05"
PRESETS = [(far, occupancy) for far in (.1, .25, 1.) for occupancy in (.01, .05, .1)]


def _check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Precursor execution deadline exhausted")


def _save_package(output):
    manifest = source_manifest()
    root = Path(__file__).resolve().parents[2]
    _write(output / "source-manifest.json", manifest)
    for name in manifest["files"]:
        destination = output / "source-package" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, destination)
    return manifest


def _verify_package(directory, manifest):
    if _digest(manifest["files"]) != manifest["sha256"]:
        raise ValueError("source manifest binding mismatch")
    for name, expected in manifest["files"].items():
        path = directory / "source-package" / name
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"source package binding mismatch: {name}")


def validate_pair_receipts(rows, events, roles, config, directory):
    directory = Path(directory)
    support, generated = pair_support(rows, events, roles, config)
    if support["status"] != "passed" or _digest(support) != _digest(json.loads((directory / "pair-support.json").read_text())):
        raise ValueError("diagnostic pair support changed")
    saved = [json.loads(line) for line in (directory / "pairs.jsonl").read_text().splitlines()]
    if generated != saved:
        raise ValueError("diagnostic pair membership changed")
    return saved


def validate_diagnostic(cache, split_path, audit_path, config):
    """Validate saved audit evidence without repeating its metadata interpretation."""
    from .investigation import read_cache

    audit_path, cache = Path(audit_path), Path(cache)
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "passed" or audit.get("metadata_status") != "passed" or audit.get("pair_support_status") != "passed":
        raise ValueError("diagnostic must pass before fitting")
    if "compact_binding" not in audit:
        raise ValueError("diagnostic compact provenance incomplete; no fitting allowed")
    compact_cache_binding(cache, expected=audit["compact_binding"])
    directory = audit_path.parent
    _verify_package(directory, audit["source"])
    if file_hash(cache / "index.json") != audit["cache_index_sha256"] or file_hash(split_path) != audit["split_sha256"]:
        raise ValueError("diagnostic cache/split binding mismatch")
    if _digest(config) != audit["config_sha256"]:
        raise ValueError("diagnostic config binding mismatch")
    rows, events, index = read_cache(cache)
    if (len(index["allowed_groups"]), len(index["records"]), len(events), len(rows)) != (121, 132, 241, 629444):
        raise ValueError("original development cohort changed")
    if index["sha256"] != audit["cache_identity"] or _extraction_contract(config) != _extraction_contract(index["config"]):
        raise ValueError("diagnostic extraction/cache identity mismatch")
    roles = make_roles(rows, events, index, json.loads(Path(split_path).read_text()))
    saved_roles = json.loads((directory / "roles.json").read_text())
    if saved_roles != dict(cache_index_sha256=audit["cache_index_sha256"], roles=roles) or _digest(roles) != audit["roles_sha256"]:
        raise ValueError("diagnostic roles binding mismatch")
    if file_hash(directory / "pairs.jsonl") != audit["pairs_sha256"]:
        raise ValueError("diagnostic pair binding mismatch")
    saved_pairs = validate_pair_receipts(rows, events, roles, config, directory)
    metadata = json.loads((directory / "metadata-audit.json").read_text())
    # The first audit predates explicit input paths; its approved metadata root was data/iridia-af.
    metadata_root = Path(__file__).resolve().parents[2] / "data/iridia-af"
    if metadata["status"] != "passed":
        raise ValueError("diagnostic metadata reference failed")
    for name, checksum in metadata["metadata_sha256"].items():
        if file_hash(metadata_root / name) != checksum:
            raise ValueError("diagnostic metadata binding changed")
    current = source_manifest()["files"]
    for name in ("src/cardiotwin/data.py", "src/cardiotwin/features.py", "src/cardiotwin/labels.py"):
        if current[name] != audit["source"]["files"][name]:
            raise ValueError("audited extraction source changed")
    return rows, events, index, roles, saved_pairs, audit


def score_linear_rows(pipeline, calibrator, rows, schema, *, deadline):
    from .investigation import _compact_prediction

    result = []
    with Progress("precursor.score", total=len(rows), device="cpu") as progress:
        for start in range(0, len(rows), 4096):
            _check_deadline(deadline)
            batch = rows[start:start + 4096]
            scored = [dict(_compact_prediction(r), risk=None, raw_logit=None) for r in batch]
            available = [i for i, r in enumerate(batch) if r["accepted"]]
            if available:
                logits = raw_score(pipeline, matrix([batch[i]["features"] for i in available], schema))
                risks = calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]
                if not np.isfinite(logits).all() or not np.isfinite(risks).all():
                    raise ValueError("nonfinite linear predictions")
                for i, logit, risk in zip(available, logits, risks, strict=True):
                    scored[i].update(raw_logit=float(logit), risk=float(np.clip(risk, np.nextafter(0., 1.), np.nextafter(1., 0.))))
            result.extend(scored)
            progress.update(completed=min(start + 4096, len(rows)))
    return result


def freeze_policy_frontiers(rows, events, config, *, deadline):
    from .investigation import select_policy

    selected, searches, search = {}, {}, None
    with Progress("precursor.policy_frontiers", total=len(PRESETS) + 1, device="cpu") as progress:
        for n, (far, occupancy) in enumerate(PRESETS):
            name = f"far_{far:g}_occupancy_{occupancy:g}"
            contract = deepcopy(config)
            contract["alert"].update(max_burden_occupancy=occupancy, max_eligible_negative_burden_occupancy=occupancy)
            policy, candidates = select_policy(rows, events, contract, budget=far, search=search, deadline=deadline)
            if policy is None:
                raise ValueError("Finite frontier has no feasible policy, including no-warning")
            selected[name], searches[name] = policy, candidates
            if search is None:
                search = candidates
            progress.update(completed=n + 1, preset=name, force=True)
        legacy = deepcopy(config)
        for key in ("policy_version", "max_duration_seconds", "rearm_seconds", "max_burden_occupancy", "max_eligible_negative_burden_occupancy"):
            legacy["alert"].pop(key, None)
        policy, candidates = select_policy(rows, events, legacy, budget=.25, deadline=deadline)
        if policy is None:
            raise ValueError("Legacy bridge has no feasible inner policy")
        selected["legacy_bridge"], searches["legacy_bridge"] = policy, candidates
        progress.update(completed=len(PRESETS) + 1)
    return selected, searches


def _pairs_in(pairs, groups):
    return [p for p in pairs if p.group_id in set(groups)]


def pair_strata(rows, pairs, events, prior_af_ends, arms):
    result = []
    for pair in pairs:
        row = rows[pair.positive_index]
        previous = [end for end in prior_af_ends.get(pair.record_id, []) if end <= row["timestamp"]]
        result.append(dict(group_id=pair.group_id, record_id=pair.record_id, onset=pair.onset,
                           recurrence="recurrent" if any(e["record_id"] == pair.record_id and e["onset"] < pair.onset for e in events) else "first",
                           prior_af_distance_seconds=row["timestamp"]-max(previous) if previous else None,
                           prior_af_reference_available=pair.record_id in prior_af_ends,
                           record_phase=row["timestamp"]/row["duration_seconds"] if row.get("duration_seconds") else None,
                           missing_feature_count=sum(row["features"].get(k) is None for k in LINEAR_SCHEMA),
                           quality_baseline={k: v for k, v in row["features"].items() if k.startswith(("quality_", "baseline_"))},
                           scores={arm: pair_metrics(model, rows, [pair], LINEAR_SCHEMA) for arm, model in arms.items()}))
    return result


def _person_rows(report, rows):
    identities = {r["patient_id"]: _group(r) for r in rows}
    grouped = defaultdict(Counter)
    for item in report["per_patient"]:
        grouped[identities[item["patient_id"]]].update({k: v for k, v in item.items()
                                                       if k.endswith("_seconds") or k in ("eligible_events", "matched_events", "false_warnings", "warnings", "unassessable_warnings")})
    return [dict(group_id=group, **dict(counts)) for group, counts in sorted(grouped.items())]


def execute_linear_folds(rows, events, roles, pairs, config, output, identity, *, deadline):
    """Fixed objectives, then natural calibration and frozen policies before outer scoring."""
    import joblib

    from .evaluation import evaluate_stream, probability_metrics
    from .investigation import _subset
    from .train import _calibrate, calibration_diagnostics

    output = Path(output)
    metadata_path = output / "metadata-audit.json"
    prior_af_ends = {r["record_id"]: r["prior_af_ends"] for r in json.loads(metadata_path.read_text())["records"]} if metadata_path.exists() else {}
    folds = []
    for role in roles:
        fold = role["fold"]
        fit_pairs, stop_pairs = (_pairs_in(pairs, role[name]) for name in ("fit", "stop"))
        _check_deadline(deadline)
        with Progress("precursor.fit", total=2, fold=fold, device="cpu") as progress:
            arms, diagnostics = fit_linear_pair_arms(rows, fit_pairs, LINEAR_SCHEMA)
            progress.update(completed=2)
        controls = diagnostic_controls(rows, fit_pairs, stop_pairs, LINEAR_SCHEMA, arms)
        for name in ("fit", "stop"):
            drift = negative_drift_pairs(rows, set(role[name]))
            controls[name + "_negative_drift"] = {arm: pair_metrics(model, rows, drift, LINEAR_SCHEMA) for arm, model in arms.items()}
            controls[name + "_offline_pair_strata"] = pair_strata(rows, _pairs_in(pairs, role[name]), events, prior_af_ends, arms)
        _write(output / f"fold-{fold}-fit-diagnostics.json", dict(identity=identity, fit=diagnostics, controls=controls))
        cal_rows, policy_rows, stop_rows = (_subset(rows, role[name]) for name in ("calibration", "policy", "stop"))
        for arm, pipeline in arms.items():
            _check_deadline(deadline)
            prefix = f"fold-{fold}-{arm}"
            with Progress("precursor.calibration", total=1, fold=fold, arm=arm, device="cpu") as progress:
                calibrator = _calibrate(pipeline, LINEAR_SCHEMA, cal_rows, 42, calibration_c=100.)
                calibration = calibration_diagnostics(pipeline, LINEAR_SCHEMA, calibrator, cal_rows)
                progress.update(completed=1)
            stop_predictions = score_linear_rows(pipeline, calibrator, stop_rows, LINEAR_SCHEMA, deadline=deadline)
            stop_metrics = dict(pair=pair_metrics(pipeline, rows, stop_pairs, LINEAR_SCHEMA),
                                natural=probability_metrics(stop_predictions))
            policy_predictions = score_linear_rows(pipeline, calibrator, policy_rows, LINEAR_SCHEMA, deadline=deadline)
            policy_events = [e for e in events if _group(e) in set(role["policy"])]
            policies, searches = freeze_policy_frontiers(policy_predictions, policy_events, config, deadline=deadline)
            freeze_name = prefix + "-policy-freeze.json"
            _write(output / freeze_name, dict(identity=identity, fold=fold, arm=arm, roles=role,
                                              calibration=calibration, policies=policies, searches=searches,
                                              frozen_before_outer_scoring=True))
            joblib.dump(dict(pipeline=pipeline, calibrator=calibrator, schema=LINEAR_SCHEMA, identity=identity,
                             config=config), output / (prefix + "-model.joblib"))
            outer_rows = _subset(rows, role["outer"])
            outer_events = [e for e in events if _group(e) in set(role["outer"])]
            predictions = score_linear_rows(pipeline, calibrator, outer_rows, LINEAR_SCHEMA, deadline=deadline)
            predictions_name = prefix + "-predictions.jsonl.gz"
            with gzip.open(output / predictions_name, "wt") as handle:
                for r in predictions:
                    handle.write(json.dumps(r, allow_nan=False) + "\n")
            reports = {}
            with Progress("precursor.outer_replay", total=len(policies), fold=fold, arm=arm, device="cpu") as progress:
                for n, (name, policy) in enumerate(policies.items(), 1):
                    _check_deadline(deadline)
                    report = evaluate_stream(predictions, outer_events, policy, bootstrap_samples=0)
                    report["per_group"] = _person_rows(report, predictions)
                    reports[name] = report
                    progress.update(completed=n, preset=name, force=True)
            entry = dict(status="complete", fold=fold, arm=arm, identity=identity, policies=policies,
                         policy_freeze=freeze_name, policy_freeze_sha256=file_hash(output / freeze_name),
                         predictions=predictions_name, predictions_sha256=file_hash(output / predictions_name),
                         model=prefix + "-model.joblib", model_sha256=file_hash(output / (prefix + "-model.joblib")),
                         outer_rows=len(predictions), outer_events=len(outer_events), outer_groups=role["outer"],
                         calibration=calibration, stop=stop_metrics, reports=reports)
            _write(output / (prefix + "-receipt.json"), entry)
            folds.append(entry)
            _write(output / "checkpoint.json", dict(status="incomplete", completed_arms=len(folds), total_arms=6,
                                                     identity=identity, folds=folds))
    return dict(status="complete", identity=identity, folds=folds)


def run_linear_comparison(cache, split_path, audit_path, config, output, *, deadline):
    from .modeling import dependency_versions

    output = fresh_output(output)
    started = time.monotonic()
    receipt = dict(status="running", stage="linear", real_fits_started=0, remote_jobs_submitted=0)
    _write(output / "receipt.json", receipt)
    try:
        rows, events, index, roles, saved_pairs, audit = validate_diagnostic(cache, split_path, audit_path, config)
        _check_deadline(deadline)
        source = _save_package(output)
        for name in ("roles.json", "pairs.jsonl", "metadata-audit.json"):
            shutil.copyfile(Path(audit_path).parent / name, output / name)
        shutil.copyfile(audit_path, output / "diagnostic-receipt.json")
        unique_pairs = {tuple(p[k] for k in EventPair.__dataclass_fields__): EventPair(**{k: p[k] for k in EventPair.__dataclass_fields__})
                        for p in saved_pairs}
        pairs = list(unique_pairs.values())
        identity = dict(source_sha256=source["sha256"], cache_index_sha256=audit["cache_index_sha256"],
                        cache_identity=index["sha256"], roles_sha256=audit["roles_sha256"], pairs_sha256=audit["pairs_sha256"],
                        config_sha256=_digest(config), split_sha256=audit["split_sha256"], diagnostic_sha256=file_hash(audit_path))
        identity["compact_binding"] = audit["compact_binding"]
        declaration = dict(identity=identity, config=config, arms=list(LINEAR_ARMS), schema=LINEAR_SCHEMA,
                           seed=42, C=.1, max_iter=3000, fit_intercept=False, calibration_c=100., null_seeds=list(range(101, 120)),
                           primary_policy=PRIMARY_POLICY, presets=PRESETS, prior_development_reuse=True,
                           execution_location="local_cpu", inputs=dict(cache=str(Path(cache).resolve()), split=str(Path(split_path).resolve())),
                           output_probability_rule="clip to open (0,1); threshold 1 starts no warnings",
                           advancement="prespecified spec: complete/source-bound, positive pooled and >=2 folds, slopes>0, all outer constraints; supported requires CI and influence/preset direction",
                           failure_criteria=["provenance", "pair_support", "nonconvergence", "nonfinite", "time_or_storage_cap", "incomplete_folds"],
                           environment=dependency_versions())
        _write(output / "predeclaration.json", declaration)
        receipt.update(identity=identity, predeclaration_sha256=file_hash(output / "predeclaration.json"), real_fits_started=1)
        _write(output / "receipt.json", receipt)
        result = execute_linear_folds(rows, events, roles, pairs, config, output, identity, deadline=deadline)
        _write(output / "summary.json", result)
        receipt.update(status="complete", completed_arms=len(result["folds"]), rows=len(rows), events=len(events), people=len(index["allowed_groups"]),
                       artifacts={str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name != "receipt.json"})
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__, original_error=str(error))
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        _write(output / "receipt.json", receipt)
    return receipt


def compare_person_counts(baseline, candidate, *, seed=42, samples=10000):
    """Paired whole-group bootstrap of fixed-model count contributions, without row resampling."""
    ids = sorted(p["group_id"] for p in baseline)
    a, b = ({p["group_id"]: p for p in people} for people in (baseline, candidate))
    if len(a) != len(ids) or set(a) != set(b):
        raise ValueError("paired independent group membership differs")
    fields = ("eligible_events", "matched_events", "false_warnings", "eligible_seconds", "monitored_seconds", "burden_seconds", "eligible_negative_seconds", "eligible_negative_burden_seconds", "available_seconds", "warning_seconds")
    x, y = (np.array([[people[g].get(k, 0) for k in fields] for g in ids], dtype=float) for people in (a, b))
    if not np.array_equal(x[:, [0, 3, 4, 6, 8]], y[:, [0, 3, 4, 6, 8]]):
        raise ValueError("paired evaluation denominators/coverage differ")

    def rates(v):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.stack((v[..., 1]/v[..., 0], v[..., 2]/(v[..., 3]/3600),
                             v[..., 5]/v[..., 4], v[..., 7]/v[..., 6], v[..., 8]/v[..., 4], v[..., 9]/v[..., 4]), axis=-1)

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ids), size=(samples, len(ids)))
    sa, sb = x[draws].sum(axis=1), y[draws].sum(axis=1)
    ra, rb = rates(sa), rates(sb)
    metrics = ("event_sensitivity", "false_warnings_per_hour", "burden_occupancy", "eligible_negative_burden_occupancy", "forecast_coverage", "active_warning_occupancy")
    intervals = {}
    for n, key in enumerate(metrics):
        for name, values in (("baseline_" + key, ra[:, n]), ("candidate_" + key, rb[:, n]), (key + "_delta", rb[:, n] - ra[:, n])):
            valid = values[np.isfinite(values)]
            low, high = np.percentile(valid, [2.5, 97.5]) if len(valid) else (None, None)
            intervals[name] = dict(low=float(low) if low is not None else None, high=float(high) if high is not None else None,
                                   valid_draws=len(valid), invalid_draws=samples-len(valid))
    pooled_a, pooled_b = rates(x.sum(axis=0)), rates(y.sum(axis=0))
    influence = []
    for n, group in enumerate(ids):
        delta = rates(y.sum(axis=0)-y[n])[0] - rates(x.sum(axis=0)-x[n])[0]
        influence.append(dict(group_id=group, sensitivity_delta=float(delta) if np.isfinite(delta) else None))
    return dict(people=len(ids), negative_only_people=int((x[:, 0] == 0).sum()), samples=samples, seed=seed,
                sensitivity_delta=float(pooled_b[0]-pooled_a[0]) if np.isfinite(pooled_b[0]-pooled_a[0]) else None,
                baseline=dict(zip(metrics, [float(v) if np.isfinite(v) else None for v in pooled_a])),
                candidate=dict(zip(metrics, [float(v) if np.isfinite(v) else None for v in pooled_b])),
                event_denominator=int(x[:, 0].sum()), invalid_draws=int((~np.isfinite(rb[:, 0]-ra[:, 0])).sum()),
                degenerate_zero_delta_draws=int(np.count_nonzero(rb[:, 0]-ra[:, 0] == 0)), bootstrap=intervals,
                leave_one_person_out=influence, people_contributions=[dict(group_id=g, baseline=a[g], candidate=b[g]) for g in ids],
                limitation="Descriptive fixed-model development bootstrap; excludes retraining and selection uncertainty.")


def _same_numeric_json(first, second):
    if isinstance(first, dict):
        return isinstance(second, dict) and first.keys() == second.keys() and all(_same_numeric_json(v, second[k]) for k, v in first.items())
    if isinstance(first, list):
        return isinstance(second, list) and len(first) == len(second) and all(_same_numeric_json(a, b) for a, b in zip(first, second))
    if isinstance(first, (int, float)) and not isinstance(first, bool):
        return isinstance(second, (int, float)) and math.isclose(first, second, rel_tol=1e-11, abs_tol=1e-12)
    return first == second


def verify_linear_artifact(run_dir, entry, predictions, outer_rows, cal_rows, stop_rows, config):
    """Load only this run's hash-bound trusted model, and independently rescore without fitting."""
    import joblib
    from scipy.special import expit

    from .evaluation import probability_metrics
    from .train import calibration_diagnostics

    path = Path(run_dir) / entry["model"]
    if file_hash(path) != entry["model_sha256"]:
        raise ValueError("model artifact binding mismatch before loading")
    package = joblib.load(path)
    if package["identity"] != entry["identity"] or package["schema"] != LINEAR_SCHEMA or package["config"] != config:
        raise ValueError("model embedded recipe/schema/identity mismatch")
    pipeline, calibrator = package["pipeline"], package["calibrator"]
    recipe = dict(C=.1, solver="lbfgs", max_iter=3000, fit_intercept=False, class_weight=None)
    if any(pipeline["model"].get_params()[key] != value for key, value in recipe.items()) or calibrator["sigmoid"].C != 100.:
        raise ValueError("model fixed recipe mismatch")
    fresh = score_linear_rows(pipeline, calibrator, outer_rows, LINEAR_SCHEMA, deadline=math.inf)
    if len(fresh) != len(predictions):
        raise ValueError("model replay row count differs")
    for expected, saved in zip(fresh, predictions, strict=True):
        for key in ("raw_logit", "risk"):
            if not _same_numeric_json(expected[key], saved.get(key)):
                raise ValueError(f"model replay {key} mismatch")
    calibration = calibration_diagnostics(pipeline, LINEAR_SCHEMA, calibrator, cal_rows)
    if not _same_numeric_json(calibration, entry["calibration"]):
        raise ValueError("independent calibration replay differs")
    stop = score_linear_rows(pipeline, calibrator, stop_rows, LINEAR_SCHEMA, deadline=math.inf)
    raw_stop = [dict(r, risk=float(expit(r["raw_logit"])) if r["raw_logit"] is not None else None) for r in stop]
    accepted = [r for r in fresh if r["risk"] is not None]
    return dict(status="passed", rows_checked=len(fresh), available_logits_checked=len(accepted),
                unavailable_rows_checked=len(fresh)-len(accepted), model_sha256=file_hash(path),
                calibration=calibration, calibration_positive_people=len({_group(r) for r in cal_rows if r["accepted"] and r["label"] == 1}),
                stop_natural_raw=probability_metrics(raw_stop), stop_natural_calibrated=probability_metrics(stop),
                outer_quantiles={key: {str(q): float(np.quantile([r[key] for r in accepted], q)) if accepted else None for q in (0., .01, .25, .5, .75, .99, 1.)}
                                 for key in ("raw_logit", "risk")},
                preprocessing_sha256=_digest({name: {key: np.asarray(getattr(pipeline[name], key)).tolist() for key in fields}
                                               for name, fields in (("impute", ("statistics_",)), ("scale", ("mean_", "scale_", "var_")))}))


def review_comparison(run_dir, output, *, seed=42, bootstrap_samples=10000, retrospective_provenance=False):
    from .evaluation import evaluate_stream, policy_constraints
    from .investigation import _compact_prediction, _subset, read_cache

    run_dir = Path(run_dir)
    receipt = json.loads((run_dir / "receipt.json").read_text())
    if receipt.get("status") != "complete" or receipt.get("completed_arms") != 6:
        raise ValueError("Review requires complete three-fold/two-arm receipt")
    actual_artifacts = {str(p.relative_to(run_dir)) for p in run_dir.rglob("*") if p.is_file() and p.name != "receipt.json"}
    if set(receipt.get("artifacts", {})) != actual_artifacts:
        raise ValueError("artifact binding manifest is incomplete")
    for name, checksum in receipt.get("artifacts", {}).items():
        if not (run_dir / name).is_file() or file_hash(run_dir / name) != checksum:
            raise ValueError(f"artifact binding mismatch: {name}")
    source = json.loads((run_dir / "source-manifest.json").read_text())
    _verify_package(run_dir, source)
    for name in ("src/cardiotwin/alerts.py", "src/cardiotwin/evaluation.py"):
        if source_manifest()["files"][name] != source["files"][name]:
            raise ValueError("Review evaluator source differs from fitted package")
    declaration = json.loads((run_dir / "predeclaration.json").read_text())
    identity = declaration["identity"]
    if "compact_binding" not in identity and not retrospective_provenance:
        raise ValueError("Historical compact provenance incomplete; explicit retrospective evaluation required")
    roles = json.loads((run_dir / "roles.json").read_text())["roles"]
    if (identity != receipt["identity"] or source["sha256"] != identity["source_sha256"]
            or _digest(roles) != identity["roles_sha256"] or file_hash(run_dir / "pairs.jsonl") != identity["pairs_sha256"]
            or _digest(declaration["config"]) != identity["config_sha256"]):
        raise ValueError("run identity binding mismatch")
    summary = json.loads((run_dir / "summary.json").read_text())
    entries = summary["folds"]
    if (len(entries) != 6 or {(e["fold"], e["arm"]) for e in entries} != {(f, arm) for f in range(3) for arm in LINEAR_ARMS}
            or any(e["status"] != "complete" or e["identity"] != identity for e in entries)):
        raise ValueError("incomplete or mismatched fold recipes")
    output = fresh_output(output)
    review_source = _save_package(output)
    report = dict(status="running", identity=identity, review_source_sha256=review_source["sha256"], run_receipt_sha256=file_hash(run_dir / "receipt.json"), folds=[])
    report.update(experiment_eligibility="source_bound" if "compact_binding" in identity else "provenance_incomplete",
                  historical_fit_time_identity_verified="compact_binding" in identity)
    correction = None
    if "compact_binding" not in identity:
        report["evaluation_scope"] = "retrospective current-source-consistent saved-model evaluation only"
        correction = dict(status="append_only_correction", original_run_receipt_sha256=file_hash(run_dir / "receipt.json"),
                          original_fit_source_sha256=identity["source_sha256"], experiment_eligibility="provenance_incomplete",
                          historical_fit_time_identity_verified=False, current_source_validation_status="pending",
                          ruling="Posthoc source consistency cannot recover missing fit-time compact identity. No refit or eligibility upgrade; no GPU advancement.")
        _write(output / "provenance-correction.json", correction)
    _write(output / "receipt.json", report)
    try:
        cache = declaration["inputs"]["cache"]
        rows, events, index = read_cache(cache)
        if (file_hash(Path(cache) / "index.json") != identity["cache_index_sha256"]
                or index["sha256"] != identity["cache_identity"]):
            raise ValueError("review cache binding changed")
        if "compact_binding" in identity:
            compact_cache_binding(cache, expected=identity["compact_binding"])
        else:
            validation = validate_compact_source(cache, rows, index, roles=roles)
            _write(output / "retrospective-source-validation.json", validation)
            correction.update(current_source_validation_status="passed", current_source_validation=validation)
            _write(output / "provenance-correction.json", correction)
        per_arm = {arm: {name: [] for name in entries[0]["policies"]} for arm in LINEAR_ARMS}
        membership = defaultdict(set)
        for entry in entries:
            fold, arm = entry["fold"], entry["arm"]
            expected = _subset(rows, roles[fold]["outer"])
            with gzip.open(run_dir / entry["predictions"], "rt") as handle:
                predictions = [json.loads(line) for line in handle]
            if len(predictions) != len(expected) or len(predictions) != entry["outer_rows"]:
                raise ValueError("complete row membership differs")
            for r, original in zip(predictions, expected, strict=True):
                if {k: r.get(k) for k in original if k != "features"} != _compact_prediction(original):
                    raise ValueError("chronological row identity/label/availability differs")
                if original["accepted"] and (r["raw_logit"] is None or not math.isfinite(r["raw_logit"]) or not 0 < r["risk"] < 1):
                    raise ValueError("invalid available prediction")
                membership[arm].add((_group(r), r["record_id"], r["timestamp"]))
            outer_events = [e for e in events if _group(e) in set(roles[fold]["outer"])]
            if len(outer_events) != entry["outer_events"]:
                raise ValueError("complete event membership differs")
            freeze = json.loads((run_dir / entry["policy_freeze"]).read_text())
            if freeze["policies"] != entry["policies"] or not freeze["frozen_before_outer_scoring"]:
                raise ValueError("outer policy freeze mismatch")
            verification = verify_linear_artifact(run_dir, entry, predictions, expected,
                                                  _subset(rows, roles[fold]["calibration"]), _subset(rows, roles[fold]["stop"]), declaration["config"])
            verification["calibration_eligible_events"] = sum(_group(e) in set(roles[fold]["calibration"]) for e in events)
            recomputed = {}
            with Progress("precursor.review_replay", total=len(entry["policies"]), arm=arm, fold=fold, device="cpu") as progress:
                for n, (name, policy) in enumerate(entry["policies"].items(), 1):
                    result = evaluate_stream(predictions, outer_events, policy, bootstrap_samples=0)
                    result["per_group"] = _person_rows(result, predictions)
                    if result != entry["reports"][name]:
                        raise ValueError(f"outer replay differs: {fold}/{arm}/{name}")
                    recomputed[name] = result
                    per_arm[arm][name].extend(result["per_group"])
                    progress.update(completed=n, force=True, preset=name)
            report["folds"].append(dict(fold=fold, arm=arm, calibration=verification["calibration"], model_verification=verification, reports=recomputed,
                                        primary_constraints=policy_constraints(recomputed[PRIMARY_POLICY], entry["policies"][PRIMARY_POLICY], budget=.25)))
        if any(len(membership[arm]) != len(rows) for arm in LINEAR_ARMS) or membership[LINEAR_ARMS[0]] != membership[LINEAR_ARMS[1]]:
            raise ValueError("full cohort chronological coverage differs")
        for fold in range(3):
            signatures = {e["model_verification"]["preprocessing_sha256"] for e in report["folds"] if e["fold"] == fold}
            if len(signatures) != 1:
                raise ValueError("matched arms preprocessing differs")
        comparisons = {name: compare_person_counts(per_arm[LINEAR_ARMS[0]][name], per_arm[LINEAR_ARMS[1]][name], seed=seed, samples=bootstrap_samples)
                       for name in entries[0]["policies"]}
        primary = comparisons[PRIMARY_POLICY]
        fold_deltas = []
        for fold in range(3):
            by_arm = {e["arm"]: e for e in report["folds"] if e["fold"] == fold}
            fold_deltas.append(by_arm[LINEAR_ARMS[1]]["reports"][PRIMARY_POLICY]["event_sensitivity"] - by_arm[LINEAR_ARMS[0]]["reports"][PRIMARY_POLICY]["event_sensitivity"])
        candidate = [e for e in report["folds"] if e["arm"] == LINEAR_ARMS[1]]
        failures = []
        if primary["sensitivity_delta"] is None or primary["sensitivity_delta"] <= 0:
            failures.append("nonpositive_pooled_sensitivity_delta")
        if sum(d > 0 for d in fold_deltas) < 2:
            failures.append("fewer_than_two_improved_folds")
        if any(e["calibration"]["raw_score_slope"] <= 0 for e in candidate):
            failures.append("nonpositive_calibration_slope")
        if any(not e["primary_constraints"]["passed"] for e in candidate):
            failures.append("outer_primary_constraints")
        if "compact_binding" not in identity:
            failures.append("historical_fit_time_compact_provenance_incomplete")
        promising = not failures
        direction = all(p["sensitivity_delta"] is not None and p["sensitivity_delta"] >= 0 for p in primary["leave_one_person_out"]) and all(
            comparisons[f"far_0.25_occupancy_{occupancy:g}"]["sensitivity_delta"] is not None and comparisons[f"far_0.25_occupancy_{occupancy:g}"]["sensitivity_delta"] > 0 for occupancy in (.01, .05, .1))
        low = primary["bootstrap"]["event_sensitivity_delta"]["low"]
        supported = promising and low is not None and low > 0 and direction
        decision = "advance" if supported else "uncertain" if promising else "reject"
        report.update(status="complete", decision=decision, promising_development_lead=promising, supported_development_improvement=supported,
                      failed_advancement_criteria=failures, fold_sensitivity_deltas=fold_deltas, comparisons=comparisons,
                      rows=len(rows), events=len(events), people=len(index["allowed_groups"]), coverage_unchanged=True,
                      direction_survives_person_and_preset_checks=direction,
                      neural_eligible=False, neural_reason="No automatic GPU queue; a promising lead still requires measurement and actual T4 cost gates.",
                      clinical_claim=False, confirmation="Repeatedly examined development cohort; fresh independent confirmation remains required.")
        _write(output / "review.json", report)
        text = f"# Precursor linear comparison\n\nDecision: **{decision}**.\n\nMatched ranking sensitivity difference: {primary['sensitivity_delta']}; paired 95% interval [{low}, {primary['bootstrap']['event_sensitivity_delta']['high']}].\n\nFold differences: {fold_deltas}. Failed advancement criteria: {failures}.\n\nAll {len(rows)} chronological rows, {len(events)} events and {len(index['allowed_groups'])} development people retained. No held-out cohort access or clinical claim. Whole-person bootstrap conditions on fitted models and prior development reuse.\n\n"
        if "compact_binding" not in identity:
            text += "Historical experiment eligibility: **provenance_incomplete**. Fit-time compact mapping was not bound. Current actual loader features were independently checked against every index-bound original source row by name, missingness, row identity/order and metadata; saved models/calibrators were replayed without fitting. This supports current-source-consistent saved-model evaluation, and cannot verify the historical training mapping or upgrade eligibility.\n\n"
        text += "| Arm | Events caught / eligible | Assessed FAR / h | Active + unresolved burden | Eligible-negative burden |\n|---|---:|---:|---:|---:|\n"
        for arm, key in zip(LINEAR_ARMS, ("baseline", "candidate"), strict=True):
            entries_for_arm = [e for e in report["folds"] if e["arm"] == arm]
            caught = sum(e["reports"][PRIMARY_POLICY]["matched_events"] for e in entries_for_arm)
            rates = primary[key]
            text += f"| {arm} | {caught} / {len(events)} | {rates['false_warnings_per_hour']:.8f} | {rates['burden_occupancy']:.8%} | {rates['eligible_negative_burden_occupancy']:.8%} |\n"
        text += "\nIndependent hash-bound model verification (no fitting):\n\n"
        for arm in LINEAR_ARMS:
            checks = [e["model_verification"] for e in report["folds"] if e["arm"] == arm]
            text += f"- {arm}: {sum(c['rows_checked'] for c in checks)} rows; {sum(c['available_logits_checked'] for c in checks)} available logits/risks; {sum(c['unavailable_rows_checked'] for c in checks)} unavailable rows. Raw calibration slopes: {[c['calibration']['raw_score_slope'] for c in checks]}.\n"
        text += "\nGPU work remains conditional on the original measurement/runtime gates; a negative linear comparison alone does not authorize it.\n"
        (output / "review.md").write_text(text)
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, original_error=str(error))
        if correction is not None:
            correction.update(review_status="failed", error_type=type(error).__name__, original_error=str(error))
            if correction["current_source_validation_status"] == "pending":
                correction["current_source_validation_status"] = "failed"
            _write(output / "provenance-correction.json", correction)
        raise
    finally:
        _write(output / "receipt.json", dict(status=report["status"], decision=report.get("decision"), identity=identity,
                                             experiment_eligibility=report.get("experiment_eligibility", "source_bound"),
                                             historical_fit_time_identity_verified=report.get("historical_fit_time_identity_verified", True),
                                             review_source_sha256=review_source["sha256"], original_error=report.get("original_error"), artifacts={str(p.relative_to(output)): file_hash(p) for p in output.rglob("*") if p.is_file() and p.name != "receipt.json"}))
    return report
