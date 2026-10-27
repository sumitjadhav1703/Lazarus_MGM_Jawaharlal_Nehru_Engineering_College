"""Source-shaped generated fixed report metadata; cost evidence only."""

import copy
import hashlib
import inspect
import json
import time
from types import SimpleNamespace

import numpy as np

# Metadata-only upper bounds from native-metadata-counts.json, ordered fit/stop by fold.
DRIFT_PAIR_BOUNDS = ((2453, 332), (2067, 331), (2128, 222))
IDENTITY_ROW_BOUNDS = ((295458, 39920), (249036, 39821), (256493, 26706))


def _digest(function):
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()


def _artifact(path):
    return dict(path=path, sha256="a" * 64, bytes=131_072)


def _generated_control_context(people, pair_count):
    pairs = [SimpleNamespace(positive_index=i, negative_index=i + pair_count,
                             group_id=f"person-{i % people:03d}") for i in range(pair_count)]
    rows = [dict(group_id=f"person-{i % people:03d}") for i in range(2 * pair_count)]
    tables = SimpleNamespace(rows=np.array([(True,)] * len(rows), dtype=[("available", bool)]))
    prepared = SimpleNamespace(features=None, tables=tables, rows=rows)

    class GeneratedHead:
        def logits(self, features, tables, indices):
            selected = np.asarray(indices, dtype=float)
            return np.sin(selected * 1.731) + np.cos(selected * .371) / 7

    return GeneratedHead(), prepared, pairs


def _generated_people(events):
    people = []
    for i in range(121):
        event_count = events // 121 + (i < events % 121)
        people.append(dict(group_id=f"person-{i:03d}", eligible_events=event_count,
                           matched_events=0, false_warnings=0, eligible_seconds=3600.0,
                           monitored_seconds=3600.0, burden_seconds=0.0,
                           eligible_negative_seconds=3600.0,
                           eligible_negative_burden_seconds=0.0, available_seconds=3600.0,
                           warning_seconds=0.0))
    return people


def generated_fixed(*, config, fold_people, fold_events, study, guard):
    """Build complete fixed fields without fits, selectors, raw data, or outcome claims."""
    if (len(fold_people) != 3 or len(fold_events) != 3 or sum(fold_people) != 121
            or sum(fold_events) != 241 or any(type(v) is not int or v < 0 for v in (*fold_people, *fold_events))):
        raise ValueError("Original three-fold people/event cardinalities required")
    if len(study.POLICY_NAMES) != 10 or len(study.STUDY_ARMS) != 2:
        raise ValueError("Frozen arm and policy cardinalities differ")
    guard()
    identity = dict(preparation_sha256="a" * 64, support_sha256="b" * 64,
                    source_sha256={name: "c" * 64 for name in study.REQUIRED_SOURCE},
                    files_sha256={"generated-source-input": "d" * 64},
                    compact_binding={"generated_measurement_only": "e" * 64}, seed=42,
                    arms=list(study.STUDY_ARMS))
    if len(json.dumps(identity, sort_keys=True, allow_nan=False).encode()) > 131_072:
        raise ValueError("Generated identity exceeds prospective authority envelope")
    policies = {}
    for name in study.POLICY_NAMES:
        policy = copy.deepcopy(config)
        if name == "legacy_bridge":
            for key in ("policy_version", "max_duration_seconds", "rearm_seconds",
                        "max_burden_occupancy", "max_eligible_negative_burden_occupancy"):
                policy["alert"].pop(key, None)
        else:
            occupancy = float(name.rsplit("_", 1)[-1])
            policy["alert"].update(max_burden_occupancy=occupancy,
                                   max_eligible_negative_burden_occupancy=occupancy)
        policy["alert"]["threshold"] = 1.0
        policies[name] = policy
    scaler = SimpleNamespace(scale_=np.array([1.0]), mean_=np.array([0.0]))
    sigmoid = SimpleNamespace(coef_=np.array([[1.0]]), intercept_=np.array([0.0]))
    calibration = study.calibration_summary({"scale": scaler, "sigmoid": sigmoid})
    head, prepared, pairs = _generated_control_context(121, 241)
    pair = study._pair_diagnostics(head, prepared, pairs, guard)
    identity_variance = study._identity_variance(head, prepared, np.arange(121), guard)
    diagnostics = []
    for fold in range(3):
        roles = {}
        for role_index, name in enumerate(("fit", "stop")):
            drift_count = DRIFT_PAIR_BOUNDS[fold][role_index]
            drift_head, drift_prepared, drift_pairs = _generated_control_context(121, drift_count)
            drift = study._pair_diagnostics(drift_head, drift_prepared, drift_pairs, guard)
            role_identity = copy.deepcopy(identity_variance)
            for person in role_identity["people"]:
                person["count"] = IDENTITY_ROW_BOUNDS[fold][role_index]
            roles[name] = {arm: dict(event_pairs=pair, negative_drift=drift,
                                    identity=role_identity) for arm in study.STUDY_ARMS}
        diagnostics.append(dict(fold=fold, identity=identity,
                                roles=roles,
                                nulls=[dict(seed=seed,
                                            orientation={f"person-{i:03d}": 1.0 for i in range(121)},
                                            stop=pair,
                                            model=dict(_artifact(f"fold-{fold}-null-{seed}.joblib"),
                                                       numeric_manifest=f"fold-{fold}-null-{seed}.joblib.json",
                                                       numeric_manifest_sha256="a" * 64))
                                       for seed in range(101, 120)],
                                nuisance=dict(model=dict(_artifact(f"fold-{fold}-nuisance.joblib"),
                                                         numeric_manifest=f"fold-{fold}-nuisance.joblib.json",
                                                         numeric_manifest_sha256="a" * 64),
                                              fit=pair, stop=pair),
                                nuisance_equals_or_exceeds_candidate_stop_improvement=True))
    generated_people = _generated_people(241)
    guard()
    comparisons = {name: study.compare_person_counts(generated_people, generated_people,
                                                      seed=42, samples=10_000)
                   for name in study.POLICY_NAMES}
    entry = dict(identity=identity, calibration=calibration, policies=policies,
                 model=dict(_artifact("fold-0-timing_pointwise-model.joblib"),
                            numeric_manifest="fold-0-timing_pointwise-model.joblib.json",
                            numeric_manifest_sha256="a" * 64),
                 policy_freeze=_artifact("fold-0-timing_pointwise-policy-freeze.json"),
                 predictions="fold-0-timing_pointwise-predictions.jsonl.gz",
                 predictions_sha256="a" * 64, outer_rows=629_444,
                 availability=dict(original_accepted_rows=629_444,
                                   common_available_rows=629_444, newly_unavailable_rows=0),
                 outer_events=fold_events[0],
                 outer_groups=[f"person-{i:03d}" for i in range(fold_people[0])],
                 primary_constraints=dict(passed=True, failed_criteria=[]))
    advancement_entries = [dict(fold=fold, arm=arm, calibration=calibration,
                                primary_constraints=entry["primary_constraints"],
                                reports={study.PRIMARY_POLICY: {"event_sensitivity": 0.0}})
                           for fold in range(3) for arm in study.STUDY_ARMS]
    advancement = study._advancement(advancement_entries, comparisons)
    advancement["confirmation"] = "Generated cost fixture only; no forecasting result"
    summary = dict(identity=identity, comparisons=comparisons, diagnostics=diagnostics,
                   rows=629_444, people=121, events=241,
                   availability={arm: dict(entry["availability"]) for arm in study.STUDY_ARMS},
                   timing_comparator_availability="Common availability; generated cost fixture",
                   coverage_unchanged_between_matched_arms=True, **advancement)
    return dict(entry=entry, summary=summary)


def measure_control_replay_leaves(*, study, guard, pair_count=723, identity_rows=None):
    """Time control diagnostics once; replay repeats these costs and must charge separately."""
    if type(pair_count) is not int or not 0 < pair_count <= 723:
        raise ValueError("Original saved-pair ceiling required")
    head, prepared, pairs = _generated_control_context(121, pair_count)
    started = time.monotonic()
    for _ in range(57 + 3 * 2):
        guard()
        result = study._pair_diagnostics(head, prepared, pairs, guard)
        if result["pairs"] != pair_count or result["people"] != min(pair_count, 121):
            raise ValueError("Generated control diagnostic cardinality differs")
    pair_seconds = time.monotonic() - started
    started = time.monotonic()
    for _ in range(3 * 2 * 2):
        guard()
        result = study._identity_variance(head, prepared, np.arange(pair_count), guard)
        if len(result["people"]) != min(pair_count, 121):
            raise ValueError("Generated identity diagnostic cardinality differs")
    identity_seconds = time.monotonic() - started
    drift_adapters = []
    for fold in DRIFT_PAIR_BOUNDS:
        for count in fold:
            guard()
            drift_head, drift_prepared, drift_pairs = _generated_control_context(121, count)
            before = time.monotonic()
            study._pair_diagnostics(drift_head, drift_prepared, drift_pairs, guard)
            drift_adapters.append(dict(seconds=time.monotonic() - before, pairs=count))
    full_identity = None
    if identity_rows is not None:
        if identity_rows != 629444:
            raise ValueError('Only complete full identity working set is declared')
        guard()
        large_head, large_prepared, _ = _generated_control_context(121, identity_rows)
        before = time.monotonic()
        value = study._identity_variance(large_head, large_prepared, np.arange(identity_rows), guard)
        full_identity = dict(seconds=time.monotonic() - before, rows=identity_rows,
                             people=len(value['people']), fake_logits='measurement slack only')
    return dict(seconds=pair_seconds + identity_seconds, pair_seconds=pair_seconds,
                identity_seconds=identity_seconds, calls=63, identity_calls=12,
                drift_adapters=drift_adapters, full_identity_adapter=full_identity,
                pairs_per_call=pair_count,
                source_sha256=_digest(study._pair_diagnostics),
                identity_source_sha256=_digest(study._identity_variance),
                scope="generated_only_control_pair_diagnostic_arithmetic_includes_fake_logits",
                fits=0, real_support_authority=False)
