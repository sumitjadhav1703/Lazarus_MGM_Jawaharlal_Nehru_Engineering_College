from collections import Counter

import pytest

from cardiotwin.evaluation import patient_bootstrap, probability_metrics


def test_probability_context_separates_window_and_patient_metrics():
    rows = [{"patient_id": "long", "label": 0, "risk": 0.1} for _ in range(9)]
    rows += [{"patient_id": "short", "label": 1, "risk": 0.1}]
    report = probability_metrics(rows)
    assert report["positive_prevalence"] == 0.1
    assert report["pr_auc_reference"] == 0.1
    assert report["cohort_prevalence_brier_reference"] == pytest.approx(0.09)
    assert report["brier"] == pytest.approx(0.09)
    assert report["patient_macro_brier"] == pytest.approx(0.41)
    assert report["probability_patients"] == 2


def test_bootstrap_keeps_duplicate_linked_patients_in_one_group():
    rows = [{"patient_id": p, "group_id": "same-person", "label": 0, "risk": 0.1} for p in ["a", "b"]]
    patients = {p: Counter(monitored_seconds=30, eligible_seconds=30) for p in ["a", "b"]}
    report = patient_bootstrap(patients, rows, samples=20)
    assert report["independent_groups"] == 1
    assert report["intervals"] == {}


def test_invalid_probabilities_fail_instead_of_creating_calibration_bins():
    for risk in [float("nan"), float("inf"), -0.1, 1.1]:
        with pytest.raises(ValueError, match="probabilities"):
            probability_metrics([{"patient_id": "p", "label": 0, "risk": risk}])


def test_policy_search_can_skip_invariant_probability_metrics(monkeypatch):
    from cardiotwin import evaluation
    from cardiotwin.config import load_config

    rows = [
        {
            "patient_id": "p",
            "record_id": "r",
            "timestamp": t,
            "duration_seconds": 900,
            "accepted": True,
            "risk": 0.99,
            "label": 1,
            "label_reason": "positive",
        }
        for t in range(0, 901, 30)
    ]
    events = [{"patient_id": "p", "record_id": "r", "onset": 720}]
    original = evaluation.evaluate_stream(rows, events, load_config(), bootstrap_samples=0)
    monkeypatch.setattr(evaluation, "probability_metrics", lambda _: pytest.fail("probabilities recomputed"))
    fast = evaluation.evaluate_stream(
        rows, events, load_config(), bootstrap_samples=0, include_probability=False
    )
    for key in ("matched_events", "false_warnings", "warning_hours", "eligible_hours"):
        assert original[key] == fast[key]
    assert fast["probability"] is None


def test_optional_quality_bounds_are_validated():
    from cardiotwin.config import load_config, validate_config

    for key, values in {
        "min_qrs_kurtosis": [0, -1, float("nan"), float("inf"), None, "invalid"],
        "max_saturation_fraction": [-0.1, 1.1, float("nan"), None, "invalid"],
    }.items():
        for value in values:
            config = load_config()
            config["quality"][key] = value
            with pytest.raises(ValueError, match=key):
                validate_config(config)


def test_feature_contract_and_training_options_are_validated():
    from cardiotwin.config import load_config, validate_config

    config = load_config()
    config["feature_version"] = "causal-rr-v1"
    with pytest.raises(ValueError, match="Feature extraction version"):
        validate_config(config)
    config = load_config()
    config["training"]["sample_weighting"] = "unsupported"
    with pytest.raises(ValueError, match="sample weighting"):
        validate_config(config)
    config = load_config()
    config["training"]["calibration_c"] = float("nan")
    with pytest.raises(ValueError, match="calibration_c"):
        validate_config(config)


def finite_rows(times, duration, risk=.99):
    return [dict(patient_id="p", record_id="r", timestamp=t, duration_seconds=duration,
                 accepted=True, risk=risk, label=0, label_reason=None) for t in times]


def test_finite_intervals_bound_gap_support_and_partial_terminal_duration():
    from cardiotwin.config import load_config
    from cardiotwin.evaluation import evaluate_stream

    config = load_config("configs/precursor-cycle.json")
    report = evaluate_stream(finite_rows([0, 30, 60, 180], 195), [], config, bootstrap_samples=0)
    assert report["monitored_seconds"] == 195
    assert report["eligible_seconds"] == report["available_seconds"] == 105
    assert report["missing_cadence_seconds"] == 90
    assert report["warning_seconds"] == 30
    assert report["unresolved_elevated_seconds"] == 15
    assert report["burden_seconds"] == 45
    assert report["warning_details"][0]["end_time"] == 90
    assert report["false_warnings_per_hour"] == pytest.approx(3600 / 105)
    assert report["eligible_negative_burden_occupancy"] == pytest.approx(45 / 105)


def test_finite_expiry_inside_final_interval_and_no_warning_keep_events():
    from cardiotwin.config import load_config
    from cardiotwin.evaluation import evaluate_stream

    cfg = load_config("configs/precursor-cycle.json")
    cfg["alert"]["max_duration_seconds"] = 605
    rows = finite_rows(range(0, 661, 30), 675)
    events = [dict(patient_id="p", record_id="r", onset=660)]
    report = evaluate_stream(rows, events, cfg, bootstrap_samples=0)
    assert report["warning_seconds"] == 605
    assert report["unresolved_elevated_seconds"] == 10
    assert report["burden_seconds"] == 615
    assert report["warning_details"][0]["end_time"] == 665
    assert report["warning_duration_summary"]["max_seconds"] == 605
    rows[-1]["accepted"] = False
    rows[-1]["risk"] = None
    rejected = evaluate_stream(rows, events, cfg, bootstrap_samples=0)
    assert rejected["eligible_seconds"] == 675
    cfg["alert"].update(threshold=1., release_threshold=.75)
    none = evaluate_stream(rows, events, cfg, bootstrap_samples=0)
    assert none["burden_seconds"] == 0
    assert none["eligible_events"] == 1 and none["event_sensitivity"] == 0


def test_both_policy_selectors_reject_persistent_burden_and_bind_search():
    from copy import deepcopy

    from cardiotwin.config import load_config
    from cardiotwin.investigation import select_policy
    from cardiotwin.train import _select_policy

    cfg = load_config("configs/precursor-cycle.json")
    cfg["alert"]["min_event_sensitivity"] = 0
    rows = finite_rows(range(0, 1801, 30), 1800)
    for r in rows:
        r["label"] = int(300 <= r["timestamp"] <= 780)
    events = [dict(patient_id="p", record_id="r", onset=900)]
    selected, search = select_policy(rows, events, cfg)
    assert selected["alert"]["threshold"] == 1
    assert all(not c["meets_constraint"] for c in search if c["threshold"] < .99)
    best, _ = _select_policy(rows, events, cfg)
    assert best is None
    assert select_policy(rows, events, cfg, search=search)[0] == selected
    mutations = []
    changed_rows = deepcopy(rows)
    changed_rows[0]["risk"] = .98
    mutations.append((changed_rows, events, cfg))
    mutations.append((rows, [{**events[0], "onset": 901}], cfg))
    changed_cfg = deepcopy(cfg)
    changed_cfg["target"]["cadence_seconds"] = 60
    mutations.append((rows, events, changed_cfg))
    changed_cfg = deepcopy(cfg)
    changed_cfg["alert"]["max_duration_seconds"] = 300
    mutations.append((rows, events, changed_cfg))
    for r, e, c in mutations:
        with pytest.raises(ValueError, match="identity"):
            select_policy(r, e, c, search=search)


@pytest.mark.parametrize('finite', [False, True])
def test_frozen_policy_packaging_uses_original_scored_stream_identity(finite):
    from copy import deepcopy

    from cardiotwin.config import load_config
    from cardiotwin.investigation import _frozen_policies, select_policy
    cfg = load_config('configs/precursor-cycle.json') if finite else load_config()
    cfg['alert']['min_event_sensitivity'] = 0
    rows = finite_rows(range(0, 1801, 30), 1800)
    for r in rows:
        r['label'] = int(300 <= r['timestamp'] <= 780)
    events = [dict(patient_id='p', record_id='r', onset=900)]
    selected, searches = select_policy(rows, events, cfg)
    packaged = _frozen_policies(rows, events, cfg, searches, selected)
    assert packaged['0.25'] == selected['alert']
    assert set(packaged) == {'0.1', '0.25', '1'}
    if finite:
        changed = deepcopy(rows)
        changed[0]['risk'] = .98
        with pytest.raises(ValueError, match='identity'):
            _frozen_policies(changed, events, cfg, searches, selected)
