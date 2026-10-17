"""Predeclared matching and independently decoded metadata boundaries."""

import csv
import json
from copy import deepcopy

import pytest

from cardiotwin.config import load_config


def api(name):
    import importlib.util

    assert importlib.util.find_spec("cardiotwin.precursor"), "precursor experiment is not implemented"
    from cardiotwin import precursor

    assert hasattr(precursor, name), f"missing precursor API: {name}"
    return getattr(precursor, name)


def row(t, label, *, group="g", patient="p", record="r", accepted=True, **extra):
    return dict(patient_id=patient, record_id=record, group_id=group, timestamp=t,
                accepted=accepted, label=label, features={}, **extra)


def event(onset=3600, *, group="g", patient="p", record="r"):
    return dict(patient_id=patient, record_id=record, group_id=group, onset=onset)


def test_pair_choice_is_fixed_and_role_bounded():
    rows = [row(900, 0), row(3210, 1), row(3240, 1), row(3270, 1)]
    pairs, report = api("build_event_pairs")(rows, [event()], {"g"}, load_config())
    assert [(p.positive_index, p.negative_index) for p in pairs] == [(2, 0)]
    assert report["paired_events"] == 1
    assert api("build_event_pairs")(rows, [event()], {"other"}, load_config())[0] == []
    changed = deepcopy(rows)
    for r in changed:
        r["features"] = {"ecg_hr": 1e9}
    assert api("build_event_pairs")(changed, [event()], {"g"}, load_config())[0] == pairs


@pytest.mark.parametrize("positive", [3000, 3480])
@pytest.mark.parametrize("negative", [0, 1800])
def test_pair_time_boundaries_inclusive(positive, negative):
    pairs, _ = api("build_event_pairs")([row(negative, 0), row(positive, 1)],
                                        [event()], {"g"}, load_config())
    assert len(pairs) == 1


def test_pair_ties_prefer_earlier_and_negative_cannot_be_reused():
    rows = [row(870, 0), row(930, 0), row(3210, 1), row(3270, 1), row(3840, 1)]
    pairs, _ = api("build_event_pairs")(rows, [event(4200), event()], {"g"}, load_config())
    assert [(p.positive_index, p.negative_index) for p in pairs] == [(2, 0), (4, 1)]


@pytest.mark.parametrize("bad", [row(900, None), row(900, 0, accepted=False), row(900, 1),
                                  row(900, 0, patient="else", record="else")])
def test_negative_requires_accepted_eligible_same_record(bad):
    pairs, report = api("build_event_pairs")([bad, row(3240, 1)], [event()], {"g"}, load_config())
    assert pairs == []
    assert report["exclusions"][0]["reason"] == "no_eligible_unused_negative"


def test_duplicate_events_and_alias_changes_fail():
    build = api("build_event_pairs")
    with pytest.raises(ValueError, match="duplicate"):
        build([row(900, 0), row(3240, 1)], [event(), event()], {"g"}, load_config())
    with pytest.raises(ValueError, match="identity"):
        build([row(900, 0), row(3240, 1, group="else")], [event()], {"g"}, load_config())


def test_positive_may_not_be_claimed_by_two_events():
    pairs, report = api("build_event_pairs")([row(900, 0), row(3240, 1)],
                                            [event(), event(3700)], {"g"}, load_config())
    assert pairs == []
    assert report["observable_events"] == 2
    assert {x["reason"] for x in report["exclusions"]} == {"ambiguous_positive_event"}


def test_no_cadence_and_rejected_opportunities_are_distinct():
    build = api("build_event_pairs")
    assert build([row(900, 0)], [event()], {"g"}, load_config())[1]["exclusions"][0]["reason"] == "no_cadence_opportunity"
    report = build([row(900, 0), row(3240, 1, accepted=False)], [event()], {"g"}, load_config())[1]
    assert report["exclusions"][0]["reason"] == "no_accepted_positive"
    assert report["paired_events"] == 0


def test_independent_intervals_cross_day_and_mask_first_30_seconds():
    decoded = api("reference_intervals")(
        [dict(start_file_index="0", start_qrs_index="17000", end_file_index="1", end_qrs_index="8000")],
        [18000, 18000], 200,
    )
    assert decoded == [(0.0, 30.0, "unknown"), (30.0, 85.0, "other"), (85.0, 90.0, "AF"),
                       (90.0, 120.0, "unknown"), (120.0, 130.0, "AF"), (130.0, 180.0, "other")]


@pytest.mark.parametrize("t,label,reason", [
    (599.995, None, "history"), (600, None, "unknown"), (629.995, None, "unknown"),
    (630, 0, None), (3000, 1, None), (2999.995, 0, None), (3480, 1, None),
    (3480.005, None, "gap"), (3600, None, "current_AF"), (3660, None, "washout"),
    (4259.995, None, "washout"), (4260, 0, None), (9400, 0, None), (9400.005, None, "future"),
])
def test_independent_label_sample_boundaries(t, label, reason, monkeypatch):
    import cardiotwin.data
    import cardiotwin.labels

    def forbidden(*args, **kwargs):
        raise AssertionError("production reference was used")

    monkeypatch.setattr(cardiotwin.labels, "label_at", forbidden)
    monkeypatch.setattr(cardiotwin.labels, "eligible_onsets", forbidden)
    monkeypatch.setattr(cardiotwin.data, "_iridia_intervals", forbidden)
    intervals = [(0, 30, "unknown"), (30, 3600, "other"), (3600, 3660, "AF"), (3660, 10000, "other")]
    assert api("reference_events")(intervals, 10000, load_config()) == [3600]
    assert api("reference_label")(t, intervals, 10000, load_config()) == (label, reason)


def test_roles_reproduce_seed_and_reject_cross_role_aliases():
    rows, events, groups = [], [], []
    for n in range(60):
        g = f"g{n:02d}"
        groups.append(g)
        rows.extend([row(900, 0, group=g, patient=g, record=g), row(3240, 1, group=g, patient=g, record=g)])
        events.append(event(group=g, patient=g, record=g))
    index = {"allowed_groups": groups, "group_ids": {g: g for g in groups}}
    split = {"development_groups": groups, "test_groups": ["forbidden"], "seed": 42,
             "group_ids": index["group_ids"]}
    roles = api("make_roles")(rows, events, index, split)
    assert len(roles) == 3
    assert sorted(g for f in roles for g in f["outer"]) == groups
    for f in roles:
        memberships = [g for role in ("fit", "stop", "calibration", "policy", "outer") for g in f[role]]
        assert sorted(memberships) == groups
    with pytest.raises(ValueError, match="seed"):
        api("make_roles")(rows, events, index, split, seed=43)
    index["allowed_groups"] = groups[:-1]
    with pytest.raises(ValueError, match="membership"):
        api("make_roles")(rows, events, index, split)


def metadata_fixture(tmp_path):
    record = {"record_id": "r", "patient_id": "iridia-af:p", "fs": 200., "duration_seconds": 10000.,
              "signal_parts": [{"samples": 2000000}]}
    folder = tmp_path / "records" / "r"
    folder.mkdir(parents=True)
    labels = folder / "r_ecg_labels.csv"
    labels.write_text("start_file_index,start_qrs_index,end_file_index,end_qrs_index\n0,720000,0,732000\n")
    with (tmp_path / "metadata.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["record_id", "patient_id", "record_samples", "record_files"])
        writer.writeheader()
        writer.writerow(dict(record_id="r", patient_id="p", record_samples=2000000, record_files=1))
    (tmp_path / "continuity-diagnostics.json").write_text(json.dumps({"records": [
        dict(record_id="r", patient_id="iridia-af:p", daily_samples=[2000000])]}))
    from cardiotwin.data import file_hash
    index = {"records": [{"record": record, "input_contract": {"annotation_sha256": file_hash(labels)}}]}
    return index


def test_missing_multiday_counts_fails_without_raw_access(tmp_path, monkeypatch):
    import h5py

    import cardiotwin.data

    index = metadata_fixture(tmp_path)
    index["records"][0]["record"]["signal_parts"] = [{"samples": 1000000}, {"samples": 1000000}]
    (tmp_path / "continuity-diagnostics.json").unlink()
    def forbidden(*args, **kwargs):
        raise AssertionError("raw access")
    monkeypatch.setattr(h5py, "File", forbidden)
    monkeypatch.setattr(cardiotwin.data, "read_signal", forbidden)
    report = api("audit_metadata")([], [], index, tmp_path, load_config())
    assert report["status"] == "independent_reference_incomplete"


def test_metadata_audit_detects_label_disagreement(tmp_path):
    index = metadata_fixture(tmp_path)
    rows = [row(3240, 0, patient="iridia-af:p", duration_seconds=10000)]
    report = api("audit_metadata")(rows, [event(patient="iridia-af:p")], index, tmp_path, load_config())
    assert report["status"] == "failed"
    assert report["label_disagreements"][0]["expected_label"] == 1


def test_cadence_accepts_terminal_partial_row_and_rejects_missing_interior(tmp_path, monkeypatch):
    import h5py
    from sklearn.linear_model import LogisticRegression

    import cardiotwin.data
    from cardiotwin.labels import RhythmInterval, label_at

    index = metadata_fixture(tmp_path)
    cfg = load_config()
    intervals = [RhythmInterval(0, 30, "unknown"), RhythmInterval(30, 3600, "other"),
                 RhythmInterval(3600, 3660, "AF"), RhythmInterval(3660, 10000, "other")]
    rows = []
    for t in [*range(0, 10000, 30), 10000]:
        label, reason = label_at(t, intervals, 10000, cfg)
        rows.append(row(t, label, patient="iridia-af:p", label_reason=reason, duration_seconds=10000))
    def forbidden(*args, **kwargs):
        raise AssertionError("audit read raw signal or fit a model")
    monkeypatch.setattr(h5py, "File", forbidden)
    monkeypatch.setattr(cardiotwin.data, "read_signal", forbidden)
    monkeypatch.setattr(LogisticRegression, "fit", forbidden)
    audit = api("audit_metadata")
    report = audit(rows, [event(patient="iridia-af:p")], index, tmp_path, cfg)
    assert report["status"] == "passed"
    assert report["checked_rows"] == 335
    report = audit([r for r in rows if r["timestamp"] != 600], [event(patient="iridia-af:p")], index, tmp_path, cfg)
    assert report["status"] == "failed"
    assert len(report["cadence_disagreements"]) == 1


def test_pair_support_keeps_unpaired_people_and_stops_before_fitting():
    rows = [row(900, 0), row(3240, 1), row(3240, 1, group="b", patient="b", record="b")]
    events = [event(), event(group="b", patient="b", record="b")]
    roles = [dict(fold=0, fit=["g", "b"], stop=["g"], calibration=["g"], policy=["g"], outer=[])]
    support, _ = api("pair_support")(rows, events, roles, load_config())
    assert support["status"] == "feasibility_failed"
    assert support["folds"][0]["roles"]["fit"]["observable_event_fraction"] == .5
    assert support["folds"][0]["roles"]["fit"]["positive_person_fraction"] == .5


def test_saved_scores_validate_hash_and_report_person_balanced_pairs(tmp_path):
    import gzip

    from cardiotwin.data import file_hash

    path = tmp_path / "scores.gz"
    rows = [row(900, 0), row(3240, 1)]
    with gzip.open(path, "wt") as handle:
        for r, risk in zip(rows, [.1, .3]):
            handle.write(json.dumps({**r, "risk": risk}) + "\n")
    pairs, _ = api("build_event_pairs")(rows, [event()], {"g"}, load_config())
    report = api("score_pair_diagnostics")([dict(path=path, sha256=file_hash(path), fold=0)], rows, pairs, {"g"})
    assert report["pair_concordance"] == 1
    assert report["person_macro_difference"] == pytest.approx(.2)
    assert report["paired_people"] == 1
    assert report["raw_pair_analysis"] == "unavailable_unsaved_logits"
    with pytest.raises(ValueError, match="checksum"):
        api("score_pair_diagnostics")([dict(path=path, sha256="changed", fold=0)], rows, pairs, {"g"})


def linear_fixture():
    from cardiotwin.features import CORE_ECG_FIELDS, ENGINEERED_ECG_FIELDS

    schema = sorted(CORE_ECG_FIELDS + ENGINEERED_ECG_FIELDS)
    rows, pairs = [], []
    Pair = api("EventPair")
    for n in range(6):
        group = str(n)
        for positive in (True, False):
            features = dict.fromkeys(schema, float(n * 100))
            features[schema[-1]] = None
            rows.append(row(3240 if positive else 900, int(positive), group=group, patient=group, record=group))
            rows[-1]["features"] = features
        pairs.append(Pair(group, group, group, 3600, 2 * n, 2 * n + 1))
    return rows, pairs, schema


def test_pair_weights_equalize_people():
    Pair = api("EventPair")
    pairs = [Pair("a", "a", "r1", 3600, 0, 1), Pair("a", "a", "r1", 7200, 2, 3), Pair("b", "b", "r2", 3600, 4, 5)]
    weights = api("pair_weights")(pairs)
    assert weights.mean() == pytest.approx(1.)
    assert weights[:2].sum() == pytest.approx(weights[2])


def test_rank_cancels_static_identity_and_shares_preprocessing():
    import pickle

    import numpy as np

    from cardiotwin.modeling import matrix, raw_score

    rows, pairs, schema = linear_fixture()
    arms, diagnostics = api("fit_linear_pair_arms")(rows, pairs, schema)
    point, rank = (arms[k] for k in ("pointwise_pair_control", "within_person_rank"))
    assert pickle.dumps(point[:2], protocol=5) == pickle.dumps(rank[:2], protocol=5)
    assert diagnostics["difference_rank"] == 0
    np.testing.assert_allclose(rank["model"].coef_, 0., atol=1e-12)
    np.testing.assert_allclose(raw_score(rank, matrix([r["features"] for r in rows], schema)), 0., atol=1e-12)
    assert rank["model"].n_features_in_ == 46
    assert all(v == pytest.approx(1 / 6) for v in diagnostics["person_total_mass"].values())


def test_fitting_ignores_later_rows_and_new_missingness_keeps_dimensions():
    import numpy as np

    from cardiotwin.modeling import matrix, raw_score

    rows, pairs, schema = linear_fixture()
    for p in pairs:
        rows[p.positive_index]["features"][schema[0]] = 2.
        rows[p.negative_index]["features"][schema[0]] = -2.
    later = row(3240, 0, group="outer", patient="outer", record="outer")
    later["features"] = dict.fromkeys(schema, 1e20)
    a, _ = api("fit_linear_pair_arms")(rows + [later], pairs, schema)
    later["features"] = dict.fromkeys(schema)
    later["label"] = 1
    b, _ = api("fit_linear_pair_arms")(rows + [later], pairs, schema)
    for name in a:
        np.testing.assert_array_equal(a[name]["model"].coef_, b[name]["model"].coef_)
        np.testing.assert_array_equal(a[name]["scale"].mean_, b[name]["scale"].mean_)
        assert np.isfinite(raw_score(b[name], matrix([later["features"]], schema))).all()
    scores = raw_score(a["within_person_rank"], matrix([r["features"] for r in rows], schema))
    assert all(scores[p.positive_index] > scores[p.negative_index] for p in pairs)
    with pytest.raises(ValueError, match="schema"):
        api("fit_linear_pair_arms")(rows, pairs, schema[:-1])


def test_null_controls_only_use_fit_pairs_and_drift_has_no_reused_rows():
    rows, pairs, schema = linear_fixture()
    arms, _ = api("fit_linear_pair_arms")(rows, pairs[:4], schema)
    controls = api("diagnostic_controls")(rows, pairs[:4], pairs[4:], schema, arms)
    assert [x["seed"] for x in controls["person_orientation_nulls"]] == list(range(101, 120))
    assert controls["stop_people"] == 2
    assert all(x["person_macro_concordance"] == .5 for x in controls["person_orientation_nulls"])
    drift_rows = [row(t, 0) for t in [1260, 3600, 4860, 7200, 9000]]
    drift = api("negative_drift_pairs")(drift_rows, {"g"})
    assert [(p.positive_index, p.negative_index) for p in drift] == [(1, 0), (3, 2)]


def test_offline_pair_strata_keep_recurrence_prior_af_and_quality_out_of_inputs():
    rows, pairs, schema = linear_fixture()
    arms, _ = api('fit_linear_pair_arms')(rows, pairs, schema)
    rows[0]['duration_seconds'] = 7200
    rows[0]['features'].update(quality_missing_fraction=.1, baseline_count=8)
    events = [dict(group_id='0', patient_id='0', record_id='0', onset=1800),
              dict(group_id='0', patient_id='0', record_id='0', onset=3600)]
    report = api('pair_strata')(rows, pairs[:1], events, {'0': [2100]}, arms)
    assert report[0]['recurrence'] == 'recurrent'
    assert report[0]['prior_af_distance_seconds'] == 1140
    assert report[0]['record_phase'] == .45
    assert report[0]['quality_baseline']['baseline_count'] == 8
    assert set(arms['within_person_rank']['model'].coef_[0].shape) == {46}
