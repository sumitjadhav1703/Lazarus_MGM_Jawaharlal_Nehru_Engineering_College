"""Fresh-output and scientific-gate command guards."""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/run_precursor_cycle.py"


def test_audit_command_rejects_existing_output_before_inputs(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    proc = subprocess.run([sys.executable, str(SCRIPT), "audit", "--cache", "missing", "--split", "missing",
                           "--metadata-root", "missing", "--config", "configs/investigation.json",
                           "--output", str(output)], capture_output=True, text=True)
    assert proc.returncode != 0
    assert "already exists" in proc.stderr
    assert list(output.iterdir()) == []


def test_linear_refuses_failed_diagnostic_before_any_fit(tmp_path, monkeypatch):
    import json

    from sklearn.linear_model import LogisticRegression

    from cardiotwin import precursor
    from cardiotwin.config import load_config

    audit = tmp_path / "receipt.json"
    audit.write_text(json.dumps({"status": "feasibility_failed"}))
    def forbidden(*args, **kwargs):
        pytest.fail("fit before scientific gate")
    monkeypatch.setattr(LogisticRegression, "fit", forbidden)
    assert hasattr(precursor, "run_linear_comparison")
    with pytest.raises(ValueError, match="diagnostic"):
        precursor.run_linear_comparison("missing", "missing", audit, load_config("configs/precursor-cycle.json"),
                                         tmp_path / "linear", deadline=float("inf"))


def study_fixture():
    from cardiotwin.precursor import LINEAR_SCHEMA, EventPair
    rows, events, pairs = [], [], []
    for n in range(15):
        group = f'g{n}'
        for t, label in ((900, 0), (3210, 1), (3240, 1), (3270, 1)):
            features = dict.fromkeys(LINEAR_SCHEMA, float(n))
            features[LINEAR_SCHEMA[0]] = float(label * 2 - 1)
            rows.append(dict(patient_id=group, group_id=group, record_id=group, timestamp=t,
                             duration_seconds=4200, accepted=True, label=label, label_reason=None,
                             reason=None, features=features))
        events.append(dict(patient_id=group, group_id=group, record_id=group, onset=3600))
        pairs.append(EventPair(group, group, group, 3600, n * 4 + 2, n * 4))
    roles = []
    for fold in range(3):
        outer = [f'g{n}' for n in range(fold*5, fold*5+5)]
        training = [f'g{n}' for n in range(15) if f'g{n}' not in outer]
        roles.append(dict(fold=fold, fit=training[:4], stop=training[4:6], calibration=training[6:8],
                          policy=training[8:10], outer=outer))
    return rows, events, roles, pairs


def test_linear_three_fold_core_freezes_policies_and_saves_complete_streams(tmp_path):
    import gzip
    import json
    import time

    from cardiotwin import precursor
    from cardiotwin.config import load_config
    rows, events, roles, pairs = study_fixture()
    output = tmp_path / 'core'
    output.mkdir()
    assert hasattr(precursor, 'execute_linear_folds')
    result = precursor.execute_linear_folds(rows, events, roles, pairs, load_config('configs/precursor-cycle.json'),
                                             output, {'source_sha256': 'source'}, deadline=time.monotonic()+60)
    assert result['status'] == 'complete'
    assert len(result['folds']) == 6
    for fold in result['folds']:
        assert set(fold['policies']) == {'far_0.1_occupancy_0.01', 'far_0.1_occupancy_0.05', 'far_0.1_occupancy_0.1',
                                        'far_0.25_occupancy_0.01', 'far_0.25_occupancy_0.05', 'far_0.25_occupancy_0.1',
                                        'far_1_occupancy_0.01', 'far_1_occupancy_0.05', 'far_1_occupancy_0.1', 'legacy_bridge'}
        stream = [json.loads(line) for line in gzip.open(output/fold['predictions'], 'rt')]
        assert len(stream) == 20
        assert all('raw_logit' in r and 0 < r['risk'] < 1 for r in stream)
        assert fold['outer_rows'] == 20 and fold['outer_events'] == 5
        assert fold['calibration']['sample_weighting'] == 'none_natural_monitoring_prevalence'
        assert (output / fold['policy_freeze']).is_file()


@pytest.mark.parametrize('mutation', ['incomplete', 'stream_hash', 'source_hash', 'pair_hash'])
def test_review_refuses_incomplete_or_stale_run_before_stream_use(tmp_path, mutation):
    import json

    from cardiotwin import precursor
    root = tmp_path / 'run'
    root.mkdir()
    (root/'receipt.json').write_text(json.dumps({'status':'running' if mutation == 'incomplete' else 'complete',
                                                'artifacts': {'missing': 'wrong'}}))
    assert hasattr(precursor, 'review_comparison')
    with pytest.raises(ValueError, match='complete|artifact|binding'):
        precursor.review_comparison(root, tmp_path/'review', bootstrap_samples=10)


def test_person_bootstrap_uses_whole_independent_groups_and_keeps_negative_only_people():
    from cardiotwin import precursor
    assert hasattr(precursor, 'compare_person_counts')
    common = dict(eligible_seconds=3600, monitored_seconds=4000, available_seconds=3900,
                  eligible_negative_seconds=3000, warning_seconds=30, burden_seconds=40,
                  eligible_negative_burden_seconds=20, false_warnings=1, warnings=2,
                  unassessable_warnings=0, eligible_events=1, matched_events=0)
    baseline = [dict(common, group_id='a'), dict(common, group_id='b', eligible_events=0)]
    candidate = [dict(baseline[0], matched_events=1), baseline[1]]
    report = precursor.compare_person_counts(baseline, candidate, seed=42, samples=10000)
    assert report['people'] == 2 and report['negative_only_people'] == 1
    assert report['sensitivity_delta'] == 1
    assert report['invalid_draws'] > 0
    assert len(report['leave_one_person_out']) == 2
    assert report['bootstrap']['event_sensitivity_delta']['low'] == 1


@pytest.mark.parametrize('mutation', ['pair', 'stream', 'source', 'artifact_manifest', 'retrospective'])
def test_complete_review_replays_full_cohort_and_detects_changed_artifact(tmp_path, monkeypatch, mutation):
    import json
    import time
    from dataclasses import asdict

    from cardiotwin import investigation, precursor
    from cardiotwin.config import load_config
    rows, events, roles, pairs = study_fixture()
    cfg = load_config('configs/precursor-cycle.json')
    # Require 90s beyond the synthetic 60s positive run, avoiding fitted-risk threshold ties.
    cfg['alert']['persistence_seconds'] = 90
    audit_dir = tmp_path/'audit'
    audit_dir.mkdir()
    (audit_dir/'pairs.jsonl').write_text(''.join(json.dumps(asdict(p))+'\n' for p in pairs))
    (audit_dir/'roles.json').write_text(json.dumps({'roles': roles}))
    (audit_dir/'metadata-audit.json').write_text(json.dumps({'records': []}))
    audit_path = audit_dir/'receipt.json'
    audit_path.write_text('{}')
    audit = dict(cache_index_sha256='cache', cache_identity='index', roles_sha256=investigation._digest(roles),
                 pairs_sha256=precursor.file_hash(audit_dir/'pairs.jsonl'), split_sha256='split', compact_binding={'test': 'binding'})
    index = dict(sha256='index', allowed_groups=sorted({r['group_id'] for r in rows}))
    monkeypatch.setattr(precursor, 'validate_diagnostic', lambda *a: (rows, events, index, roles, [asdict(p) for p in pairs], audit))
    monkeypatch.setattr(investigation, 'read_cache', lambda *a: (rows, events, index))
    monkeypatch.setattr(precursor, 'compact_cache_binding', lambda *a, **k: audit['compact_binding'])
    actual_hash = precursor.file_hash
    monkeypatch.setattr(precursor, 'file_hash', lambda p: 'cache' if Path(p).name == 'index.json' else actual_hash(p))
    run = tmp_path/'run'
    result = precursor.run_linear_comparison(tmp_path/'cache', tmp_path/'split', audit_path, cfg, run,
                                               deadline=time.monotonic()+60)
    assert result['status'] == 'complete' and result['completed_arms'] == 6
    if mutation == 'retrospective':
        import joblib
        declaration = json.loads((run/'predeclaration.json').read_text())
        declaration['identity'].pop('compact_binding')
        (run/'predeclaration.json').write_text(json.dumps(declaration))
        result['identity'].pop('compact_binding')
        summary = json.loads((run/'summary.json').read_text())
        summary['identity'].pop('compact_binding')
        for entry in summary['folds']:
            entry['identity'].pop('compact_binding')
            package = joblib.load(run/entry['model'])
            package['identity'].pop('compact_binding')
            joblib.dump(package, run/entry['model'])
            entry['model_sha256'] = actual_hash(run/entry['model'])
        (run/'summary.json').write_text(json.dumps(summary))
        result['artifacts'] = {str(p.relative_to(run)):actual_hash(p) for p in run.rglob('*') if p.is_file() and p.name != 'receipt.json'}
        (run/'receipt.json').write_text(json.dumps(result))
        with pytest.raises(ValueError, match='explicit retrospective'):
            precursor.review_comparison(run, tmp_path/'implicit')
        monkeypatch.setattr(precursor, 'validate_compact_source', lambda *a, **k: {'status':'passed','historical_fit_time_identity_verified':False})
        review = precursor.review_comparison(run, tmp_path/'retrospective', bootstrap_samples=100, retrospective_provenance=True)
        assert review['experiment_eligibility'] == 'provenance_incomplete'
        assert review['historical_fit_time_identity_verified'] is False
        assert review['decision'] == 'reject' and not review['promising_development_lead']
        assert (tmp_path/'retrospective/provenance-correction.json').is_file()
        def fail(*a, **k):
            raise ValueError('actual source feature mismatch')
        monkeypatch.setattr(precursor, 'validate_compact_source', fail)
        with pytest.raises(ValueError, match='actual source feature mismatch'):
            precursor.review_comparison(run, tmp_path/'failed-retrospective', retrospective_provenance=True)
        failed = json.loads((tmp_path/'failed-retrospective/receipt.json').read_text())
        assert failed['status'] == 'failed' and failed['experiment_eligibility'] == 'provenance_incomplete'
        return
    review = precursor.review_comparison(run, tmp_path/'review', bootstrap_samples=100)
    assert review['status'] == 'complete'
    assert (review['rows'], review['events'], review['people']) == (60, 15, 15)
    assert review['decision'] == 'reject' and review['neural_eligible'] is False
    if mutation == 'artifact_manifest':
        result['artifacts'] = {}
        (run/'receipt.json').write_text(json.dumps(result))
    else:
        name = 'pairs.jsonl' if mutation == 'pair' else 'fold-0-pointwise_pair_control-predictions.jsonl.gz' if mutation == 'stream' else 'source-package/src/cardiotwin/alerts.py'
        (run/name).write_text('changed')
    with pytest.raises(ValueError, match='artifact binding'):
        precursor.review_comparison(run, tmp_path/'stale')


def test_linear_refuses_historical_audit_without_compact_binding_before_inputs(tmp_path):
    import json

    from cardiotwin import precursor
    from cardiotwin.config import load_config
    audit = tmp_path/'old.json'
    audit.write_text(json.dumps(dict(status='passed',metadata_status='passed',pair_support_status='passed')))
    with pytest.raises(ValueError, match='compact provenance incomplete'):
        precursor.run_linear_comparison('missing', 'missing', audit, load_config('configs/precursor-cycle.json'),
                                        tmp_path/'blocked', deadline=float('inf'))


def test_pair_support_receipt_roundtrip_preserves_integer_class_counts(tmp_path):
    import json

    from cardiotwin import precursor
    from cardiotwin.config import load_config
    rows, events, roles, _ = study_fixture()
    cfg = load_config('configs/precursor-cycle.json')
    support, pairs = precursor.pair_support(rows, events, roles, cfg)
    (tmp_path/'pair-support.json').write_text(json.dumps(support))
    (tmp_path/'pairs.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in pairs))
    assert hasattr(precursor, 'validate_pair_receipts')
    assert precursor.validate_pair_receipts(rows, events, roles, cfg, tmp_path) == pairs
    saved = json.loads((tmp_path/'pair-support.json').read_text())
    saved['folds'][0]['roles']['calibration']['natural_label_counts']['1'] += 1
    (tmp_path/'pair-support.json').write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='support changed'):
        precursor.validate_pair_receipts(rows, events, roles, cfg, tmp_path)


@pytest.mark.parametrize('mutation', ['raw_logit', 'risk', 'slope', 'model_hash'])
def test_independent_model_replay_detects_score_calibration_and_artifact_changes(tmp_path, mutation):
    import copy

    import joblib

    from cardiotwin import investigation, precursor
    from cardiotwin.config import load_config
    from cardiotwin.train import _calibrate, calibration_diagnostics

    rows, events, roles, pairs = study_fixture()
    cfg = load_config('configs/precursor-cycle.json')
    role = roles[0]
    arms, _ = precursor.fit_linear_pair_arms(rows, [p for p in pairs if p.group_id in role['fit']], precursor.LINEAR_SCHEMA)
    pipeline = arms['within_person_rank']
    cal_rows, stop_rows, outer_rows = (investigation._subset(rows, role[k]) for k in ('calibration','stop','outer'))
    calibrator = _calibrate(pipeline, precursor.LINEAR_SCHEMA, cal_rows, 42, calibration_c=100.)
    predictions = precursor.score_linear_rows(pipeline, calibrator, outer_rows, precursor.LINEAR_SCHEMA, deadline=float('inf'))
    model_path = tmp_path/'model.joblib'
    identity = {'source_sha256': 'test'}
    joblib.dump(dict(pipeline=pipeline,calibrator=calibrator,schema=precursor.LINEAR_SCHEMA,config=cfg,identity=identity),model_path)
    entry = dict(model=model_path.name,model_sha256=precursor.file_hash(model_path),identity=identity,
                 calibration=calibration_diagnostics(pipeline,precursor.LINEAR_SCHEMA,calibrator,cal_rows))
    assert hasattr(precursor, 'verify_linear_artifact')
    result = precursor.verify_linear_artifact(tmp_path, entry, predictions, outer_rows, cal_rows, stop_rows, cfg)
    assert result['rows_checked'] == 20
    assert result['available_logits_checked'] == 20
    assert result['calibration']['raw_score_slope'] == entry['calibration']['raw_score_slope']
    changed = copy.deepcopy(predictions)
    if mutation in ('raw_logit','risk'):
        changed[0][mutation] += .01
    elif mutation == 'slope':
        entry['calibration']['raw_score_slope'] += 1
    else:
        entry['model_sha256'] = 'stale'
    with pytest.raises(ValueError, match='model|logit|risk|calibration'):
        precursor.verify_linear_artifact(tmp_path, entry, changed, outer_rows, cal_rows, stop_rows, cfg)


def compact_fixture(tmp_path):
    import json

    from cardiotwin import investigation
    from cardiotwin.config import load_config
    from cardiotwin.data import file_hash
    rows = [dict(patient_id='p', group_id='g', record_id='r', timestamp=i*30., duration_seconds=90.,
                 accepted=bool(i), label=int(i == 2) if i else None, reason=None if i else 'history',
                 label_reason=None if i else 'history', features={'ecg_hr':70.+i,'ecg_rmssd':None if i == 0 else .02}) for i in range(3)]
    (tmp_path/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    cfg = load_config()
    index = dict(cache_version=investigation.CACHE_VERSION, feature_version=cfg['feature_version'], config=cfg,
                 config_sha256=investigation._digest(cfg), group_ids={'p':'g'}, allowed_groups=['g'], events=[],
                 row_count=3, rows_sha256=file_hash(tmp_path/'rows.jsonl'), feature_fields=['ecg_hr','ecg_rmssd'])
    index['sha256'] = investigation._digest(index)
    (tmp_path/'index.json').write_text(json.dumps(index))
    loaded, _, index = investigation.read_cache(tmp_path)
    return loaded, index


@pytest.mark.parametrize('mutation', ['schema', 'records', 'categories', 'features', 'metadata'])
def test_compact_binding_rejects_every_consumed_mapping_or_array_change(tmp_path, mutation):
    import json

    import numpy as np

    from cardiotwin import precursor
    rows, index = compact_fixture(tmp_path)
    assert hasattr(precursor, 'compact_cache_binding')
    original = precursor.compact_cache_binding(tmp_path)
    assert original == json.loads(json.dumps(original))
    if mutation in ('features', 'metadata'):
        path = tmp_path/(mutation+'.npy')
        values = np.load(path).copy()
        if mutation == 'features':
            values[1,0] += 1
        else:
            values[1]['timestamp'] += 1
        np.save(path, values)
    else:
        path = tmp_path/'compact.json'
        contract = json.loads(path.read_text())
        if mutation == 'schema':
            contract['features'].reverse()
        elif mutation == 'records':
            contract['records'][0]['patient_id'] = 'changed'
        else:
            contract['categories']['reason'][1] = 'changed'
        path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match='compact|Compact|schema'):
        precursor.compact_cache_binding(tmp_path, expected=original)


@pytest.mark.parametrize('mutation', ['feature', 'missingness', 'identity', 'order', 'category'])
def test_retrospective_source_validation_checks_all_features_and_metadata(tmp_path, mutation):
    from cardiotwin import precursor
    rows, index = compact_fixture(tmp_path)
    assert hasattr(precursor, 'validate_compact_source')
    report = precursor.validate_compact_source(tmp_path, rows, index)
    assert report['rows_checked'] == 3 and report['feature_values_checked'] == 6
    assert report['historical_fit_time_identity_verified'] is False
    copied = [{**dict(r), 'features':dict(r['features'])} for r in rows]
    if mutation == 'feature':
        copied[1]['features']['ecg_hr'] += 1
    elif mutation == 'missingness':
        copied[0]['features']['ecg_rmssd'] = 0
    elif mutation == 'identity':
        copied[0]['patient_id'] = 'wrong'
    elif mutation == 'order':
        copied.reverse()
    else:
        copied[0]['reason'] = 'wrong'
    with pytest.raises(ValueError, match='source|feature|metadata'):
        precursor.validate_compact_source(tmp_path, copied, index)
