import json
from pathlib import Path

import frozen_replay

from cardiotwin import ecgfounder_study as study
from cardiotwin import evaluation


def test_tiny_frozen_reconstruction_keeps_actual_equality_tie_logic():
    root = Path(__file__).resolve().parents[4]
    result = frozen_replay.measure(count=7, config=json.loads((root / 'configs/precursor-cycle.json').read_text()),
                                   study=study, evaluation=evaluation, guard=lambda: None)
    assert result['numerical_evaluation_calls'] == 2 * result['thresholds']
    assert result['numerical_evaluation_rows'] == 7 * result['numerical_evaluation_calls']
    assert result['selectors'] == result['fits'] == 0
    assert result['full_replay_latency'] is False


def test_tiny_real_generated_evaluation_kernels():
    root = Path(__file__).resolve().parents[4]
    result = frozen_replay.numerical_evaluations(count=7,
        config=json.loads((root / 'configs/precursor-cycle.json').read_text()),
        evaluation=evaluation, guard=lambda: None)
    assert len(result['evaluations']) == 24
    assert all(value['rows'] == 7 for value in result['evaluations'])
    assert result['numerical_substitution'] is False


def test_exact_record_event_geometry_exercises_all_missed_scan_weight():
    import metadata_geometry
    from test_metadata_geometry import tiny_shape
    shape=tiny_shape()
    root=Path(__file__).resolve().parents[4]
    result=frozen_replay.numerical_evaluations(shape=shape,
        config=json.loads((root/'configs/precursor-cycle.json').read_text()),
        evaluation=evaluation, guard=lambda:None)
    assert len(result['evaluations'])==24
    assert {r['all_missed_row_event_visits'] for r in result['evaluations'] if r['role']=='outer'}=={24}
    assert {r['rows'] for r in result['evaluations'] if r['role']=='policy'}=={10}
    assert all(r['events']==3 for r in result['evaluations'] if r['role']=='outer')
    assert any(r['unknown_warnings'] for r in result['evaluations'] if r['pattern']=='unknown_warnings')
    assert result['projection']['calls']=={'search':480,'outer':60}
    assert result['projection']['fixed_call_controls']==0
    assert result['aggregate_geometry']==metadata_geometry.public_summary(shape)
