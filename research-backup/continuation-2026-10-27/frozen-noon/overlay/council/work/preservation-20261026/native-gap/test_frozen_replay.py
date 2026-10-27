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
    assert len(result['evaluations']) == 8
    assert all(value['rows'] == 7 for value in result['evaluations'].values())
    assert result['numerical_substitution'] is False
