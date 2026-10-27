from tempfile import TemporaryDirectory

import package_replay

from cardiotwin import ecgfounder_compare as compare
from cardiotwin import ecgfounder_study as study


def test_one_fold_generated_states_and_exact_independent_replay():
    with TemporaryDirectory() as temp:
        result = package_replay.measure(temp + '/packages', identity={'fixture': 'generated-only'},
            study=study, compare=compare, guard=lambda: None,
            storage_check=lambda: dict(allocated_bytes=0, logical_bytes=0), folds=1)
    assert result['packages'] == result['manifests'] == 22
    assert result['fit_pairs'] == 241
    assert result['fits'] == 0
