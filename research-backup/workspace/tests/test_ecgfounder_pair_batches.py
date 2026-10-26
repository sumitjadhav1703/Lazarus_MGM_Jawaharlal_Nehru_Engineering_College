"""Diagnostic batching preserves the positive-then-negative score ordering."""
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from cardiotwin import ecgfounder_compare as compare
from cardiotwin import ecgfounder_study as study
from cardiotwin.ecgfounder_tables import ROW_DTYPE
from cardiotwin.precursor import LINEAR_SCHEMA, EventPair


@pytest.mark.parametrize("count", [1, 241, 2049])
def test_pair_batches_preserve_mixed_scores_and_person_summaries(count):
    pairs = [
        EventPair(f"g{i % 3}", f"p{i % 3}", "r0", float(i), i, i + count)
        for i in range(count)
    ]
    rows = np.zeros(2 * count, dtype=ROW_DTYPE)
    rows["available"] = True
    delta = np.zeros((2 * count, 1024), dtype=np.float32)
    # Positive, negative and tied differences straddle the4096-row batch boundary.
    delta[:count, 0] = np.arange(count) % 5 - 2
    delta[count:, 0] = np.arange(count) % 3 - 1
    tables = compare.VerifiedTables(
        {"rows.npy": rows, "delta.npy": delta, "nuisance.npy": np.zeros((2 * count, 6))},
        [dict(record_id="r0", group_id="g0", patient_id="p0")],
    )
    transforms = compare.FitTransforms(
        SimpleImputer().fit(np.zeros((2, 46))),
        StandardScaler().fit(np.zeros((2, 46))),
        StandardScaler().fit(np.zeros((2, 1024))), np.array([0, 1]), np.array([0.5, 0.5]),
    )
    model = LogisticRegression(C=0.1, fit_intercept=False, max_iter=3000, random_state=42)
    model.coef_, model.intercept_, model.classes_ = np.zeros((1, 1070)), np.zeros(1), np.array([0, 1])
    model.coef_[0, 46] = 1.0
    model.n_features_in_ = 1070
    head = compare.PointwiseHead(model, transforms, "candidate")
    features = [dict.fromkeys(LINEAR_SCHEMA, 0.0)] * (2 * count)
    prepared = SimpleNamespace(features=features, tables=tables)
    callbacks = []
    result = study._pair_diagnostics(head, prepared, pairs, lambda: callbacks.append(True))
    expected = {}
    for pair in pairs:
        difference = float(delta[pair.positive_index, 0] - delta[pair.negative_index, 0])
        expected.setdefault(pair.group_id, []).append(difference)
    assert result["person_differences"] == expected
    assert result["person_macro_difference"] == np.mean([np.mean(v) for v in expected.values()])
    assert result["person_macro_loss"] == np.mean([np.mean(np.logaddexp(0, -np.asarray(v))) for v in expected.values()])
    assert result["person_macro_concordance"] == np.mean([
        np.mean(np.where(np.asarray(v) > 0, 1, np.where(np.asarray(v) == 0, 0.5, 0)))
        for v in expected.values()
    ])
    assert len(callbacks) == 1 + (2 * count + study.BATCH_ROWS - 1) // study.BATCH_ROWS
    if count <= 241:
        indices = np.array([p.positive_index for p in pairs] + [p.negative_index for p in pairs])
        original = head.logits(features, tables, indices)
        assert np.array_equal(original[:count] - original[count:], delta[:count, 0] - delta[count:, 0])


def test_empty_pairs_do_not_score_or_call_work():
    def forbidden(*args):
        raise AssertionError("Empty diagnostic must do no scoring or work")
    result = study._pair_diagnostics(SimpleNamespace(logits=forbidden), None, [], forbidden)
    assert result["pairs"] == result["people"] == 0
    assert result["person_differences"] == {}
