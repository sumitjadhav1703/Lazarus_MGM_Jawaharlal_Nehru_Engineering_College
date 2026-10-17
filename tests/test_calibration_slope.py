from types import SimpleNamespace

import numpy as np
import pytest

from cardiotwin.train import _calibrate


def _rows(scores, labels):
    return [
        {"accepted": True, "label": int(y), "features": {"scalar": float(x)}}
        for x, y in zip(scores, labels, strict=True)
    ]


_PIPELINE = SimpleNamespace(decision_function=lambda x: x[:, 0])
_LABELS = [0, 0, 0, 1, 1, 1]
_SCORES = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]


def test_anti_predictive_scores_block_calibrator_export():
    rows = _rows(_SCORES, _LABELS[::-1])  # high score -> negative class: slope < 0
    with pytest.raises(ValueError, match="slope is not positive"):
        _calibrate(_PIPELINE, ["scalar"], rows, 42, calibration_c=100.0)


def test_predictive_scores_pass_slope_guard():
    calibrator = _calibrate(_PIPELINE, ["scalar"], _rows(_SCORES, _LABELS), 42, calibration_c=100.0)
    assert calibrator["sigmoid"].coef_[0][0] > 0
    assert np.isfinite(calibrator.predict_proba(np.array([[0.0]]))).all()
