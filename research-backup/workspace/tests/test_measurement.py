import importlib.util
from pathlib import Path

import numpy as np

from cardiotwin.config import load_config
from cardiotwin.features import assess_quality, extract_features


def test_common_mode_noise_is_not_ecg_quality():
    noise = np.random.default_rng(321).normal(0, 0.01, 60000)
    row = extract_features(np.column_stack([noise, noise]), 100)
    config = load_config()
    config["quality"]["min_qrs_kurtosis"] = 5.0
    assert not assess_quality(row, config)["accepted"]
    assert assess_quality(row, config)["reason"] == "non_qrs_signal"


def test_quality_extension_preserves_frozen_v2_contract():
    noise = np.random.default_rng(321).normal(0, 0.01, 60000)
    row = extract_features(np.column_stack([noise, noise]), 100)
    assert assess_quality(row, load_config())["accepted"]


def test_rr_benchmark_does_not_bridge_extra_or_missed_beats():
    path = Path(__file__).parents[1] / "scripts" / "benchmark_beats.py"
    spec = importlib.util.spec_from_file_location("benchmark_beats", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    extra = module.match_beats([1, 2, 3], [1.1, 2.1, 2.5, 3.1])
    assert extra["matched_beats"] == 3
    assert extra["extra_beats"] == 1
    assert extra["matched_adjacent_rr_intervals"] == 1
    missed = module.match_beats([1, 2, 3, 4], [1.1, 3.1, 4.1])
    assert missed["missed_beats"] == 1
    assert missed["matched_adjacent_rr_intervals"] == 1
