import importlib.util
from pathlib import Path


def test_calibration_selection_uses_development_metric_and_keeps_group_balancing():
    spec = importlib.util.spec_from_file_location(
        "experiments", Path(__file__).parents[1] / "scripts/run_ml_experiments.py"
    )
    experiments = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiments)

    def regime(weighting, brier, status="frozen"):
        return {
            "training": {"sample_weighting": weighting},
            "status": status,
            "validation_policy": {
                "probability": {"group_macro_brier": brier, "patient_macro_brier": 1 - brier}
            },
        }

    chosen = regime("patient_class", 0.02)
    choices = [
        regime("window_class", 0.001),
        regime("patient_class", 0.01, "failed"),
        regime("patient_class", 0.3),
        chosen,
    ]
    assert experiments.select_calibration(choices) is chosen
    assert experiments.select_calibration([]) is None
