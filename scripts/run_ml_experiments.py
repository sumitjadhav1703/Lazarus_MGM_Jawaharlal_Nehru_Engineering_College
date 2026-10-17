"""Predeclared development ablations; does not open final-test features or labels."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

from cardiotwin.config import load_config, validate_config
from cardiotwin.train import train_dataset

REGIMES = (
    ("window_class", "window_class", 100.0),
    ("patient_class", "patient_class", 100.0),
    ("patient_class_regularized", "patient_class", 1.0),
)


def select_calibration(regimes):
    """Keep group-balanced training; compare calibration strength on validation only."""
    eligible = [
        r for r in regimes if r["status"] == "frozen" and r["training"]["sample_weighting"] == "patient_class"
    ]
    return min(
        eligible,
        key=lambda r: r["validation_policy"]["probability"]["group_macro_brier"],
        default=None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/demo"))
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--false-warning-budget", type=float, default=1.0)
    parser.add_argument("--minimum-sensitivity", type=float, default=0.5)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Experiment output exists; use a fresh result path")
    if args.artifacts.exists() and any(args.artifacts.iterdir()):
        raise ValueError("Experiment artifact directory must be fresh; previous releases are preserved")
    config = load_config(args.config)
    config["alert"].update(
        max_false_warnings_per_hour=args.false_warning_budget, min_event_sensitivity=args.minimum_sensitivity
    )
    results = []
    for name, weighting, calibration_c in REGIMES:
        candidate = deepcopy(config)
        candidate["training"].update(sample_weighting=weighting, calibration_c=calibration_c)
        validate_config(candidate)
        release = args.artifacts / name
        result = train_dataset(args.data_dir, args.audit, release, candidate, synthetic=args.synthetic)
        results.append(
            {
                "regime": name,
                "artifact": str(release),
                "training": candidate["training"],
                "status": result["status"],
                "selected_model": result.get("selected_model"),
                "validation_policy": result.get("validation_policy"),
                "comparisons": [
                    {k: c.get(k) for k in ["model", "status", "reason", "chosen", "selected_policy"]}
                    for c in result["comparisons"]
                ],
            }
        )
    recommended = select_calibration(results)
    report = {
        "selection_protocol": "Group-balanced training retained; candidate/threshold uses validation. Calibration C is selected by validation independent-group-macro Brier among the two group-balanced regimes.",
        "test_opened_for_selection": False,
        "synthetic": args.synthetic,
        "regimes": results,
        "recommended_artifact": recommended["artifact"] if recommended else None,
        "protocol_revision": "C1 was the initial prespecified hypothesis; after worse development calibration, C100 was retained before final-test evaluation. This adds validation selection and is disclosed, not an untouched protocol claim.",
        "limitations": [
            "Three regimes and ten models have high model-selection uncertainty on tiny groups.",
            "Synthetic success is a software check; no clinical forecasting gain is implied.",
            "All parameter choices are development hypotheses; an untouched real cohort is still required.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in report.items() if k != "regimes"}, indent=2))


if __name__ == "__main__":
    main()
