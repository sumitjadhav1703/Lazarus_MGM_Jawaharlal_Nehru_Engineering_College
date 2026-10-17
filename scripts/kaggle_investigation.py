#!/usr/bin/env python3
"""Portable private Kaggle entrypoint using a packaged project and prepared data.

Upload the repository/source wheel, frozen development split, config and cache as
private inputs. This script never uploads data or creates a public Kaggle kernel.
Authentication and job submission remain connector operations, not credential
handling inside the training program. Public sources retain their own licenses.
"""

import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Prepared source-aware IRIDIA directory")
    parser.add_argument("--cache", required=True, help="Complete development-only feature cache")
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", default="/kaggle/working/cardiotwin-neural")
    parser.add_argument("--max-hours", type=float, default=6)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--training-windows", type=int, default=10000)
    parser.add_argument("--models", nargs="+", choices=("rr16", "waveform16", "waveform_resnet16", "rr32"))
    parser.add_argument(
        "--early-stop-weighting", choices=("patient_only", "patient_class"), default="patient_only"
    )
    args = parser.parse_args()
    if not 0 < args.max_hours <= 6:
        raise ValueError("The authorized GPU budget is at most six hours")
    from cardiotwin.gpu_telemetry import GpuTelemetry
    from cardiotwin.progress import Progress

    with Progress("cuda_check") as progress:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Kaggle GPU acceleration is unavailable; request a GPU for the private job")
        progress.finish(
            available_devices=torch.cuda.device_count(),
            scheduled_device="cuda:0",
            gpu_name=torch.cuda.get_device_name(0),
        )
    from cardiotwin.data import list_records
    from cardiotwin.investigation import read_cache, validate_live_inputs
    from cardiotwin.neural import run_neural_ladder

    with GpuTelemetry():
        with Progress("development_cache_loading") as progress:
            rows, events, index = read_cache(args.cache)
            progress.finish(rows=len(rows), records=len(index["records"]))
        with Progress("development_membership") as progress:
            split = json.loads(Path(args.split).read_text())
            if set(index["allowed_groups"]) != set(split["development_groups"]):
                raise ValueError("Only the frozen full development cohort can be supplied to GPU screening")
            records = [
                record
                for record in list_records(args.data)
                if split["group_ids"][record.patient_id] in set(split["development_groups"])
            ]
            progress.finish(records=len(records), groups=len(index["allowed_groups"]))
        validate_live_inputs(records, index)
        started = time.monotonic()
        with Progress(
            "neural_screening",
            max_epochs=args.epochs,
            max_fitting_windows=args.training_windows,
            outer_folds=3,
        ) as progress:
            result = run_neural_ladder(
                rows,
                events,
                records,
                index["config"],
                args.output,
                max_hours=args.max_hours,
                seed=split["seed"],
                max_training_windows=args.training_windows,
                screening_epochs=args.epochs,
                specs=[
                    {
                        "rr16": ("rr", 16),
                        "waveform16": ("waveform", 16),
                        "waveform_resnet16": ("waveform_resnet", 16),
                        "rr32": ("rr", 32),
                    }[name]
                    for name in args.models
                ]
                if args.models
                else None,
                early_stop_weighting=args.early_stop_weighting,
                source_contract=index["input_sha256"],
                cache_contract={key: index[key] for key in ("rows_sha256", "input_sha256", "sha256")},
            )
            progress.finish(result_status=result["status"], complete_candidates=sorted(result["combined"]))
    print(
        json.dumps(
            {"summary": result, "job_wall_seconds": time.monotonic() - started}, indent=2, allow_nan=False
        )
    )


if __name__ == "__main__":
    main()
