#!/usr/bin/env python3
"""Explicit development cache, nested ladder, freeze, then held-out evaluation."""

import argparse
import json
import os
import sys
from pathlib import Path

# macOS Torch/XGBoost libomp interoperability: https://github.com/pytorch/pytorch/issues/155201
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

from cardiotwin.config import load_config  # noqa: E402
from cardiotwin.data import list_records, read_intervals  # noqa: E402
from cardiotwin.investigation import (  # noqa: E402
    _extraction_contract,
    _write,
    audit_development,
    cache_records,
    evaluate_frozen,
    freeze_finalist,
    make_primary_split,
    read_cache,
    run_ladder,
    select_reports,
    validate_live_inputs,
)
from cardiotwin.labels import eligible_onsets  # noqa: E402


def sequence_cache_capacity(cache, output, limit=25_000_000_000, reserve=5_000_000_000):
    """Keep the optional sequence cache inside the investigation storage budget."""
    roots = {Path(cache).resolve().parent, Path(output).resolve().parent}
    roots = {
        root for root in roots if not any(root != other and root.is_relative_to(other) for other in roots)
    }
    occupied = sum(path.stat().st_size for root in roots for path in root.rglob("*") if path.is_file())
    return max(0, min(4_000_000_000, limit - occupied - reserve))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    split = commands.add_parser("split")
    split.add_argument("--data", required=True)
    split.add_argument("--groups", help="JSON patient-ID to independently audited group-ID mapping")
    split.add_argument("--config")
    split.add_argument("--output", required=True)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument(
        "--development-only-patients",
        nargs="*",
        default=[],
        help="Previously inspected patients, declared before modeling",
    )
    cache = commands.add_parser("cache")
    cache.add_argument("--data", required=True)
    cache.add_argument("--split", required=True)
    cache.add_argument("--partition", choices=("development", "test"), default="development")
    cache.add_argument("--artifact", help="Required before opening primary test signals")
    cache.add_argument("--config")
    cache.add_argument("--output", required=True)
    cache.add_argument("--max-hours", type=float, default=8)
    cache.add_argument("--motifs", action="store_true")
    cache.add_argument("--resume", action="store_true", help="Resume only checksum-bound completed records")
    cache.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=2)
    audit = commands.add_parser("audit")
    audit.add_argument("--cache", required=True)
    audit.add_argument("--split", required=True)
    audit.add_argument("--output", required=True)
    ladder = commands.add_parser("ladder")
    ladder.add_argument("--cache", required=True)
    ladder.add_argument("--split", required=True)
    ladder.add_argument("--output", required=True)
    ladder.add_argument("--trials", type=int, default=6)
    ladder.add_argument("--max-hours", type=float, default=8)
    ladder.add_argument("--fitting-cadence-seconds", type=int, choices=(30, 60, 120), default=120)
    neural = commands.add_parser("neural")
    neural.add_argument("--cache", required=True)
    neural.add_argument("--split", required=True)
    neural.add_argument("--data", required=True)
    neural.add_argument("--output", required=True)
    neural.add_argument("--max-hours", type=float, default=6)
    neural.add_argument("--epochs", type=int, default=8)
    neural.add_argument("--training-windows", type=int, default=10000)
    neural.add_argument("--models", nargs="+", choices=("rr16", "waveform16", "waveform_resnet16", "rr32"))
    neural.add_argument(
        "--early-stop-weighting", choices=("patient_only", "patient_class"), default="patient_only"
    )
    neural.add_argument(
        "--sequence-cache", type=Path, help="Reuse a contract-matching RR cache with fresh outputs"
    )
    select = commands.add_parser("select")
    select.add_argument("--summary", required=True, help="Classical development summary")
    select.add_argument("--neural", required=True, help="Neural development summary")
    select.add_argument("--split", required=True)
    select.add_argument("--output", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--cache", required=True)
    freeze.add_argument("--split", required=True)
    freeze.add_argument("--summary", required=True)
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--trials", type=int, default=6)
    freeze.add_argument(
        "--research-baseline",
        action="store_true",
        help="Freeze the exact current recipe only when no development candidate qualifies; never deploy it",
    )
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--artifact", required=True)
    evaluate.add_argument("--cache", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--cohort", choices=("primary", "external"), default="primary")
    args = parser.parse_args()
    if args.command == "split":
        records, config = list_records(args.data), load_config(args.config)
        groups = json.loads(Path(args.groups).read_text()) if args.groups else None
        events = [
            {"patient_id": r.patient_id, "record_id": r.record_id, "onset": onset}
            for r in records
            for onset in eligible_onsets(
                read_intervals(r, n_is_other=config["audit"].get("n_rhythm_is_other", False)),
                r.duration_seconds,
                config,
            )
        ]
        if Path(args.output).exists():
            raise ValueError("Preserve existing primary split; choose a fresh output")
        audited_groups = groups or {record.patient_id: record.patient_id for record in records}
        if not set(args.development_only_patients) <= set(audited_groups):
            raise ValueError("Previously inspected patient must exist in the included cohort")
        forced = {audited_groups[patient] for patient in args.development_only_patients}
        _write(args.output, make_primary_split(records, events, audited_groups, args.seed, forced))
        result = {"split": args.output, "eligible_events": len(events)}
    elif args.command == "cache":
        config, split = load_config(args.config), json.loads(Path(args.split).read_text())
        if args.partition == "test":
            if not args.artifact:
                raise ValueError("Freeze a finalist before opening primary test signals")
            from cardiotwin.modeling import Predictor

            frozen = Predictor.load(Path(args.artifact), research=True)
            manifest = frozen.manifest
            if manifest["primary_split"] != split or manifest["cohort_contract"]["primary_split"] != split:
                raise ValueError("Primary split differs from frozen finalist")
            if _extraction_contract(config) != manifest["cohort_contract"]["extraction"]:
                raise ValueError("Primary extraction recipe differs from frozen finalist")
        result = cache_records(
            list_records(args.data),
            config,
            split["group_ids"],
            args.output,
            split[args.partition + "_groups"],
            args.max_hours,
            args.motifs,
            args.workers,
            resume=args.resume,
        )
        result = {key: result[key] for key in ("row_count", "wall_seconds", "rows_sha256")}
    elif args.command == "select":
        from cardiotwin.investigation import _digest

        classical, neural = [json.loads(Path(path).read_text()) for path in (args.summary, args.neural)]
        split = json.loads(Path(args.split).read_text())
        for summary in (classical, neural):
            if summary.get("sha256") != _digest({k: v for k, v in summary.items() if k != "sha256"}):
                raise ValueError("Development summary checksum mismatch")
        for key in ("seed", "extraction_contract", "development_cache_contract"):
            if classical[key] != neural[key]:
                raise ValueError("Classical and neural development recipe differs")
        if classical["seed"] != split["seed"]:
            raise ValueError("Development seed differs from the frozen primary split")
        combined = {**classical["combined"], **neural["combined"]}
        for report in combined.values():
            observed = {split["group_ids"][p["patient_id"]] for p in report["per_patient"]}
            if observed != set(split["development_groups"]):
                raise ValueError("Candidate outer coverage differs from frozen development groups")
        result = dict(
            classical, combined=combined, **select_reports(combined, split["group_ids"], split["seed"])
        )
        result.update(
            research_winner=result["selected_candidate"],
            source_summaries_sha256={"classical": classical["sha256"], "neural": neural["sha256"]},
            complete_outer_models=sorted(combined),
            neural_screening={
                key: neural[key] for key in ("specs", "screening_epochs", "screening_max_training_windows")
            },
            status="complete"
            if classical["status"] == neural["status"] == "complete"
            else "budget_exhausted",
            wall_seconds=classical["wall_seconds"] + neural["wall_seconds"],
            failed_experiments=classical.get("failed_experiments", []) + neural.get("failures", []),
        )
        result["acceptance_gate_passed"] = bool(
            result["selected_candidate"]
            and combined[result["selected_candidate"]]["event_sensitivity"]
            >= result["acceptance_gate_threshold"]
        )
        if Path(args.output).exists():
            raise ValueError("Preserve previous selection; choose a fresh output")
        result["sha256"] = _digest({k: v for k, v in result.items() if k != "sha256"})
        _write(args.output, result)
    elif args.command in ("audit", "ladder", "freeze", "neural"):
        rows, events, index = read_cache(args.cache)
        split = json.loads(Path(args.split).read_text())
        if any(
            item["input_contract"] != split["record_contracts"].get(item["record"]["record_id"])
            for item in index["records"]
        ):
            raise ValueError("Development inputs differ from the primary split acquisition identities")
        if set(index["allowed_groups"]) != set(split["development_groups"]):
            raise ValueError("The full development cohort must match the frozen split")
        development_contract = {key: index[key] for key in ("rows_sha256", "input_sha256", "sha256")}
        if args.command == "audit":
            result = audit_development(rows, events, index["config"], split, index, args.output)
        elif args.command == "ladder":
            result = run_ladder(
                rows,
                events,
                index["config"],
                args.output,
                args.max_hours,
                split["seed"],
                args.trials,
                fitting_cadence_seconds=args.fitting_cadence_seconds,
                cache_contract=development_contract,
            )
        elif args.command == "neural":
            from cardiotwin.neural import run_neural_ladder

            records = [
                record
                for record in list_records(args.data)
                if split["group_ids"][record.patient_id] in set(split["development_groups"])
            ]
            validate_live_inputs(records, index)
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
                cache_contract=development_contract,
                max_sequence_cache_bytes=sequence_cache_capacity(args.cache, args.output),
                sequence_cache_path=args.sequence_cache,
            )
        else:
            summary = json.loads(Path(args.summary).read_text())
            if not summary["selected_candidate"] and not args.research_baseline:
                raise ValueError(
                    "No development-selected finalist; explicit --research-baseline can preserve a failed comparison"
                )
            result = freeze_finalist(
                rows,
                events,
                index["config"],
                split,
                "logistic_current_recipe" if args.research_baseline else summary["selected_candidate"],
                args.output,
                split["seed"],
                args.trials,
                fitting_cadence_seconds=summary["fitting_cadence_seconds"],
                selection_summary=summary,
                development_contract=development_contract,
                research_baseline=args.research_baseline,
            )
    else:
        rows, events, index = read_cache(args.cache)
        result = evaluate_frozen(args.artifact, rows, events, args.output, args.cohort, index)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
