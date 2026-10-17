"""Reproducible offline pipeline and local replay entry point."""

import argparse
import json
import sys
from pathlib import Path

from .config import load_config, validate_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="CardioTwin research pipeline — historical ECG simulation")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Generate manufactured ECG/profile fixtures (not clinical evidence)")
    demo.add_argument("--output", type=Path, default=Path("data/demo"))
    download = sub.add_parser(
        "download", help="Download official SHDB-AF files and verify published checksums"
    )
    download.add_argument("--output", type=Path, default=Path("data/shdb-af"))
    download.add_argument("--records", nargs="+")
    download.add_argument("--metadata-only", action="store_true")
    audit = sub.add_parser("audit", help="Audit cohort, timing, signal quality and target feasibility")
    train = sub.add_parser("train", help="Develop/calibrate/select and freeze a model; never evaluate test")
    evaluate = sub.add_parser("evaluate", help="Open final test only after frozen development artifact")
    serve = sub.add_parser("serve", help="Run local historical ECG replay API/dashboard")
    for command in [demo, audit, train, serve]:
        command.add_argument("--config", type=Path)
    for command in [audit, train, evaluate, serve]:
        command.add_argument("--data-dir", type=Path, default=Path("data/demo"))
    for command in [audit, train, evaluate]:
        command.add_argument("--output", type=Path, required=True)
    for command in [train, evaluate]:
        command.add_argument("--audit", type=Path, required=True)
    train.add_argument("--synthetic", action="store_true")
    train.add_argument("--false-warning-budget", type=float)
    train.add_argument("--minimum-sensitivity", type=float)
    evaluate.add_argument("--model", type=Path, required=True)
    serve.add_argument("--model", type=Path)
    serve.add_argument("--metrics", type=Path)
    serve.add_argument(
        "--research-model",
        action="store_true",
        help="Allow a model package that failed deployment gates (manifest deployment_ready=false); requires --model",
    )
    serve.add_argument("--db", type=Path, default=Path("data/state.sqlite"))
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument(
        "--playback-seconds", type=float, default=1.0, help="Wall seconds per historical cadence"
    )
    args = parser.parse_args(argv)
    try:
        config = load_config(getattr(args, "config", None))
        if args.command == "demo":
            from .data import make_demo

            result = make_demo(args.output, config)
        elif args.command == "download":
            from .data import download_shdb

            result = download_shdb(args.output, args.records, args.metadata_only)
        elif args.command == "audit":
            from .audit import audit_dataset

            result = audit_dataset(args.data_dir, args.output, config)
        elif args.command == "train":
            from .train import train_dataset

            if args.false_warning_budget is not None:
                config["alert"]["max_false_warnings_per_hour"] = args.false_warning_budget
            if args.minimum_sensitivity is not None:
                config["alert"]["min_event_sensitivity"] = args.minimum_sensitivity
            validate_config(config)
            result = train_dataset(args.data_dir, args.audit, args.output, config, args.synthetic)
        elif args.command == "evaluate":
            from .evaluation import evaluate_artifact

            result = evaluate_artifact(args.data_dir, args.audit, args.model, args.output)
        else:
            import uvicorn

            from .api import create_app
            from .service import TwinService

            if args.research_model and not args.model:
                raise ValueError("--research-model requires --model")
            if args.model:
                frozen = load_config(args.model / "config.json")
                if args.config and config != frozen:
                    raise ValueError("Explicit runtime config differs from frozen artifact")
                config = frozen
            service = TwinService(args.data_dir, args.db, config, args.model, args.metrics, args.research_model)
            uvicorn.run(
                create_app(service, playback_seconds=args.playback_seconds), host=args.host, port=args.port
            )
            return 0
        print(json.dumps(result, indent=2, allow_nan=False))
        if args.command == "audit" and not result.get("feasible"):
            return 2
        if args.command == "train" and result.get("status") == "constraint_not_met":
            return 2
        return 0
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"CardioTwin: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
