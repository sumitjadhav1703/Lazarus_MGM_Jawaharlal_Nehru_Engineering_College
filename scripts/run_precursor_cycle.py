"""Explicit local commands for the fixed, development-only precursor protocol."""

import argparse
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

from cardiotwin.config import load_config  # noqa: E402
from cardiotwin.precursor import review_comparison, run_diagnostic, run_linear_comparison  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "linear"):
        command = commands.add_parser(name)
        for field in ("cache", "split", "config", "output"):
            command.add_argument(f"--{field}", required=True)
        command.add_argument("--metadata-root" if name == "audit" else "--audit", required=True)
        command.add_argument("--max-seconds", type=float, default=41100)
    review = commands.add_parser("review")
    review.add_argument("--run", required=True)
    review.add_argument("--output", required=True)
    review.add_argument("--bootstrap-samples", type=int, default=10000)
    review.add_argument("--seed", type=int, default=42)
    review.add_argument("--retrospective-provenance", action="store_true",
                        help="Evaluate a legacy completed run without upgrading its missing fit-time compact provenance")
    args = parser.parse_args(argv)
    if args.command == "review":
        if args.seed != 42 or args.bootstrap_samples != 10000:
            parser.error("The predeclared review requires seed 42 and 10000 bootstrap draws")
        receipt = review_comparison(args.run, args.output, seed=args.seed, bootstrap_samples=args.bootstrap_samples,
                                    retrospective_provenance=args.retrospective_provenance)
    else:
        if args.max_seconds <= 0 or args.max_seconds > 41100:
            parser.error("max-seconds must be positive and at most 41100")
        deadline = time.monotonic() + args.max_seconds
        if args.command == "audit":
            receipt = run_diagnostic(args.cache, args.split, args.metadata_root, load_config(args.config), args.output, deadline=deadline)
        else:
            receipt = run_linear_comparison(args.cache, args.split, args.audit, load_config(args.config), args.output, deadline=deadline)
    print(receipt["status"], flush=True)
    return 0 if receipt["status"] in {"passed", "complete"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
