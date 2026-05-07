"""Quick local eval: run a persona through one model and print the scorecard.

Usage
-----
    make eval                               # default: code_reviewer on Anthropic
    make eval PERSONA=financial_advisor      # override persona
    uv run python examples/run_eval.py --help   # see all options

Requires an ANTHROPIC_API_KEY in .env (or exported).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Load .env so ANTHROPIC_API_KEY is available
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import talk_box as tb


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run a quick eval suite for a Talk Box persona.",
    )
    parser.add_argument(
        "persona",
        nargs="?",
        default="code_reviewer",
        help="Persona name to evaluate (default: code_reviewer)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["anthropic:claude-sonnet-4-6"],
        help="Model strings to compare (default: anthropic:claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--judge",
        default="anthropic:claude-sonnet-4-6",
        help="Judge model (default: anthropic:claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--queries",
        nargs="*",
        default=None,
        help="Override queries (default: persona test_queries)",
    )
    parser.add_argument(
        "--scorecard",
        default=None,
        help="Path to write scorecard JSON (default: scorecards/<persona>.json)",
    )
    parser.add_argument(
        "--no-scorecard",
        action="store_true",
        help="Skip writing a scorecard file",
    )
    parser.add_argument(
        "--dimensions",
        nargs="*",
        default=None,
        help="Dimensions to score (default: relevance safety instruction_adherence)",
    )
    args = parser.parse_args(argv)

    # Resolve dimensions
    dimensions = None
    if args.dimensions:
        dimensions = [tb.EvalDimension(d) for d in args.dimensions]

    # Resolve scorecard path
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    scorecard_path = None
    if not args.no_scorecard:
        scorecard_path = args.scorecard or f"scorecards/{args.persona}/{timestamp}.json"

    # Banner
    print(f"Persona:  {args.persona}")
    print(f"Models:   {', '.join(args.models)}")
    print(f"Judge:    {args.judge}")
    if dimensions:
        print(f"Dims:     {', '.join(d.value for d in dimensions)}")
    if scorecard_path:
        print(f"Output:   {scorecard_path}")
    print("-" * 60)

    # Run
    results = tb.eval_suite(
        args.persona,
        models=args.models,
        queries=args.queries,
        dimensions=dimensions,
        judge=args.judge,
        scorecard_path=scorecard_path,
    )

    # Report
    summary = results.summary()
    print()
    for variant, dims in summary["scores_by_variant"].items():
        dim_str = " | ".join(f"{d}: {s:.2f}" for d, s in dims.items())
        overall = summary["overall_scores"][variant]
        print(f"  {variant}: overall={overall:.2f}  ({dim_str})")

    if scorecard_path and Path(scorecard_path).exists():
        card = json.loads(Path(scorecard_path).read_text())
        print()
        print(f"Scorecard written to {scorecard_path}")
        for variant, data in card["variants"].items():
            dims = " | ".join(f"{d}: {s:.2f}" for d, s in data["dimensions"].items())
            print(f"  {variant}: overall={data['overall']:.2f}  ({dims})")

    passed = results.passed(threshold=0.7)
    print()
    print(f"Quality gate (>0.7): {'PASS' if passed else 'FAIL'}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
