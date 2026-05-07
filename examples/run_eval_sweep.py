"""Run eval_suite across multiple personas and models, producing a summary.

Usage
-----
    make eval-sweep                         # default personas + models
    make eval-sweep SWEEP_MODELS="github:gpt-4o"  # override models
    uv run python examples/run_eval_sweep.py --help

Requires ANTHROPIC_API_KEY and/or GITHUB_TOKEN in .env.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import talk_box as tb

# Representative personas spanning different categories
DEFAULT_PERSONAS = [
    "code_reviewer",
    "financial_advisor",
    "customer_support_tier1",
    "data_analyst",
    "technical_writer",
]

DEFAULT_MODELS = [
    "anthropic:claude-sonnet-4-6",
    "github:gpt-4o",
]


def run_sweep(
    personas: list[str],
    models: list[str],
    judge: str,
    output_dir: str,
    threshold: float,
) -> bool:
    """Run eval_suite for each persona. Returns True if all pass."""
    out = Path(output_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")

    all_passed = True
    results_summary: list[dict] = []
    total_start = time.time()

    for i, persona in enumerate(personas, 1):
        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(personas)}] {persona}")
        print(f"{'=' * 60}")

        persona_dir = out / persona
        persona_dir.mkdir(parents=True, exist_ok=True)
        scorecard_path = persona_dir / f"{timestamp}.json"

        try:
            start = time.time()
            results = tb.eval_suite(
                persona,
                models=models,
                judge=judge,
                scorecard_path=scorecard_path,
            )
            elapsed = time.time() - start

            summary = results.summary()
            passed = results.passed(threshold=threshold)

            if not passed:
                all_passed = False

            # Print per-variant scores
            for variant, dims in summary["scores_by_variant"].items():
                dim_str = " | ".join(f"{d}: {s:.2f}" for d, s in dims.items())
                overall = summary["overall_scores"][variant]
                status = "PASS" if overall >= threshold else "FAIL"
                print(f"  [{status}] {variant}: {overall:.2f}  ({dim_str})")

            results_summary.append(
                {
                    "persona": persona,
                    "passed": passed,
                    "elapsed": round(elapsed, 1),
                    "scores": summary["overall_scores"],
                }
            )

        except Exception as e:
            print(f"  ERROR: {e}")
            all_passed = False
            results_summary.append(
                {
                    "persona": persona,
                    "passed": False,
                    "elapsed": 0,
                    "error": str(e),
                }
            )

    # Final summary
    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"SWEEP SUMMARY  ({total_elapsed:.0f}s total)")
    print(f"{'=' * 60}")

    for entry in results_summary:
        status = "PASS" if entry["passed"] else "FAIL"
        if "error" in entry:
            print(f"  [{status}] {entry['persona']}: ERROR - {entry['error']}")
        else:
            scores_str = ", ".join(f"{m}: {s:.2f}" for m, s in entry["scores"].items())
            print(f"  [{status}] {entry['persona']}: {scores_str}  ({entry['elapsed']}s)")

    passed_count = sum(1 for e in results_summary if e["passed"])
    total_count = len(results_summary)
    print(f"\n{passed_count}/{total_count} personas passed (threshold={threshold})")

    # Write combined summary
    sweeps_dir = out / "_sweeps"
    sweeps_dir.mkdir(parents=True, exist_ok=True)
    summary_path = sweeps_dir / f"{timestamp}.json"
    sweep_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": models,
        "judge": judge,
        "threshold": threshold,
        "elapsed_seconds": round(total_elapsed, 1),
        "passed": passed_count,
        "total": total_count,
        "results": results_summary,
    }
    summary_path.write_text(json.dumps(sweep_data, indent=2), encoding="utf-8")
    print(f"\nSweep summary written to {summary_path}")

    return all_passed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run eval sweep across multiple personas and models.",
    )
    parser.add_argument(
        "--personas",
        nargs="+",
        default=DEFAULT_PERSONAS,
        help=f"Personas to evaluate (default: {' '.join(DEFAULT_PERSONAS)})",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=f"Models to compare (default: {' '.join(DEFAULT_MODELS)})",
    )
    parser.add_argument(
        "--judge",
        default="anthropic:claude-sonnet-4-6",
        help="Judge model (default: anthropic:claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--output-dir",
        default="scorecards",
        help="Directory for scorecard JSON files (default: scorecards/)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Minimum passing score (default: 0.7)",
    )
    args = parser.parse_args(argv)

    passed = run_sweep(
        personas=args.personas,
        models=args.models,
        judge=args.judge,
        output_dir=args.output_dir,
        threshold=args.threshold,
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
