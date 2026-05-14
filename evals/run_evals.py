"""
CLI entrypoint for running evals.

Usage:
    # Run all cases
    uv run python -m evals.run_evals

    # Run a single case
    uv run python -m evals.run_evals --year 2023 --gp Bahrain --driver VER

    # Run by circuit type
    uv run python -m evals.run_evals --circuit high_deg
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
from evals.dataset import EVAL_DATASET, get_case, get_cases_by_circuit_type
from evals.runner import run_all, run_single
from evals.report import print_report


def main():
    parser = argparse.ArgumentParser(description="F1 Strategy Optimizer — Eval Runner")
    parser.add_argument("--year",    type=int, help="Filter by year")
    parser.add_argument("--gp",      type=str, help="Filter by Grand Prix")
    parser.add_argument("--driver",  type=str, help="Filter by driver")
    parser.add_argument("--circuit", type=str, help="Filter by circuit type: high_deg, low_deg, mixed")
    args = parser.parse_args()

    # ── Single case ────────────────────────────────────────────────────────
    if args.year and args.gp and args.driver:
        case = get_case(args.year, args.gp, args.driver)
        if not case:
            print(f"No eval case found for {args.driver} {args.gp} {args.year}")
            return
        result = run_single(case)
        if result:
            print_report([result])
        return

    # ── Filter by circuit type ─────────────────────────────────────────────
    if args.circuit:
        cases = get_cases_by_circuit_type(args.circuit)
        if not cases:
            print(f"No cases found for circuit type: {args.circuit}")
            return
        scores = run_all(cases)
        print_report(scores)
        return

    # ── Run all ────────────────────────────────────────────────────────────
    scores = run_all(EVAL_DATASET)
    print_report(scores)


if __name__ == "__main__":
    main()