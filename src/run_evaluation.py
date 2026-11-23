from __future__ import annotations

import argparse
from pathlib import Path

from evaluation_engine import EvalConfig, evaluate_run, save_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation metrics for a captured response run.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--judge-provider", default="heuristic", choices=["heuristic", "openai"])
    parser.add_argument("--judge-model", default="heuristic-v1")
    parser.add_argument("--golden", default="data/golden_dataset.csv")
    parser.add_argument("--raw", default="data/raw_responses.csv")
    parser.add_argument("--output-dir", default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EvalConfig(
        run_id=args.run_id,
        golden_path=Path(args.golden),
        raw_responses_path=Path(args.raw),
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
    )
    results = evaluate_run(config)
    path = save_results(results, run_id=args.run_id, output_dir=Path(args.output_dir))
    print(f"Saved evaluation results: {path}")


if __name__ == "__main__":
    main()
