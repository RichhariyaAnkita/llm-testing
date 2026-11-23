from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from response_capture import CaptureConfig, append_manual_response, pending_questions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture UI chatbot responses for a run.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--collector", default="manual")
    parser.add_argument("--app-url", default="https://llm-rag-testing-workshop.streamlit.app/")
    parser.add_argument("--golden", default="data/golden_dataset.csv")
    parser.add_argument("--raw", default="data/raw_responses.csv")
    parser.add_argument("--show-pending", action="store_true")
    parser.add_argument("--question-id")
    parser.add_argument("--actual-answer")
    parser.add_argument("--retrieved-context", default="")
    parser.add_argument("--session-id", default="")
    parser.add_argument("--turn-index", type=int, default=1)
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    golden_path = Path(args.golden)
    raw_path = Path(args.raw)

    if args.show_pending:
        pending = pending_questions(golden_path, raw_path, args.run_id)
        pd.set_option("display.max_colwidth", 120)
        print(pending.to_string(index=False))
        return

    if not args.question_id or not args.actual_answer:
        raise SystemExit("When not using --show-pending, both --question-id and --actual-answer are required.")

    golden = pd.read_csv(golden_path)
    question_row = golden.loc[golden["id"].astype(str) == str(args.question_id)]
    if question_row.empty:
        raise SystemExit(f"Question ID {args.question_id} not found in golden dataset.")

    question = str(question_row.iloc[0]["question"])
    category = str(question_row.iloc[0]["category"])

    response_id = append_manual_response(
        raw_path,
        run_id=args.run_id,
        config=CaptureConfig(app_url=args.app_url, collector=args.collector),
        question_id=str(args.question_id),
        question=question,
        category=category,
        actual_answer=args.actual_answer,
        retrieved_context=args.retrieved_context,
        session_id=args.session_id,
        turn_index=args.turn_index,
        notes=args.notes,
    )
    print(f"Saved response {response_id}")


if __name__ == "__main__":
    main()
