from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import app export CSV into raw_responses.csv format.")
    parser.add_argument("--export", required=True, help="Path to app export CSV (question, contexts, answer, ground_truth)")
    parser.add_argument("--run-id", required=True, help="Run id in YYYYMMDD_HHMMSS format")
    parser.add_argument("--golden", default="data/golden_dataset.csv")
    parser.add_argument("--raw", default="data/raw_responses.csv")
    parser.add_argument("--app-url", default="https://llm-rag-testing-workshop.streamlit.app/Application")
    parser.add_argument("--collector", default="app-export")
    parser.add_argument("--model-hint", default="streamlit-chatbot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    export_df = pd.read_csv(Path(args.export))
    golden_df = pd.read_csv(Path(args.golden))
    raw_path = Path(args.raw)

    required_export = {"question", "contexts", "answer"}
    missing_export = required_export - set(export_df.columns)
    if missing_export:
        raise ValueError(f"Export missing required columns: {sorted(missing_export)}")

    required_golden = {"id", "category", "question"}
    missing_golden = required_golden - set(golden_df.columns)
    if missing_golden:
        raise ValueError(f"Golden dataset missing required columns: {sorted(missing_golden)}")

    mapping = golden_df[["id", "category", "question"]].rename(columns={"id": "question_id"})
    merged = export_df.merge(mapping, on="question", how="left")

    unmatched = merged[merged["question_id"].isna()]
    if not unmatched.empty:
        print(f"Warning: {len(unmatched)} rows had questions not found in golden dataset and will be skipped.")
        merged = merged[merged["question_id"].notna()].copy()

    if raw_path.exists():
        raw_df = pd.read_csv(raw_path)
    else:
        raw_df = pd.DataFrame(
            columns=[
                "response_id",
                "run_id",
                "timestamp_utc",
                "app_url",
                "question_id",
                "question",
                "category",
                "actual_answer",
                "retrieved_context",
                "session_id",
                "turn_index",
                "model_hint",
                "collector",
                "notes",
            ]
        )

    # Replace any existing rows for this run id to keep imports idempotent.
    raw_df = raw_df[raw_df["run_id"].astype(str) != str(args.run_id)].copy()

    imported_rows = []
    for idx, row in merged.reset_index(drop=True).iterrows():
        question_id = str(row["question_id"])
        response_id = f"{args.run_id}-{question_id}-{idx + 1:04d}"
        imported_rows.append(
            {
                "response_id": response_id,
                "run_id": args.run_id,
                "timestamp_utc": utc_now_iso(),
                "app_url": args.app_url,
                "question_id": question_id,
                "question": str(row["question"]),
                "category": str(row["category"]),
                "actual_answer": "" if pd.isna(row["answer"]) else str(row["answer"]),
                "retrieved_context": "" if pd.isna(row["contexts"]) else str(row["contexts"]),
                "session_id": args.run_id,
                "turn_index": idx + 1,
                "model_hint": args.model_hint,
                "collector": args.collector,
                "notes": "imported from app export csv",
            }
        )

    updated = pd.concat([raw_df, pd.DataFrame(imported_rows)], ignore_index=True)
    updated.to_csv(raw_path, index=False)
    print(f"Imported {len(imported_rows)} rows into {raw_path}")


if __name__ == "__main__":
    main()
