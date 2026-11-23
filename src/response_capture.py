from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


RAW_RESPONSE_COLUMNS = [
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


@dataclass(frozen=True)
class CaptureConfig:
    app_url: str
    collector: str
    model_hint: str = "ui-manual"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_raw_responses_file(path: Path) -> None:
    if path.exists():
        return
    pd.DataFrame(columns=RAW_RESPONSE_COLUMNS).to_csv(path, index=False)


def load_golden_questions(golden_dataset_path: Path) -> pd.DataFrame:
    df = pd.read_csv(golden_dataset_path)
    required = {"id", "category", "question"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Golden dataset missing required columns: {sorted(missing)}")
    return df[["id", "category", "question"]].rename(columns={"id": "question_id"})


def load_raw_responses(raw_responses_path: Path) -> pd.DataFrame:
    ensure_raw_responses_file(raw_responses_path)
    df = pd.read_csv(raw_responses_path)
    missing = set(RAW_RESPONSE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Raw responses file missing required columns: {sorted(missing)}")
    return df


def pending_questions(golden_dataset_path: Path, raw_responses_path: Path, run_id: str) -> pd.DataFrame:
    golden = load_golden_questions(golden_dataset_path)
    raw = load_raw_responses(raw_responses_path)
    answered = set(raw.loc[raw["run_id"] == run_id, "question_id"].astype(str))
    return golden[~golden["question_id"].astype(str).isin(answered)].copy()


def append_manual_response(
    raw_responses_path: Path,
    *,
    run_id: str,
    config: CaptureConfig,
    question_id: str,
    question: str,
    category: str,
    actual_answer: str,
    retrieved_context: Optional[str] = None,
    session_id: Optional[str] = None,
    turn_index: int = 1,
    notes: Optional[str] = None,
) -> str:
    df = load_raw_responses(raw_responses_path)
    response_id = f"{run_id}-{question_id}-{len(df) + 1:04d}"

    record = {
        "response_id": response_id,
        "run_id": run_id,
        "timestamp_utc": utc_now_iso(),
        "app_url": config.app_url,
        "question_id": question_id,
        "question": question,
        "category": category,
        "actual_answer": actual_answer,
        "retrieved_context": retrieved_context or "",
        "session_id": session_id or "",
        "turn_index": int(turn_index),
        "model_hint": config.model_hint,
        "collector": config.collector,
        "notes": notes or "",
    }

    updated = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    updated.to_csv(raw_responses_path, index=False)
    return response_id
