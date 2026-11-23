from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd
from dotenv import load_dotenv


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def overlap_ratio(a: str, b: str) -> float:
    a_tokens = tokenize(a)
    b_tokens = tokenize(b)
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


@dataclass
class EvalConfig:
    run_id: str
    golden_path: Path = Path("data/golden_dataset.csv")
    raw_responses_path: Path = Path("data/raw_responses.csv")
    app_url: str = "https://llm-rag-testing-workshop.streamlit.app/"
    judge_provider: str = "heuristic"
    judge_model: str = "heuristic-v1"
    relevance_threshold: float = 0.80
    faithfulness_threshold: float = 0.85
    precision_threshold: float = 0.80


class HeuristicJudge:
    def score(self, question: str, expected_answer: str, actual_answer: str, retrieved_context: str) -> Dict[str, float]:
        relevance = overlap_ratio(question, actual_answer)
        precision = overlap_ratio(expected_answer, actual_answer)

        if (retrieved_context or "").strip():
            faithfulness = overlap_ratio(actual_answer, retrieved_context)
        else:
            faithfulness = precision

        return {
            "answer_relevance": round(float(relevance), 4),
            "faithfulness": round(float(faithfulness), 4),
            "context_precision": round(float(precision), 4),
        }


class OpenAIJudge:
    def __init__(self, model: str) -> None:
        from langchain_openai import ChatOpenAI

        self.model = model
        self.llm = ChatOpenAI(model=model, temperature=0)

    def score(self, question: str, expected_answer: str, actual_answer: str, retrieved_context: str) -> Dict[str, float]:
        prompt = f"""
You are scoring an LLM clinical assistant response.
Return JSON only with keys: answer_relevance, faithfulness, context_precision.
Each value must be a float from 0 to 1.

Question:
{question}

Expected Answer:
{expected_answer}

Actual Answer:
{actual_answer}

Retrieved Context:
{retrieved_context}
""".strip()

        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        payload = self._extract_json(content)

        return {
            "answer_relevance": self._clip(payload.get("answer_relevance", 0.0)),
            "faithfulness": self._clip(payload.get("faithfulness", 0.0)),
            "context_precision": self._clip(payload.get("context_precision", 0.0)),
        }

    @staticmethod
    def _extract_json(content: str) -> Dict[str, float]:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            return json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _clip(value: float) -> float:
        try:
            v = float(value)
        except (TypeError, ValueError):
            v = 0.0
        return round(max(0.0, min(1.0, v)), 4)


def select_judge(provider: str, model: str):
    provider = provider.lower().strip()
    if provider == "openai":
        return OpenAIJudge(model=model)
    return HeuristicJudge()


def build_eval_frame(config: EvalConfig) -> pd.DataFrame:
    golden = pd.read_csv(config.golden_path)
    raw = pd.read_csv(config.raw_responses_path)

    required_golden = {"id", "category", "question", "expected_answer", "source_reference"}
    required_raw = {"run_id", "question_id", "actual_answer", "retrieved_context", "timestamp_utc"}

    if not required_golden.issubset(golden.columns):
        missing = sorted(required_golden - set(golden.columns))
        raise ValueError(f"Golden dataset missing columns: {missing}")

    if not required_raw.issubset(raw.columns):
        missing = sorted(required_raw - set(raw.columns))
        raise ValueError(f"Raw responses missing columns: {missing}")

    run_raw = raw.loc[raw["run_id"].astype(str) == str(config.run_id)].copy()
    run_raw = run_raw[["question_id", "actual_answer", "retrieved_context", "timestamp_utc"]]
    merged = golden.merge(run_raw, how="left", left_on="id", right_on="question_id")

    return merged[
        [
            "id",
            "category",
            "question",
            "expected_answer",
            "source_reference",
            "actual_answer",
            "retrieved_context",
            "timestamp_utc",
        ]
    ].rename(columns={"id": "question_id", "timestamp_utc": "response_timestamp_utc"})


def evaluate_run(config: EvalConfig) -> pd.DataFrame:
    load_dotenv()
    judge = select_judge(config.judge_provider, config.judge_model)
    frame = build_eval_frame(config)

    metric_rows: List[Dict[str, float]] = []
    for _, row in frame.iterrows():
        actual = "" if pd.isna(row["actual_answer"]) else str(row["actual_answer"])
        context = "" if pd.isna(row["retrieved_context"]) else str(row["retrieved_context"])

        if not actual.strip():
            metrics = {"answer_relevance": 0.0, "faithfulness": 0.0, "context_precision": 0.0}
        else:
            metrics = judge.score(
                question=str(row["question"]),
                expected_answer=str(row["expected_answer"]),
                actual_answer=actual,
                retrieved_context=context,
            )
        metric_rows.append(metrics)

    metrics_df = pd.DataFrame(metric_rows)
    results = pd.concat([frame.reset_index(drop=True), metrics_df], axis=1)
    results["run_id"] = config.run_id
    results["judge_provider"] = config.judge_provider
    results["judge_model"] = config.judge_model
    results["evaluated_at_utc"] = utc_now_iso()
    results["app_url"] = config.app_url
    results["pass_relevance"] = results["answer_relevance"] >= config.relevance_threshold
    results["pass_faithfulness"] = results["faithfulness"] >= config.faithfulness_threshold
    results["pass_precision"] = results["context_precision"] >= config.precision_threshold
    results["status"] = results.apply(
        lambda row: "Pass"
        if bool(row["pass_relevance"]) and bool(row["pass_faithfulness"]) and bool(row["pass_precision"])
        else "Fail",
        axis=1,
    )
    return results


def build_summary(results: pd.DataFrame, config: EvalConfig) -> pd.DataFrame:
    answered = results[results["actual_answer"].fillna("").astype(str).str.strip() != ""]

    summary = {
        "run_id": config.run_id,
        "total_cases": int(len(results)),
        "answered_cases": int(len(answered)),
        "pass_cases": int((results["status"] == "Pass").sum()),
        "fail_cases": int((results["status"] == "Fail").sum()),
        "average_relevance": round(float(results["answer_relevance"].mean()), 4),
        "average_faithfulness": round(float(results["faithfulness"].mean()), 4),
        "average_precision": round(float(results["context_precision"].mean()), 4),
        "threshold_relevance": config.relevance_threshold,
        "threshold_faithfulness": config.faithfulness_threshold,
        "threshold_precision": config.precision_threshold,
    }
    return pd.DataFrame([summary])


def save_results(results: pd.DataFrame, run_id: str, output_dir: Path = Path("data")) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"evaluation_results_{run_id}.csv"
    results.to_csv(output_path, index=False)
    return output_path
