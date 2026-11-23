from __future__ import annotations

import argparse
import importlib.util
import os
import random
import subprocess
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation from test case and response files.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=["online", "offline"], default="online")
    parser.add_argument("--offline-seed", type=int, default=42)
    parser.add_argument("--test-cases", default="data/test_cases.csv")
    parser.add_argument("--raw-responses", default="data/raw_responses.csv")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--report-dir", default="results")
    parser.add_argument("--open-dashboard", action="store_true", help="Launch Streamlit dashboard after evaluation completes")
    parser.add_argument("--dashboard-script", default="src/report_dashboard.py")
    return parser.parse_args()


def normalize_context(value: str) -> list[str]:
    text = "" if pd.isna(value) else str(value)
    if not text.strip():
        return []
    parts = [chunk.strip() for chunk in text.split("\n\n---\n\n") if chunk.strip()]
    return parts if parts else [text]


def load_inputs(test_cases_path: Path, raw_responses_path: Path, run_id: str) -> pd.DataFrame:
    cases = pd.read_csv(test_cases_path)
    raw = pd.read_csv(raw_responses_path)

    required_cases = {"test_id", "category", "user_input", "reference", "source_reference"}
    required_raw = {"run_id", "question_id", "actual_answer", "retrieved_context"}
    if not required_cases.issubset(cases.columns):
        raise ValueError(f"Missing test-case columns: {sorted(required_cases - set(cases.columns))}")
    if not required_raw.issubset(raw.columns):
        raise ValueError(f"Missing raw-response columns: {sorted(required_raw - set(raw.columns))}")

    run_raw = raw.loc[raw["run_id"].astype(str) == str(run_id)].copy()
    run_raw = run_raw.rename(columns={"question_id": "test_id", "actual_answer": "response"})

    merged = cases.merge(
        run_raw[["test_id", "response", "retrieved_context", "timestamp_utc", "session_id", "turn_index"]],
        how="left",
        on="test_id",
    )
    merged["response"] = merged["response"].fillna("")
    merged["retrieved_context"] = merged["retrieved_context"].fillna("")
    merged = merged.loc[merged["response"].astype(str).str.strip() != ""].copy()
    return merged


def build_dataset(rows: pd.DataFrame) -> Dataset:
    payload = []
    for _, row in rows.iterrows():
        payload.append(
            {
                "user_input": str(row["user_input"]),
                "retrieved_contexts": normalize_context(row["retrieved_context"]),
                "response": str(row["response"]),
                "reference": str(row["reference"]),
            }
        )
    return Dataset.from_list(payload)


def build_evaluators():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env")

    llm_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    llm = ChatOpenAI(model=llm_model, temperature=0, api_key=api_key)
    embeddings = OpenAIEmbeddings(model=embedding_model, api_key=api_key)
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings), llm_model, embedding_model


def run_offline_mock(merged: pd.DataFrame, run_id: str, seed: int) -> pd.DataFrame:
    random.seed(seed)
    final = merged.reset_index(drop=True).copy()
    final["faithfulness"] = [round(random.random(), 4) for _ in range(len(final))]
    final["answer_relevancy"] = [round(random.random(), 4) for _ in range(len(final))]
    final["context_precision"] = [round(random.random(), 4) for _ in range(len(final))]
    final["run_id"] = run_id
    final["judge_provider"] = "mock-random"
    final["judge_model"] = f"offline-seed-{seed}"
    final["embedding_model"] = "mock"
    return final


def run_online_ragas(merged: pd.DataFrame, run_id: str) -> pd.DataFrame:
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    dataset = build_dataset(merged)
    evaluator_llm, evaluator_embeddings, llm_model, embedding_model = build_evaluators()

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    ragas_df = result.to_pandas().rename(columns={"response": "actual_answer"})
    final = pd.concat(
        [merged.reset_index(drop=True), ragas_df[["faithfulness", "answer_relevancy", "context_precision"]]],
        axis=1,
    )
    final["run_id"] = run_id
    final["judge_provider"] = "openai-ragas"
    final["judge_model"] = llm_model
    final["embedding_model"] = embedding_model
    return final


def write_markdown_report(results: pd.DataFrame, output_path: Path, run_id: str) -> None:
    total = len(results)
    passed = int((results["status"] == "Pass").sum())
    failed = total - passed
    pass_rate = 0.0 if total == 0 else (passed / total) * 100

    category_summary = (
        results.groupby("category", as_index=False)
        .agg(
            count=("test_id", "count"),
            avg_relevance=("answer_relevancy", "mean"),
            avg_faithfulness=("faithfulness", "mean"),
            avg_precision=("context_precision", "mean"),
            pass_rate=("status", lambda s: (s.eq("Pass").mean() * 100)),
        )
        .sort_values("category")
    )

    lines = [
        f"# Execution Summary - {run_id}",
        "",
        f"- Total Answered Cases: {total}",
        f"- Pass: {passed}",
        f"- Fail: {failed}",
        f"- Pass Rate: {pass_rate:.2f}%",
        f"- Average Relevance: {results['answer_relevancy'].mean():.4f}",
        f"- Average Faithfulness: {results['faithfulness'].mean():.4f}",
        f"- Average Precision: {results['context_precision'].mean():.4f}",
        "",
        "## Category Summary",
        "",
        "| Category | Count | Avg Relevance | Avg Faithfulness | Avg Precision | Pass Rate % |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for _, row in category_summary.iterrows():
        lines.append(
            f"| {row['category']} | {int(row['count'])} | {row['avg_relevance']:.4f} | {row['avg_faithfulness']:.4f} | {row['avg_precision']:.4f} | {row['pass_rate']:.2f} |"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def launch_dashboard(dashboard_script: str) -> None:
    script_path = Path(dashboard_script)
    if not script_path.exists():
        print(f"Dashboard launch skipped: script not found at {dashboard_script}")
        return

    if importlib.util.find_spec("streamlit") is None:
        print(
            "Dashboard launch skipped: streamlit module unavailable in current environment. "
            "Install with: python -m pip install -r requirements.txt"
        )
        return

    try:
        subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(script_path)])
        print("Dashboard launched in a new process.")
    except FileNotFoundError:
        print("Dashboard launch skipped: Python executable not found.")
    except Exception as ex:
        print(f"Dashboard launch skipped due to unexpected error: {ex}")


def main() -> None:
    args = parse_args()
    run_id = args.run_id
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    merged = load_inputs(Path(args.test_cases), Path(args.raw_responses), run_id=run_id)
    if merged.empty:
        raise ValueError("No answered rows found for this run_id in raw responses.")

    if args.mode == "offline":
        final = run_offline_mock(merged, run_id=run_id, seed=args.offline_seed)
    else:
        final = run_online_ragas(merged, run_id=run_id)
    final["pass_relevance"] = final["answer_relevancy"] >= 0.80
    final["pass_faithfulness"] = final["faithfulness"] >= 0.85
    final["pass_precision"] = final["context_precision"] >= 0.80
    final["status"] = final.apply(
        lambda row: "Pass"
        if bool(row["pass_relevance"]) and bool(row["pass_faithfulness"]) and bool(row["pass_precision"])
        else "Fail",
        axis=1,
    )

    results_path = output_dir / f"ragas_results_{run_id}.csv"
    summary_path = output_dir / f"ragas_summary_{run_id}.csv"
    report_path = report_dir / f"execution_summary_{run_id}.md"

    final.to_csv(results_path, index=False)

    summary = (
        final.groupby("category", as_index=False)
        .agg(
            count=("test_id", "count"),
            pass_rate=("status", lambda s: (s.eq("Pass").mean() * 100)),
            avg_relevance=("answer_relevancy", "mean"),
            avg_faithfulness=("faithfulness", "mean"),
            avg_precision=("context_precision", "mean"),
        )
        .sort_values("category")
    )
    summary.to_csv(summary_path, index=False)
    write_markdown_report(final, report_path, run_id=run_id)

    print(f"Saved: {results_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {report_path}")
    print(f"Mode: {args.mode}")

    if args.open_dashboard:
        launch_dashboard(args.dashboard_script)


if __name__ == "__main__":
    main()
