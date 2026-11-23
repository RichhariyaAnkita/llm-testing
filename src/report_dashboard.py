from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="LLM Evaluation Report Dashboard", layout="wide")
st.title("LLM Evaluation Report Dashboard")


def list_result_files() -> list[Path]:
    search_dirs = [Path("results"), Path("data")]
    files: list[Path] = []

    for directory in search_dirs:
        if not directory.exists():
            continue
        files.extend(sorted(directory.glob("ragas_results_*.csv"), reverse=True))
        files.extend(sorted(directory.glob("evaluation_results_*.csv"), reverse=True))

    return sorted(files, reverse=True)


result_files = list_result_files()
if not result_files:
    st.warning("No result files found. Run evaluation first.")
    st.stop()

selected = st.selectbox("Select result file", [f.name for f in result_files])
path = next(p for p in result_files if p.name == selected)
df = pd.read_csv(path)

if "answer_relevancy" not in df.columns and "answer_relevance" in df.columns:
    df = df.rename(columns={"answer_relevance": "answer_relevancy"})

if "actual_answer" not in df.columns and "response" in df.columns:
    df = df.rename(columns={"response": "actual_answer"})

if "question" not in df.columns and "user_input" in df.columns:
    df = df.rename(columns={"user_input": "question"})

required = ["category", "status", "faithfulness", "context_precision", "answer_relevancy"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Selected file is missing expected columns: {missing}")
    st.stop()

total = len(df)
passed = int((df["status"] == "Pass").sum())
failed = total - passed
pass_rate = 0 if total == 0 else (passed / total) * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total", total)
col2.metric("Pass", passed)
col3.metric("Fail", failed)
col4.metric("Pass Rate", f"{pass_rate:.2f}%")

col5, col6, col7 = st.columns(3)
col5.metric("Avg Relevance", f"{df['answer_relevancy'].mean():.4f}")
col6.metric("Avg Faithfulness", f"{df['faithfulness'].mean():.4f}")
col7.metric("Avg Precision", f"{df['context_precision'].mean():.4f}")

summary = (
    df.groupby("category", as_index=False)
    .agg(
        count=("status", "count"),
        pass_rate=("status", lambda s: (s.eq("Pass").mean() * 100)),
        avg_relevance=("answer_relevancy", "mean"),
        avg_faithfulness=("faithfulness", "mean"),
        avg_precision=("context_precision", "mean"),
    )
    .sort_values("category")
)

st.subheader("Category Summary")
st.dataframe(summary, use_container_width=True)

chart = px.bar(summary, x="category", y="pass_rate", title="Pass Rate by Category", text_auto=".2f")
chart.update_layout(yaxis_title="Pass Rate %", xaxis_title="Category")
st.plotly_chart(chart, use_container_width=True)

st.subheader("Run Results")
default_cols = [
    "test_id",
    "question_id",
    "category",
    "question",
    "actual_answer",
    "reference",
    "source_reference",
    "answer_relevancy",
    "faithfulness",
    "context_precision",
    "status",
]
available_default_cols = [c for c in default_cols if c in df.columns]
st.dataframe(df[available_default_cols], use_container_width=True)

st.subheader("Failed Cases")
failed_df = df[df["status"] == "Fail"].copy()
if not failed_df.empty:
    cols = [
        c
        for c in [
            "test_id",
            "question_id",
            "category",
            "question",
            "actual_answer",
            "reference",
            "source_reference",
            "answer_relevancy",
            "faithfulness",
            "context_precision",
            "status",
        ]
        if c in failed_df.columns
    ]
    st.dataframe(failed_df[cols], use_container_width=True)

    st.subheader("Failed Case Details")
    for _, row in failed_df.iterrows():
        case_id = row.get("test_id") or row.get("question_id") or "case"
        title = f"{case_id}: {str(row.get('question', 'Question'))[:120]}"
        with st.expander(title):
            st.markdown("**Question**")
            st.write(row.get("question", ""))
            st.markdown("**Actual Answer**")
            st.write(row.get("actual_answer", ""))
            st.markdown("**Expected Answer**")
            st.write(row.get("reference", ""))
            if "source_reference" in row:
                st.markdown("**Source Reference**")
                st.write(row.get("source_reference", ""))
            if "retrieved_context" in row:
                st.markdown("**Retrieved Context**")
                st.text(str(row.get("retrieved_context", ""))[:4000])
else:
    st.success("No failed cases in this result file.")
