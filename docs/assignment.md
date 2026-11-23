# Assignment: Build an LLM Evaluation Framework for a Clinical Review Assistant

## Background
Design and implement a comprehensive LLM Evaluation Framework for the Clinical Review Assistant chatbot and clinical knowledge base.

Application under test:
- https://llm-rag-testing-workshop.streamlit.app/

## Objective
Evaluate end-to-end:
- Answer Quality
- Faithfulness to Knowledge Base
- Precision and Accuracy
- Safety and Compliance
- Regression Stability
- Overall Benchmarking

## Scope of Work

### 1) Golden Dataset Creation
Create a curated dataset with these target counts:

| Category | Count |
|---|---:|
| Factual Clinical Questions | 20 |
| Medication Questions | 15 |
| Procedure Questions | 15 |
| Policy/Guideline Questions | 15 |
| Edge Cases | 10 |
| Out-of-Scope Questions | 10 |
| Adversarial Prompts | 5 |

Each row must include:
- Question
- Expected Answer
- Source Reference
- Category

### 2) Automated LLM Evaluation Framework
Execute all golden dataset questions against the chatbot and compute:
- Answer Relevance (0-1)
- Faithfulness (0-1)
- ContextPrecision (0-1)

Optional:
- Context Recall
- Similarity Score
- Hallucination Rate

### 3) Execution Results Dashboard
Generate per-question results and pass/fail status.

Acceptance thresholds:

| Metric | Pass Threshold |
|---|---:|
| Relevance | >= 0.80 |
| Faithfulness | >= 0.85 |
| Precision | >= 0.80 |

### 4) Safety Testing Guidelines
Validate:
- Clinical safety (incorrect treatment, unsafe advice, wrong dosage)
- Hallucination behavior (missing KB content, fabricated references, nonexistent entities)
- Prompt injection resistance
- Sensitive data exposure prevention (internal prompts/config/hidden instructions/user info)

Safety report fields:
- Test Scenario
- Risk Level
- Actual Result
- Expected Result
- Status

### 5) Defect Reporting
Log and prioritize:
- Functional defects
- AI defects
- Safety defects

Bug template:
- Bug ID
- Title
- Severity
- Priority
- Steps
- Question Asked
- Actual Response
- Expected Response
- Evidence

### 6) LLM Regression Test Suite
Include:
- Critical clinical questions
- High-risk scenarios
- Safety tests
- Prompt injection tests
- Multi-turn conversations

Regression goals:
- No answer degradation
- No increased hallucinations
- No safety regressions
- Consistent response quality

Required regression outputs:
- Regression Pass Rate
- Average Relevance Score
- Average Faithfulness Score
- Average Precision Score

### 7) Final Benchmarking Report
Include:
- Executive summary
- Dataset overview (size, categories, coverage)
- Evaluation results (averages)
- Safety assessment (critical findings, gaps, recommendations)
- Defect summary by severity
- Regression trends
- Final recommendation:
  - Production Ready
  - Conditionally Ready
  - Not Ready

## Expected Deliverables
1. Golden Dataset (CSV/Excel)
2. Automated Evaluation Framework
3. Execution Results Report
4. Safety Testing Report
5. Defect Log
6. Regression Test Suite
7. Final Benchmarking Report
8. Demo Presentation (10 min)

## Success Criteria
- Golden dataset created and reviewed
- Automated evaluation executed successfully
- Relevance, Faithfulness, and Precision generated
- Safety assessment completed
- Defects documented with evidence
- Regression suite reusable for future releases
- Final benchmarking report with actionable recommendations
