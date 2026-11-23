# Clinical Review Assistant LLM Evaluation Framework

This repository contains an end-to-end evaluation framework for a Clinical Review Assistant chatbot, aligned to the assignment scope:
- Golden dataset creation
- Automated evaluation (RAGAS metrics)
- Safety/regression-ready artifacts
- Result reporting and dashboard visualization

## Project Objectives
Evaluate chatbot performance for:
- Answer quality
- Faithfulness to retrieved knowledge
- Precision/accuracy
- Safety/compliance behavior
- Regression stability

## Repository Structure

```text
.
├─ data/
│  ├─ golden_dataset.csv               # Source-of-truth test set
│  ├─ test_cases.csv                   # Normalized test-case input (generated)
│  ├─ raw_responses.csv                # Actual chatbot responses (captured/imported)
│  ├─ safety_scenarios.csv             # Safety test template
│  ├─ regression_baseline.csv          # Regression baseline template
│  ├─ knowledge_base_seed.txt          # Small KB for quick testing
│  ├─ knowledge_base_expanded.txt      # Medium KB
│  └─ knowledge_base_detailed.txt      # Detailed KB aligned with golden set
│
├─ results/
│  ├─ ragas_results_<run_id>.csv       # Per-case metric output
│  ├─ ragas_summary_<run_id>.csv       # Category summary
│  ├─ execution_summary_<run_id>.md    # Markdown summary report
│  └─ defect_log_template.csv          # Defect log template
│
├─ docs/
│  ├─ assignment.md
│  ├─ manual_capture_workflow.md
│  ├─ output_conventions.md
│  └─ ragas_evaluation_workflow.md
│
└─ src/
   ├─ sync_test_cases.py               # golden_dataset -> test_cases
   ├─ capture_ui_responses.py          # Manual response capture helper
   ├─ response_capture.py              # Capture utility functions
   ├─ import_app_export.py             # Import app export CSV to raw_responses
   ├─ run_ragas_evaluation.py          # Main evaluator (online/offline mode)
   ├─ run_evaluation.py                # Alternative heuristic/openai evaluator
   ├─ evaluation_engine.py             # Core non-RAGAS evaluation engine
   └─ report_dashboard.py              # Streamlit dashboard
```

## Requirements
Install dependencies:

```powershell
pip install -r requirements.txt
```

## Environment Configuration
Create `.env` at repo root:

```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

Notes:
- `.env` is ignored by git.
- `.env.example` is provided as template.

## Step-by-Step Workflow

## Quick Testing (Recommended)

If you already have app responses exported, run this exact sequence:

```powershell
python src/sync_test_cases.py
python src/import_app_export.py --export data/app_export_20261123_1140.csv --run-id 20261123_114000
python src/run_ragas_evaluation.py --run-id 20261123_114000 --mode offline --offline-seed 42 --open-dashboard
```

Notes:
- `offline` mode is API-independent and good for validating pipeline/reporting.
- Switch to `--mode online` when OpenAI connectivity is available.

### 1) Build test cases from golden dataset

```powershell
python src/sync_test_cases.py
```

Output:
- `data/test_cases.csv`

### 2) Populate the application knowledge base
Upload one of these to the target app (`/Application`):
- `data/knowledge_base_seed.txt` (quick smoke)
- `data/knowledge_base_expanded.txt`
- `data/knowledge_base_detailed.txt` (recommended)

### 3) Collect chatbot outputs
You have two options:

#### Option A: Import app export CSV
If you exported from app as columns `question,contexts,answer,ground_truth`:

```powershell
python src/import_app_export.py --export data/app_export_20261123_1140.csv --run-id 20261123_114000
```

#### Option B: Manual capture per question
Use helper:

```powershell
python src/capture_ui_responses.py --run-id 20261123_114000 --show-pending
```

Then append each response:

```powershell
python src/capture_ui_responses.py --run-id 20261123_114000 --question-id F001 --actual-answer "<answer>" --retrieved-context "<context>"
```

### 4) Run evaluation

#### Online mode (actual OpenAI + RAGAS)

```powershell
python src/run_ragas_evaluation.py --run-id 20261123_114000 --mode online
```

#### Offline mode (mock random metrics, no API)

```powershell
python src/run_ragas_evaluation.py --run-id 20261123_114000 --mode offline --offline-seed 42
```

#### One-step run + open dashboard

```powershell
python src/run_ragas_evaluation.py --run-id 20261123_114000 --mode offline --offline-seed 42 --open-dashboard
```

### 4.1) Mode selection guide
- Use `--mode offline` when:
  - OpenAI is blocked/unavailable
  - you only want to test ingestion, scoring pipeline, file generation, and dashboard
- Use `--mode online` when:
  - `.env` has valid `OPENAI_API_KEY`
  - network access to OpenAI endpoints is available
  - you need real quality metrics for assignment reporting

### 5) Open dashboard

```powershell
streamlit run src/report_dashboard.py
```

Optional: this can be launched automatically using `--open-dashboard` during step 4.

If `streamlit` command is not found, use:

```powershell
python -m streamlit run src/report_dashboard.py
```

## Metrics and Thresholds
Primary metrics:
- `answer_relevancy`
- `faithfulness`
- `context_precision`

Pass thresholds:
- Relevance >= 0.80
- Faithfulness >= 0.85
- Precision >= 0.80

A row is `Pass` only if all three thresholds pass.

## Output Files
All generated artifacts are written under `results/` by default:
- `ragas_results_<run_id>.csv`
- `ragas_summary_<run_id>.csv`
- `execution_summary_<run_id>.md`

## Troubleshooting

### APIConnectionError during online evaluation
Symptoms:
- RAGAS run completes with many connection errors
- Output metrics contain blank/NaN values

Actions:
1. Verify internet/proxy/firewall access to:
   - `https://api.openai.com/v1/chat/completions`
   - `https://api.openai.com/v1/embeddings`
2. Validate `.env` key and model names
3. Retry using smaller runs
4. Use `--mode offline` to validate pipeline while network issues persist

### No rows evaluated
Cause:
- `data/raw_responses.csv` has no rows for the specified `run_id`

Fix:
- Import or capture responses first, then rerun evaluation.

### Dashboard opens but missing answer text
Cause:
- Result file may use `response` column name instead of `actual_answer`.

Fix:
- Current dashboard normalizes this automatically. Re-run evaluation if file is stale.

## Assignment Deliverables Mapping
- Golden Dataset: `data/golden_dataset.csv`
- Automated Evaluation Framework: `src/run_ragas_evaluation.py` (+ helpers)
- Execution Results Report: `results/execution_summary_<run_id>.md`
- Safety Report: `data/safety_scenarios.csv` (template; execution pending)
- Defect Log: `results/defect_log_template.csv` (template)
- Regression Suite: `data/regression_baseline.csv` (template)
- Benchmarking Report: to be generated from evaluation outputs

## License / Usage
This project is for assignment/demo evaluation purposes. Clinical content is educational and does not replace licensed medical judgment.
