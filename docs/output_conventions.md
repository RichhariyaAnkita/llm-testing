# Output and Naming Conventions

## Run Identifier
- Format: `YYYYMMDD_HHMMSS`
- Example: `20261120_214500`

## Data Files
- `data/golden_dataset.csv` - canonical expected dataset
- `data/raw_responses.csv` - captured chatbot outputs from UI runs
- `data/regression_baseline.csv` - frozen baseline set for regression comparisons

## Results Files
- `results/evaluation_results_<run_id>.csv`
- `results/ragas_results_<run_id>.csv`
- `results/ragas_summary_<run_id>.csv`
- `results/execution_summary_<run_id>.md`
- `results/safety_report_<run_id>.md`
- `results/defect_log_<run_id>.csv`
- `results/benchmark_report_<run_id>.md`

## Required Metadata Columns
For run-generated files, include:
- `run_id`
- `timestamp_utc`
- `judge_model`
- `app_url`
