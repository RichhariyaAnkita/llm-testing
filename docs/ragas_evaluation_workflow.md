# RAGAS Evaluation Workflow (File-Based)

## 1) Generate separate test-case file
This keeps test questions and expected answers in a dedicated file.

```powershell
python src/sync_test_cases.py
```

Output:
- `data/test_cases.csv`

## 2) Capture chatbot outputs
Use the existing manual capture workflow to populate:
- `data/raw_responses.csv`

## 3) Run RAGAS evaluation
Requires `.env` with `OPENAI_API_KEY`.

```powershell
python src/run_ragas_evaluation.py --run-id 20261120_220000
```

Offline mock mode (no API calls, random deterministic scores):

```powershell
python src/run_ragas_evaluation.py --run-id 20261120_220000 --mode offline --offline-seed 42
```

Single command with dashboard auto-launch:

```powershell
python src/run_ragas_evaluation.py --run-id 20261120_220000 --mode offline --offline-seed 42 --open-dashboard
```

Outputs:
- `results/ragas_results_<run_id>.csv`
- `results/ragas_summary_<run_id>.csv`
- `results/execution_summary_<run_id>.md`

## 4) Open report UI
```powershell
streamlit run src/report_dashboard.py
```
