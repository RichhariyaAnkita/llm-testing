# Manual Response Capture Workflow

## 1) Start a run
Choose a run id in format `YYYYMMDD_HHMMSS`.

Example:
- `20261120_220000`

## 2) List pending questions
Run:

`python src/capture_ui_responses.py --run-id 20261120_220000 --show-pending`

## 3) Ask each question in the app UI
- Open: `https://llm-rag-testing-workshop.streamlit.app/`
- Copy the exact answer from the chatbot UI.
- If retrieved context is visible, capture it in `--retrieved-context`.

## 4) Save each response
Run per question:

`python src/capture_ui_responses.py --run-id 20261120_220000 --question-id F001 --actual-answer "<paste answer>" --retrieved-context "<optional context>" --session-id "S1" --turn-index 1 --collector "rahul"`

## 5) Resume safely
Use `--show-pending` anytime to continue incomplete runs.
