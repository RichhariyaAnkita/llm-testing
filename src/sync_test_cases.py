from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    golden_path = Path("data/golden_dataset.csv")
    output_path = Path("data/test_cases.csv")

    df = pd.read_csv(golden_path)
    required = ["id", "category", "question", "expected_answer", "source_reference", "risk_level"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in golden dataset: {missing}")

    test_cases = df[required].rename(
        columns={
            "id": "test_id",
            "question": "user_input",
            "expected_answer": "reference",
        }
    )
    test_cases["active"] = True
    test_cases.to_csv(output_path, index=False)
    print(f"Saved {len(test_cases)} test cases to {output_path}")


if __name__ == "__main__":
    main()
