
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

TARGET = "Machine failure"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    test = pd.read_csv(Path(args.data_dir) / "test.csv")
    X_test = test.drop(columns=[TARGET])
    y_test = test[TARGET].astype(int)

    model = joblib.load(args.model)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= args.threshold).astype(int)

    result = {
        "test_pr_auc": float(
            average_precision_score(y_test, probabilities)
        ),
        "test_roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
        "test_precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "test_recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "test_f1": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
        "decision_threshold": args.threshold,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
