
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

TARGET = "Machine failure"


def build_pipeline(model_name, X):
    categorical = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    transformers = []

    if numeric:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(
            ("numeric", numeric_pipeline, numeric)
        )

    if categorical:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ])
        transformers.append(
            ("categorical", categorical_pipeline, categorical)
        )

    preprocessor = ColumnTransformer(transformers)

    if model_name == "logistic_regression":
        classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        )
    elif model_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == "xgboost":
        classifier = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
            n_jobs=4,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def evaluate_model(model, X, y, threshold):
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return {
        "pr_auc": float(average_precision_score(y, probabilities)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y, predictions, zero_division=0)
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    metrics_path = Path(args.metrics)

    train = pd.read_csv(data_dir / "train.csv")
    validation = pd.read_csv(data_dir / "validation.csv")

    X_train = train.drop(columns=[TARGET])
    y_train = train[TARGET].astype(int)
    X_val = validation.drop(columns=[TARGET])
    y_val = validation[TARGET].astype(int)

    if set(y_train.unique()) != {0, 1}:
        raise ValueError("Training target must contain both classes")

    if set(y_val.unique()) != {0, 1}:
        raise ValueError("Validation target must contain both classes")

    if not 0 < args.threshold < 1:
        raise ValueError("Threshold must be between 0 and 1")

    results = {}
    fitted_models = {}

    for model_name in [
        "logistic_regression",
        "random_forest",
        "xgboost",
    ]:
        print(f"Training {model_name}...")
        model = build_pipeline(model_name, X_train)
        model.fit(X_train, y_train)

        results[model_name] = evaluate_model(
            model, X_val, y_val, args.threshold
        )
        fitted_models[model_name] = model

        print(
            f"{model_name}: "
            f"{json.dumps(results[model_name], indent=2)}"
        )

    # Select using validation PR-AUC, not test-set performance.
    best_name = max(
        results,
        key=lambda name: results[name]["pr_auc"],
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        fitted_models[best_name],
        model_dir / "best_model.joblib",
    )

    output = {
        "selection_metric": "pr_auc",
        "decision_threshold": args.threshold,
        "selected_model": best_name,
        "validation_metrics": results,
    }

    metrics_path.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print(f"Selected model: {best_name}")


if __name__ == "__main__":
    main()
