
from pathlib import Path
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "Machine failure"

LEAKAGE_COLUMNS = [
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

IDENTIFIER_COLUMNS = [
    "UID",
    "Product ID",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument(
        "--validation-size", type=float, default=0.15
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if not 0 < args.test_size < 1:
        raise ValueError("test-size must be between 0 and 1")

    if not 0 < args.validation_size < 1:
        raise ValueError(
            "validation-size must be between 0 and 1"
        )

    if args.test_size + args.validation_size >= 1:
        raise ValueError(
            "Test and validation fractions must sum to less than 1"
        )

    df = pd.read_csv(args.input)

    if df.empty or TARGET not in df.columns:
        raise ValueError("Dataset is empty or target is missing")

    if df[TARGET].isna().any():
        raise ValueError("Target contains missing values")

    if set(df[TARGET].unique()) != {0, 1}:
        raise ValueError("Expected a binary target encoded as 0 and 1")

    excluded = [
        col for col in IDENTIFIER_COLUMNS + LEAKAGE_COLUMNS
        if col in df.columns
    ]

    X = df.drop(columns=[TARGET] + excluded)
    y = df[TARGET].astype(int)

    if X.empty:
        raise ValueError("No input features remain")

    if X.isna().any().any():
        raise ValueError(
            "Input features contain missing values; "
            "define an explicit imputation strategy"
        )

    # First reserve the untouched final test set.
    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    # Convert the requested overall validation fraction
    # to a fraction of the remaining development data.
    val_fraction = (
        args.validation_size / (1 - args.test_size)
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_dev,
        y_dev,
        test_size=val_fraction,
        random_state=args.random_state,
        stratify=y_dev,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, features, labels in [
        ("train", X_train, y_train),
        ("validation", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        result = features.copy()
        result[TARGET] = labels.to_numpy()
        result.to_csv(output_dir / f"{name}.csv", index=False)

        print(
            f"{name}: rows={len(result)}, "
            f"failures={int(labels.sum())}"
        )

    print("Excluded identifiers and failure-mode labels:", excluded)
    print("Feature columns:", X.columns.tolist())


if __name__ == "__main__":
    main()
