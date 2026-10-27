
import json
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "data" / "processed" / "train.csv"
CURRENT_PATH = ROOT / "reports" / "drift" / "predictions.csv"
OUTPUT_DIR = ROOT / "reports" / "drift"

FEATURES = [
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {REFERENCE_PATH}")
    if not CURRENT_PATH.exists():
        raise FileNotFoundError(f"Prediction log not found: {CURRENT_PATH}")

    reference = pd.read_csv(REFERENCE_PATH)
    current_log = pd.read_csv(CURRENT_PATH)

    for name, frame in [
        ("training data", reference),
        ("prediction log", current_log),
    ]:
        missing = set(FEATURES) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")

    reference = reference[FEATURES].copy()
    current = current_log[FEATURES].copy()

    reference["Type"] = reference["Type"].astype(str)
    current["Type"] = current["Type"].astype(str)

    for column in FEATURES[1:]:
        reference[column] = pd.to_numeric(reference[column], errors="coerce")
        current[column] = pd.to_numeric(current[column], errors="coerce")

    reference = reference.dropna()
    current = current.dropna()

    if reference.empty or current.empty:
        raise ValueError("No usable rows remain after cleaning.")

    print(f"Reference rows: {len(reference)}")
    print(f"Current prediction rows: {len(current)}")

    report = Report([DataDriftPreset()])
    snapshot = report.run(
        reference_data=reference,
        current_data=current,
    )

    snapshot.save_html(str(OUTPUT_DIR / "data_drift_report.html"))
    (OUTPUT_DIR / "data_drift_report.json").write_text(
        snapshot.json(), encoding="utf-8"
    )

    metadata = {
        "reference_dataset": str(REFERENCE_PATH.relative_to(ROOT)),
        "current_dataset": str(CURRENT_PATH.relative_to(ROOT)),
        "reference_rows_used": len(reference),
        "current_rows_used": len(current),
        "features_checked": FEATURES,
        "note": (
            "Prediction logs may include synthetic requests. "
            "This report tests the monitoring pipeline and does not "
            "establish real-world production drift."
        ),
    }
    (OUTPUT_DIR / "data_drift_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("Drift report generated:")
    print(OUTPUT_DIR / "data_drift_report.html")
    print(OUTPUT_DIR / "data_drift_report.json")
    print(OUTPUT_DIR / "data_drift_metadata.json")


if __name__ == "__main__":
    main()
