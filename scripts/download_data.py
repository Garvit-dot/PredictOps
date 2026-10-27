
from pathlib import Path

from ucimlrepo import fetch_ucirepo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = RAW_DIR / "ai4i2020.csv"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    dataset = fetch_ucirepo(id=601)
    df = dataset.data.original

    if df is None or df.empty:
        raise RuntimeError(
            "The original dataset was not returned by UCI."
        )

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved dataset to: {OUTPUT_FILE}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Column names: {df.columns.tolist()}")
    print("\nFirst five rows:")
    print(df.head())
    print("\nMissing values per column:")
    print(df.isnull().sum())


if __name__ == "__main__":
    main()
