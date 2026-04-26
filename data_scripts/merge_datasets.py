from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = BASE_DIR / "Dataset"

RAW_DIR = DATASET_DIR / "raw_sessions"
MERGED_DIR = DATASET_DIR / "merged"

OUTPUT_FILE = MERGED_DIR / "localization_dataset_A.csv"

FEATURE_COLUMNS = ["rssi_living", "rssi_kitchen", "rssi_bedroom"]

REQUIRED_COLUMNS = [
    "pc_time",
    "timestamp_ms",
    "rssi_living",
    "rssi_kitchen",
    "rssi_bedroom",
    "label",
    "session",
    "note",
]

def main():
    MERGED_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(RAW_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")

    dataframes = []

    for csv_file in csv_files:
        # Dataset A: 
        if "feeding" in csv_file.name.lower():
            print(f"Skipped feeding file: {csv_file.name}")
            continue

        df = pd.read_csv(csv_file, sep=";")

        missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"{csv_file.name} is missing columns: {missing_columns}\n"
                f"Existing columns: {list(df.columns)}"
            )

        for col in FEATURE_COLUMNS + ["timestamp_ms"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=FEATURE_COLUMNS + ["label"])

        df["source_file"] = csv_file.name

        dataframes.append(df)

        print(
            f"Added: {csv_file.name} | "
            f"label={df['label'].iloc[0]} | "
            f"rows={len(df)}"
        )

    merged_df = pd.concat(dataframes, ignore_index=True)

    merged_df.to_csv(OUTPUT_FILE, index=False)

    print("\nMerge completed.")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Total rows: {len(merged_df)}")

    print("\nLabel distribution:")
    print(merged_df["label"].value_counts())

    print("\nSessions:")
    print(merged_df[["session", "label", "note", "source_file"]].drop_duplicates())

if __name__ == "__main__":
    main()