# This script combines two different collected dataset versions
# into a single unified dataset for model training.
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = BASE_DIR / "Dataset"

STATIC_FILE = DATASET_DIR / "merged" / "localization_dataset.csv"
MOVING_FILE = DATASET_DIR / "moving_dataset.csv"

OUTPUT_FILE = DATASET_DIR / "merged" / "fiinal_localization_dataset.csv"


def main():
    static_df = pd.read_csv(STATIC_FILE)
    moving_df = pd.read_csv(MOVING_FILE)

    combined_df = pd.concat([static_df, moving_df], ignore_index=True)

    combined_df.to_csv(OUTPUT_FILE, index=False)

    print("Final dataset created.")
    print("Output:", OUTPUT_FILE)
    print("Rows:", len(combined_df))

    print("\nLabel distribution:")
    print(combined_df["label"].value_counts())

    print("\nSources:")
    if "source_file" in combined_df.columns:
        print(combined_df["source_file"].value_counts().head(20))


if __name__ == "__main__":
    main()