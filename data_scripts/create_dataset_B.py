from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = BASE_DIR / "Dataset"

A_FILE = DATASET_DIR / "merged" / "localization_dataset_A.csv"
FEEDING_FILE = DATASET_DIR / "raw_S04_living_room_20260424_164652.csv"

OUTPUT_FILE = DATASET_DIR / "merged" / "localization_dataset_B.csv"


def main():
    # Dataset A
    df_a = pd.read_csv(A_FILE)

    # normal veriler
    df_a["is_feeding_area"] = 0

    # feeding session
    df_f = pd.read_csv(FEEDING_FILE, sep=";")

    df_f["label"] = "living_room"
    df_f["is_feeding_area"] = 1

    # birleşim
    df_b = pd.concat([df_a, df_f], ignore_index=True)

    df_b.to_csv(OUTPUT_FILE, index=False)

    print("Dataset B created.")
    print("Output:", OUTPUT_FILE)
    print("Rows:", len(df_b))

    print("\nLabel distribution:")
    print(df_b["label"].value_counts())

    print("\nFeeding counts:")
    print(df_b["is_feeding_area"].value_counts())


if __name__ == "__main__":
    main()