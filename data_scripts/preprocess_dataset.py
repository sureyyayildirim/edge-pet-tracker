from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = BASE_DIR / "Dataset"

INPUT_FILE = DATASET_DIR / "merged" / "localization_dataset_A.csv"
PROCESSED_DIR = DATASET_DIR / "processed"

FEATURE_COLUMNS = ["rssi_living", "rssi_kitchen", "rssi_bedroom"]

LABEL_MAP = {
    "living_room": 0,
    "kitchen": 1,
    "bedroom": 2,
}

REVERSE_LABEL_MAP = {
    0: "living_room",
    1: "kitchen",
    2: "bedroom",
}


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    df = df.dropna(subset=FEATURE_COLUMNS + ["label"])

    unknown_labels = set(df["label"].unique()) - set(LABEL_MAP.keys())
    if unknown_labels:
        raise ValueError(f"Unknown labels found: {unknown_labels}")

    X = df[FEATURE_COLUMNS].copy()
    y = df["label"].map(LABEL_MAP)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_df = pd.DataFrame(X_train_scaled, columns=FEATURE_COLUMNS)
    train_df["label"] = y_train.to_numpy()

    test_df = pd.DataFrame(X_test_scaled, columns=FEATURE_COLUMNS)
    test_df["label"] = y_test.to_numpy()

    train_df.to_csv(PROCESSED_DIR / "train_dataset_A.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test_dataset_A.csv", index=False)

    joblib.dump(scaler, PROCESSED_DIR / "scaler.pkl")
    joblib.dump(LABEL_MAP, PROCESSED_DIR / "label_map.pkl")
    joblib.dump(REVERSE_LABEL_MAP, PROCESSED_DIR / "reverse_label_map.pkl")

    print("Preprocessing completed.")
    print("Train rows:", len(train_df))
    print("Test rows:", len(test_df))
    print("\nLabel mapping:")
    for label_name, label_id in LABEL_MAP.items():
        print(f"{label_id}: {label_name}")


if __name__ == "__main__":
    main()