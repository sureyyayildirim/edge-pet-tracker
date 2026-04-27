# uses localization_dataset_A.csv
# split merged dataset as train(S1,S2) and test(S3)
# apply standard scaler to datasets and then creates csv files

from pathlib import Path
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = BASE_DIR / "Dataset"

INPUT_FILE = DATASET_DIR / "merged" / "localization_dataset_A.csv"
PROCESSED_SS_DIR = DATASET_DIR / "processed-split-datas"

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
    PROCESSED_SS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    df = df.dropna(subset=FEATURE_COLUMNS + ["label", "session"])

    unknown_labels = set(df["label"].unique()) - set(LABEL_MAP.keys())
    if unknown_labels:
        raise ValueError(f"Unknown labels found: {unknown_labels}")

    df["label"] = df["label"].map(LABEL_MAP)

    #  SESSION SPLIT
    train_df = df[df["session"].isin(["S01", "S02"])].copy()
    test_df = df[df["session"] == "S03"].copy()

    print("Train sessions:", train_df["session"].unique())
    print("Test sessions:", test_df["session"].unique())

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["label"]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["label"]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_out = pd.DataFrame(X_train_scaled, columns=FEATURE_COLUMNS)
    train_out["label"] = y_train.to_numpy()

    test_out = pd.DataFrame(X_test_scaled, columns=FEATURE_COLUMNS)
    test_out["label"] = y_test.to_numpy()

    train_out.to_csv(PROCESSED_SS_DIR / "train_session_split_A.csv", index=False)
    test_out.to_csv(PROCESSED_SS_DIR / "test_session_split_A.csv", index=False)

    joblib.dump(scaler, PROCESSED_SS_DIR / "scaler.pkl")
    joblib.dump(LABEL_MAP, PROCESSED_SS_DIR / "label_map.pkl")
    joblib.dump(REVERSE_LABEL_MAP, PROCESSED_SS_DIR / "reverse_label_map.pkl")

    print("\nPreprocessing completed.")
    print("Train rows:", len(train_out))
    print("Test rows:", len(test_out))

    print("\nLabel mapping:")
    for k, v in LABEL_MAP.items():
        print(f"{v}: {k}")


if __name__ == "__main__":
    main()