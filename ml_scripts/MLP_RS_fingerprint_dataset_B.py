from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = BASE_DIR / "Dataset"

INPUT_FILE = DATASET_DIR / "merged" / "localization_dataset_B.csv"
PROCESSED_RS_DIR = DATASET_DIR / "processed-dataset-B"

FEATURE_COLUMNS = [
    "rssi_living",
    "rssi_kitchen",
    "rssi_bedroom"
]

LABEL_MAP = {
    "living_room":0,
    "kitchen":1,
    "bedroom":2,
    "feeding_area":3
}

REVERSE_LABEL_MAP = {
    0:"living_room",
    1:"kitchen",
    2:"bedroom",
    3:"feeding_area"
}


def main():

    PROCESSED_RS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    df = df.dropna(
        subset=FEATURE_COLUMNS + ["label"]
    )

    unknown_labels = set(df["label"].unique()) - set(LABEL_MAP.keys())

    if unknown_labels:
        raise ValueError(
            f"Unknown labels found: {unknown_labels}"
        )

    X = df[FEATURE_COLUMNS].copy()

    y_room = df["label"].map(LABEL_MAP)


    # Random sampling fixed
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X,
        y_room,
        df,
        test_size=0.2,
        random_state=42,
        stratify=y_room,
    )


    # Standardization
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)


    # save scaler for later TinyML preprocessing reference
    joblib.dump(
        scaler,
        PROCESSED_RS_DIR / "scaler_B.pkl"
    )


    class_names = [
        REVERSE_LABEL_MAP[0],
        REVERSE_LABEL_MAP[1],
        REVERSE_LABEL_MAP[2],
        REVERSE_LABEL_MAP[3],
    ]


    print("Train rows:", len(X_train))
    print("Test rows:", len(X_test))
    print("Classes:", class_names)


    # MLP
    model = MLPClassifier(
        hidden_layer_sizes=(32,),
        activation="relu",
        max_iter=1000,
        random_state=42
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    y_pred = model.predict(
        X_test_scaled
    )


    print("\nAccuracy:")
    print(
        accuracy_score(
            y_test,
            y_pred
        )
    )


    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=class_names
        )
    )


    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )


    # feeding area recall özellikle kritik
    print("\nFeeding Area Detection Check:")
    feeding_idx = LABEL_MAP["feeding_area"]

    feeding_true = (y_test == feeding_idx).sum()
    feeding_pred = (y_pred == feeding_idx).sum()

    print(
        "Actual feeding_area samples:",
        feeding_true
    )

    print(
        "Predicted feeding_area samples:",
        feeding_pred
    )


    # save model
    joblib.dump(
        model,
        PROCESSED_RS_DIR / "mlp_feeding_area_B.pkl"
    )


if __name__ == "__main__":
    main()