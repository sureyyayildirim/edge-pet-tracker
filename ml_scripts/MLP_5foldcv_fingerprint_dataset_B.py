from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    cross_val_predict
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)


BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = BASE_DIR / "Dataset"

INPUT_FILE = DATASET_DIR / "merged" / "localization_dataset_B.csv"

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

    df = pd.read_csv(INPUT_FILE)

    df = df.dropna(
        subset=FEATURE_COLUMNS + ["label"]
    )

    X = df[FEATURE_COLUMNS]

    y = df["label"].map(
        LABEL_MAP
    )

    class_names = [
        REVERSE_LABEL_MAP[0],
        REVERSE_LABEL_MAP[1],
        REVERSE_LABEL_MAP[2],
        REVERSE_LABEL_MAP[3]
    ]

    print("Rows:", len(df))
    print("Classes:", class_names)


    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    mlp = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "mlp",
            MLPClassifier(
                hidden_layer_sizes=(8,),
                activation="relu",
                max_iter=1500,
                random_state=42
            )
        )
    ])


    scores = cross_val_score(
        mlp,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )


    print("\nFold Accuracies:")
    print(scores)

    print(
        "\nMean Accuracy:",
        np.mean(scores)
    )

    print(
        "Std:",
        np.std(scores)
    )


    y_pred = cross_val_predict(
        mlp,
        X,
        y,
        cv=cv
    )


    print("\nClassification Report:")
    print(
        classification_report(
            y,
            y_pred,
            target_names=class_names
        )
    )


    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y,
            y_pred
        )
    )


if __name__ == "__main__":
    main()