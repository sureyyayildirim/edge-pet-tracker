from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = BASE_DIR / "Dataset"

INPUT_FILE = DATASET_DIR / "merged" / "localization_dataset_A.csv"

FEATURE_COLUMNS = [
    "rssi_living",
    "rssi_kitchen",
    "rssi_bedroom"
]

LABEL_MAP = {
    "living_room":0,
    "kitchen":1,
    "bedroom":2
}

REVERSE_LABEL_MAP = {
    0:"living_room",
    1:"kitchen",
    2:"bedroom"
}


def cross_validate_model(model_name, model, X, y, cv, class_names):

    print(f"\n{'='*20}")
    print(model_name)
    print(f"{'='*20}")

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    print("Fold Accuracies:", scores)
    print("Mean Accuracy:", np.mean(scores))
    print("Std:", np.std(scores))


    y_pred = cross_val_predict(
        model,
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

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y,
            y_pred
        )
    )


def main():

    df = pd.read_csv(INPUT_FILE)

    df = df.dropna(
        subset=FEATURE_COLUMNS + ["label"]
    )

    if df["label"].dtype == "object":
        df["label"] = df["label"].map(LABEL_MAP)

    X = df[FEATURE_COLUMNS]
    y = df["label"]

    class_names = [
        REVERSE_LABEL_MAP[0],
        REVERSE_LABEL_MAP[1],
        REVERSE_LABEL_MAP[2]
    ]

    print("Dataset rows:", len(df))
    print("Classes:", class_names)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )


    models = {

        "KNN k=3":
            KNeighborsClassifier(
                n_neighbors=3
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                random_state=42
            ),

        "SVM RBF":
            Pipeline([
                ("scaler", StandardScaler()),
                ("svm", SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale"
                ))
            ]),

        "Logistic Regression":
            Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(
                    max_iter=1000,
                    random_state=42
                ))
            ]),

        "MLP 3-8-3":
            Pipeline([
                ("scaler", StandardScaler()),
                ("mlp", MLPClassifier(
                    hidden_layer_sizes=(32,),
                    activation="relu",
                    max_iter=1000,
                    random_state=42
                ))
            ])
    }


    for name, model in models.items():
        cross_validate_model(
            name,
            model,
            X,
            y,
            cv,
            class_names
        )


if __name__ == "__main__":
    main()