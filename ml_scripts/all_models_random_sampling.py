from pathlib import Path
import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = BASE_DIR / "Dataset"

PROCESSED_RS_DIR = DATASET_DIR / "datasetA_random_split"

TRAIN_FILE = PROCESSED_RS_DIR / "train_dataset_A.csv"
TEST_FILE = PROCESSED_RS_DIR / "test_dataset_A.csv"
REVERSE_LABEL_MAP_FILE = PROCESSED_RS_DIR / "reverse_label_map.pkl"

FEATURE_COLUMNS = ["rssi_living", "rssi_kitchen", "rssi_bedroom"]


def evaluate_model(model_name, model, X_train, y_train, X_test, y_test, class_names):
    print(f"\n{'=' * 20}")
    print(model_name)
    print(f"{'=' * 20}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def main():
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    reverse_label_map = joblib.load(REVERSE_LABEL_MAP_FILE)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["label"]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["label"]

    class_names = [
        reverse_label_map[0],
        reverse_label_map[1],
        reverse_label_map[2],
    ]

    print("Train rows:", len(train_df))
    print("Test rows:", len(test_df))
    print("Classes:", class_names)

    models = {
        "KNN k=3": KNeighborsClassifier(
            n_neighbors=3
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "SVM RBF": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale"
        ),

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "MLP 3-8-3": MLPClassifier(
            hidden_layer_sizes=(32,),
            activation="relu",
            max_iter=1000,
            random_state=42
        ),
    }

    for model_name, model in models.items():
        evaluate_model(
            model_name,
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            class_names
        )


if __name__ == "__main__":
    main()