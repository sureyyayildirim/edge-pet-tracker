from pathlib import Path
import pandas as pd
import joblib

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = BASE_DIR / "Dataset"

PROCESSED_SS_DIR = DATASET_DIR / "processed-split-datas"

TRAIN_SS_FILE = PROCESSED_SS_DIR / "train_session_split_A.csv"
TEST_SS_FILE = PROCESSED_SS_DIR / "test_session_split_A.csv"
REVERSE_LABEL_MAP_FILE = PROCESSED_SS_DIR / "reverse_label_map.pkl"

FEATURE_COLUMNS = ["rssi_living", "rssi_kitchen", "rssi_bedroom"]


def evaluate_model(model_name, model, X_train, y_train, X_test, y_test, class_names):
    print(f"\n===== {model_name} =====")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def main():
    train_df = pd.read_csv(TRAIN_SS_FILE)
    test_df = pd.read_csv(TEST_SS_FILE)

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

    knn = KNeighborsClassifier(n_neighbors=3)

    mlp = MLPClassifier(
        hidden_layer_sizes=(32,),
        learning_rate_init=0.001,
        activation="relu",
        max_iter=500,
        random_state=42,
    ) 

    # ----HYPERPARAMETER TUNING FOR MLP MODEL----

    # print("\nRunning GridSearch for MLP...")

    # param_grid = {
    #     "hidden_layer_sizes": [(8,), (16,), (32,), (16, 8)],
    #     "learning_rate_init": [0.001, 0.0001],
    #     "early_stopping": [True,False],
    #     "max_iter": [500, 800, 1000],
    # }

    # grid = GridSearchCV(
    #     MLPClassifier(
    #         activation="relu",
    #         random_state=42,
    #     ),
    #     param_grid,
    #     cv=3,
    #     n_jobs=-1,
    #     verbose=1
    # )

    # grid.fit(X_train, y_train)

    # print("\nBest Parameters:", grid.best_params_)
    # print("Best CV Score:", grid.best_score_)

    # mlp = grid.best_estimator_

    evaluate_model("Session Splitted Dataset - KNN k=3", knn, X_train, y_train, X_test, y_test, class_names)
    evaluate_model("Session Splitted Dataset - MLP 3-8-3", mlp, X_train, y_train, X_test, y_test, class_names)


if __name__ == "__main__":
    main()

