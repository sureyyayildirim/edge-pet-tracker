from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = BASE_DIR / "Dataset"

DATA_FILE = DATASET_DIR / "merged" / "localization_dataset_B.csv"

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

FEEDING_THRESHOLD = -50


def apply_hybrid_rule(room_pred_label, rssi_living):
    """
    Final state decision:
    - First use MLP for room-level localization.
    - Then override living_room as feeding_area if RSSI is very strong.
    """
    if room_pred_label == "living_room" and rssi_living > FEEDING_THRESHOLD:
        return "feeding_area"

    return room_pred_label


def main():
    df = pd.read_csv(DATA_FILE)

    df = df.dropna(subset=FEATURE_COLUMNS + ["label", "is_feeding_area"])

    # Dataset B'de feeding session label = living_room.
    # Room model sadece 3-class öğrenir.
    X = df[FEATURE_COLUMNS].copy()
    y_room = df["label"].map(LABEL_MAP)

    # Final ground truth:
    # feeding ise feeding_area, değilse oda label'ı
    df["final_label"] = df.apply(
        lambda row: "feeding_area" if row["is_feeding_area"] == 1 else row["label"],
        axis=1
    )

    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X,
        y_room,
        df,
        test_size=0.2,
        random_state=42,
        stratify=y_room,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    mlp = MLPClassifier(
        hidden_layer_sizes=(8,),
        activation="relu",
        max_iter=1000,
        random_state=42,
    )

    mlp.fit(X_train_scaled, y_train)

    room_pred_encoded = mlp.predict(X_test_scaled)
    room_pred_labels = [REVERSE_LABEL_MAP[int(pred)] for pred in room_pred_encoded]

    final_predictions = []

    for room_pred, (_, row) in zip(room_pred_labels, df_test.iterrows()):
        final_state = apply_hybrid_rule(
            room_pred_label=room_pred,
            rssi_living=row["rssi_living"],
        )
        final_predictions.append(final_state)

    y_true_final = df_test["final_label"].tolist()

    class_names = ["living_room", "kitchen", "bedroom", "feeding_area"]

    print("===== Hybrid MLP + Rule-Based Feeding Detection =====")
    print(f"Feeding threshold: rssi_living > {FEEDING_THRESHOLD}")

    print("\nAccuracy:", accuracy_score(y_true_final, final_predictions))

    print("\nClassification Report:")
    print(classification_report(
        y_true_final,
        final_predictions,
        labels=class_names,
        target_names=class_names
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(
        y_true_final,
        final_predictions,
        labels=class_names
    ))

    print("\nPrediction counts:")
    print(pd.Series(final_predictions).value_counts())

    print("\nTrue final label counts:")
    print(pd.Series(y_true_final).value_counts())


if __name__ == "__main__":
    main()