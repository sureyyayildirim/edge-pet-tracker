from pathlib import Path
import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = BASE_DIR / "Dataset"

PROCESSED_DIR = DATASET_DIR / "processed" / "datasetC_random_split"

MODEL_FILE = PROCESSED_DIR / "mlp_3_8_4_model.pkl"
SCALER_FILE = PROCESSED_DIR / "scaler.pkl"
REVERSE_LABEL_MAP_FILE = PROCESSED_DIR / "reverse_label_map.pkl"

MOVING_TEST_FILE = DATASET_DIR / "moving_test_dataset_fixed.csv"

FEATURE_COLUMNS = ["rssi_living", "rssi_kitchen", "rssi_bedroom"]
CLASS_NAMES = ["living_room", "kitchen", "bedroom", "feeding_area"]


def fix_feeding_labels(df):
    """
    If feeding moving sessions were logged as living_room,
    convert them to feeding_area based on session/note/source text.
    """
    text_cols = []

    for col in ["session", "note", "source_file"]:
        if col in df.columns:
            text_cols.append(col)

    if not text_cols:
        return df

    combined_text = ""
    for col in text_cols:
        combined_text = combined_text + " " + df[col].astype(str).str.lower()

    feeding_mask = combined_text.str.contains("feeding", na=False)

    df.loc[feeding_mask, "label"] = "feeding_area"

    return df


def main():
    df = pd.read_csv(MOVING_TEST_FILE)

    df = fix_feeding_labels(df)
    df = df.dropna(subset=FEATURE_COLUMNS + ["label"])

    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    reverse_label_map = joblib.load(REVERSE_LABEL_MAP_FILE)

    X_raw = df[FEATURE_COLUMNS]
    y_true = df["label"]

    X_scaled = scaler.transform(X_raw)

    y_pred_encoded = model.predict(X_scaled)
    y_pred = [reverse_label_map[int(x)] for x in y_pred_encoded]

    print("===== MLP 3-8-4 on Moving Test Dataset =====")
    print("Rows:", len(df))

    print("\nTrue label distribution:")
    print(y_true.value_counts())

    print("\nPredicted label distribution:")
    print(pd.Series(y_pred).value_counts())

    print("\nAccuracy:", accuracy_score(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=CLASS_NAMES,
        target_names=CLASS_NAMES,
        zero_division=0,
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(
        y_true,
        y_pred,
        labels=CLASS_NAMES,
    ))

    result_df = df.copy()
    result_df["predicted_label"] = y_pred
    result_df["is_correct"] = result_df["label"] == result_df["predicted_label"]

    output_file = DATASET_DIR / "moving_test_predictions_B.csv"
    result_df.to_csv(output_file, index=False)

    print("\nPredictions saved to:", output_file)


if __name__ == "__main__":
    main()