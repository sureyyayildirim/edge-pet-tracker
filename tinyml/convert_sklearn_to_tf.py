from pathlib import Path
import joblib
import tensorflow as tf


REPO_DIR = Path(__file__).resolve().parents[1]   # edge-pet-tracker
CODE_DIR = Path(__file__).resolve().parents[2]   # Code

MODEL_FILE = CODE_DIR / "Dataset" / "processed" / "datasetC_random_split" / "mlp_3_8_4_model.pkl"
OUTPUT_TF_MODEL = REPO_DIR / "tinyml" / "tf_model"


def main():
    print("Loading model from:", MODEL_FILE)

    sklearn_model = joblib.load(MODEL_FILE)

    w1 = sklearn_model.coefs_[0]
    b1 = sklearn_model.intercepts_[0]
    w2 = sklearn_model.coefs_[1]
    b2 = sklearn_model.intercepts_[1]

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(3,)),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(4, activation="softmax"),
    ])

    model.layers[0].set_weights([w1, b1])
    model.layers[1].set_weights([w2, b2])

    model.export(OUTPUT_TF_MODEL)

    print("TensorFlow SavedModel exported to:", OUTPUT_TF_MODEL)


if __name__ == "__main__":
    main()