from pathlib import Path
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parents[1]

TF_MODEL_PATH = BASE_DIR / "tinyml" / "tf_model"
TFLITE_MODEL_PATH = BASE_DIR / "tinyml" / "mlp_3_8_4_model.tflite"


def main():
    converter = tf.lite.TFLiteConverter.from_saved_model(str(TF_MODEL_PATH))

    # Modeli küçültmek için temel optimizasyon
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open(TFLITE_MODEL_PATH, "wb") as f:
        f.write(tflite_model)

    print("TFLite model saved to:", TFLITE_MODEL_PATH)
    print("Model size:", len(tflite_model), "bytes")


if __name__ == "__main__":
    main()