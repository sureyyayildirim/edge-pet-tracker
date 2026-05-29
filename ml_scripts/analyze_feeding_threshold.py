from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = BASE_DIR / "Dataset"

DATA_FILE = DATASET_DIR / "merged" / "localization_dataset_B.csv"


def main():
    df = pd.read_csv(DATA_FILE)

    # sadece living room satırları
    living = df[df["label"] == "living_room"]

    normal = living[living["is_feeding_area"] == 0]
    feeding = living[living["is_feeding_area"] == 1]

    print("Normal living_room rows:", len(normal))
    print("Feeding rows:", len(feeding))

    print("\n--- NORMAL LIVING ROOM ---")
    print(normal["rssi_living"].describe())

    print("\n--- FEEDING AREA ---")
    print(feeding["rssi_living"].describe())

    print("\nSuggested thresholds to test:")
    for t in [-60, -55, -50, -45, -40]:
        feeding_detected = (feeding["rssi_living"] > t).mean() * 100
        false_alarm = (normal["rssi_living"] > t).mean() * 100

        print(
            f"Threshold {t}: "
            f"feeding capture={feeding_detected:.1f}% | "
            f"false alarm={false_alarm:.1f}%"
        )


if __name__ == "__main__":
    main()