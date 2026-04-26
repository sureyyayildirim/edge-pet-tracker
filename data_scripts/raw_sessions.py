import serial
import csv
from datetime import datetime

PORT = "COM6"
BAUD = 115200

label = input("Label girin (living_room / bedroom / kitchen): ").strip()
session = input("Session ID girin (örn: S01): ").strip()
note = input("Not girin (örn: sofa_near, bed_near): ").strip()

filename = f"raw_{session}_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

ser = serial.Serial(PORT, BAUD, timeout=2)

with open(filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "pc_time",
        "timestamp_ms",
        "rssi_living",
        "rssi_kitchen",
        "rssi_bedroom",
        "label",
        "session",
        "note"
    ])

    print(f"Kayıt başladı: {filename}")
    print("Durdurmak için CTRL+C")

    try:
        while True:
            line = ser.readline().decode(errors="ignore").strip()

            if not line:
                continue

            if line.startswith("Timestamp") or line.startswith("timestamp"):
                continue

            parts = line.split(",")

            if len(parts) != 4:
                print("Atlandı:", line)
                continue

            timestamp_ms, living, kitchen, bedroom = parts

            writer.writerow([
                datetime.now().isoformat(),
                timestamp_ms,
                living,
                kitchen,
                bedroom,
                label,
                session,
                note
            ])

            print(line)

    except KeyboardInterrupt:
        print("\nKayıt durduruldu.")
        ser.close()