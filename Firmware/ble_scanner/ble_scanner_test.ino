#include <BLEDevice.h>
#include <BLEScan.h>

// Different node name for each Esp32 device 
#define NODE_NAME "BEDROOM"
String TARGET_MAC = "D0:FF:50:64:45:6D";

BLEScan* scanner;
int scanTime = 2;

bool found = false;
int rssi = -100;

static int lastValidRSSI = -100;
static int missCount = 0;

class Callbacks : public BLEAdvertisedDeviceCallbacks {
  void onResult(BLEAdvertisedDevice d) {
    String mac = d.getAddress().toString().c_str();
    mac.toUpperCase();

    if (mac == TARGET_MAC) {
      found = true;
      rssi = d.getRSSI();
    }
  }
};

void setup() {
  Serial.begin(115200);
  BLEDevice::init("");

  scanner = BLEDevice::getScan();
  scanner->setAdvertisedDeviceCallbacks(new Callbacks());
  scanner->setActiveScan(true);
  scanner->setInterval(100);
  scanner->setWindow(99);
}

void loop() {
  found = false;

  scanner->start(scanTime, false);

  if (found) {
    lastValidRSSI = rssi;
    missCount = 0;
  } else {
    missCount++;
  }

  if (missCount > 2) {
    lastValidRSSI = -110;
  }

  Serial.print(NODE_NAME);
  Serial.print(" RSSI: ");
  Serial.println(lastValidRSSI);

  scanner->clearResults();
}