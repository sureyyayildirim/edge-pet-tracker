#include <WiFi.h>
#include <esp_now.h>
#include <BLEDevice.h>
#include <BLEScan.h>


const char* TARGET_MAC = "d0:ff:50:64:45:6d"; 
const long TIMEOUT_MS = 4000;  
const int WINDOW_MS = 2000;   


int rssi_living = -110;
int rssi_kitchen = -110;
int rssi_bedroom = -110;

unsigned long last_k_time = 0;
unsigned long last_b_time = 0;
unsigned long window_start = 0;
bool l_ok = false;

typedef struct {
  char id;
  int rssi;
} Msg;

Msg incoming;


void onDataRecv(const uint8_t * mac, const uint8_t *data, int len) {
  if (len == sizeof(Msg)) {
    Msg* temp = (Msg*)data; 
    if (temp->id == 'K') {
      rssi_kitchen = temp->rssi;
      last_k_time = millis();
    } else if (temp->id == 'B') {
      rssi_bedroom = temp->rssi;
      last_b_time = millis();
    }
  }
}

// --- BLE CALLBACK ---
class MyCallbacks : public BLEAdvertisedDeviceCallbacks {
  void onResult(BLEAdvertisedDevice d) {
    if (d.getAddress().toString() == TARGET_MAC) {
      rssi_living = d.getRSSI();
      l_ok = true;
    }
  }
};

void setup() {
  Serial.begin(115200);
  
  WiFi.mode(WIFI_STA);
  WiFi.disconnect(); 
  delay(100);

  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW Hatasi!");
    return;
  }
  
  
  esp_now_register_recv_cb((esp_now_recv_cb_t)onDataRecv);

  BLEDevice::init("");
  BLEScan* scanner = BLEDevice::getScan();
  scanner->setAdvertisedDeviceCallbacks(new MyCallbacks());
  scanner->setActiveScan(true);
  scanner->setInterval(150); 
  scanner->setWindow(120);

  Serial.println("Timestamp,Living,Kitchen,Bedroom");
  window_start = millis();
}

void loop() {
  // BLE Scan
  BLEDevice::getScan()->start(1, false);
  
 
  if (millis() - window_start > WINDOW_MS) {
    
    if (!l_ok) rssi_living = -110;
    if (millis() - last_k_time > TIMEOUT_MS) rssi_kitchen = -110;
    if (millis() - last_b_time > TIMEOUT_MS) rssi_bedroom = -110;

   
    Serial.print(millis());
    Serial.print(",");
    Serial.print(rssi_living);
    Serial.print(",");
    Serial.print(rssi_kitchen);
    Serial.print(",");
    Serial.println(rssi_bedroom);

    l_ok = false; 
    window_start = millis();
  }
  
  BLEDevice::getScan()->clearResults(); 
  delay(100); 
}