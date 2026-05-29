#include <WiFi.h>
#include <esp_now.h>
#include <BLEDevice.h>
#include <BLEScan.h>


uint8_t MASTER_MAC[] = {0xB0, 0xA7, 0x32, 0xDB, 0x51, 0x94};
const char* TARGET_MAC = "d0:ff:50:64:45:6d";


char SLAVE_ID = 'B'; // Kitchen için 'K', Bedroom için 'B'


typedef struct {
  char id;
  int rssi;
} Msg;


Msg dataToSend;


void setup() {
  Serial.begin(115200);


  WiFi.mode(WIFI_STA);


  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW init failed!");
    return;
  }


  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, MASTER_MAC, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;


  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("Failed to add master peer!");
    return;
  }


  BLEDevice::init("");


  dataToSend.id = SLAVE_ID;


  Serial.println("Kitchen slave started.");
}


void loop() {
  BLEScan* scanner = BLEDevice::getScan();


  scanner->setActiveScan(true);
  scanner->setInterval(100);
  scanner->setWindow(99);


  BLEScanResults* results = scanner->start(1, false);


  int found_rssi = -110;


  for (int i = 0; i < results->getCount(); i++) {
    BLEAdvertisedDevice device = results->getDevice(i);


    String deviceAddress = device.getAddress().toString().c_str();


    if (deviceAddress.equalsIgnoreCase(TARGET_MAC)) {
      found_rssi = device.getRSSI();


      Serial.print("Target found. RSSI: ");
      Serial.println(found_rssi);
      break;
    }
  }


  if (found_rssi == -110) {
    Serial.println("Target not found. RSSI: -110");
  }


  dataToSend.rssi = found_rssi;


  esp_err_t result = esp_now_send(MASTER_MAC, (uint8_t *) &dataToSend, sizeof(Msg));


  if (result == ESP_OK) {
    Serial.print("Sent to master. ID: ");
    Serial.print(dataToSend.id);
    Serial.print(" RSSI: ");
    Serial.println(dataToSend.rssi);
  } else {
    Serial.println("ESP-NOW send failed!");
  }


  scanner->clearResults();


  delay(500);
}
