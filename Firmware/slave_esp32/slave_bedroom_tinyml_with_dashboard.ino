#include <WiFi.h>
#include <esp_now.h>
#include <BLEDevice.h>
#include <BLEScan.h>


uint8_t MASTER_MAC[] = {0xB0, 0xA7, 0x32, 0xDB, 0x51, 0x94};
const char* TARGET_MAC = "d0:ff:50:64:45:6d";
char SLAVE_ID = 'B'; // Kitchen için 'K', Bedroom için 'B' yapın


typedef struct {
  char id;
  int rssi;
} Msg;


Msg dataToSend;


void setup() {
  Serial.begin(115200);
  WiFi.mode(WIFI_STA);
  esp_wifi_set_channel(6, WIFI_SECOND_CHAN_NONE);
  esp_now_init();
 
  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, MASTER_MAC, 6);
  esp_now_add_peer(&peerInfo);
 
  BLEDevice::init("");
  dataToSend.id = SLAVE_ID;
}


void loop() {
  BLEScan* scanner = BLEDevice::getScan();
  BLEScanResults results = scanner->start(1, false);
 
  int found_rssi = -110;
  for (int i = 0; i < results.getCount(); i++) {
    if (results.getDevice(i).getAddress().toString() == TARGET_MAC) {
      found_rssi = results.getDevice(i).getRSSI();
    }
  }
 
  dataToSend.rssi = found_rssi;
  esp_now_send(MASTER_MAC, (uint8_t *) &dataToSend, sizeof(Msg));
 
  scanner->clearResults();
  delay(500);
}


