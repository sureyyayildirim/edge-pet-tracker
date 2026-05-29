#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <BLEDevice.h>
#include <BLEScan.h>

#include "model_data.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <PubSubClient.h>

// ---------------- DEBUG SETTINGS ----------------

#define DEBUG_BLE_ALL false  

// ---------------- BLE / ESP-NOW ----------------

const char* TARGET_MAC = "d0:ff:50:64:45:6d";

// ---------------- WiFi / MQTT ----------------
#define ESPNOW_CHANNEL 8

const char* WIFI_SSID = "TURKNET_B9707";
const char* WIFI_PASSWORD = "59zZGdPC";

const char* MQTT_SERVER = "broker.hivemq.com";
const int MQTT_PORT = 1883;
const char* MQTT_TOPIC = "pettracker/data";

WiFiClient espClient;
PubSubClient mqttClient(espClient);

// ---------------- Timing ----------------

const long TIMEOUT_MS = 30000;     // 10 saniye boyunca veri gelmezse -110 yapar
const int WINDOW_MS = 5000;        // 3 saniyede bir çıktı/publish
const int BLE_SCAN_SECONDS = 5;    // BLE scan süresi

const int NO_SIGNAL_RSSI = -110;
const int NO_SIGNAL_INDEX = -1;

// ---------------- RSSI VALUES ----------------

int rssi_living = -110;
int rssi_kitchen = -110;
int rssi_bedroom = -110;

unsigned long last_l_time = 0;
unsigned long last_k_time = 0;
unsigned long last_b_time = 0;

unsigned long window_start = 0;

// ---------------- ESP-NOW MESSAGE STRUCT ----------------

typedef struct {
  char id;
  int rssi;
} Msg;

// ---------------- ESP-NOW RECEIVE CALLBACK ----------------

void printMacAddress(const uint8_t *mac) {
  for (int i = 0; i < 6; i++) {
    if (mac[i] < 16) Serial.print("0");
    Serial.print(mac[i], HEX);
    if (i < 5) Serial.print(":");
  }
  Serial.println();
}

void onDataRecv(const uint8_t *mac, const uint8_t *data, int len) {
  Serial.println("----- ESP-NOW DATA RECEIVED -----");

  Serial.print("Len: ");
  Serial.println(len);

  Serial.print("From MAC: ");
  printMacAddress(mac);

  if (len != sizeof(Msg)) {
    Serial.println("Message size mismatch!");
    return;
  }

  Msg* temp = (Msg*)data;

  Serial.print("Slave ID: ");
  Serial.println(temp->id);

  Serial.print("Slave RSSI: ");
  Serial.println(temp->rssi);

  if (temp->id == 'K') {
    rssi_kitchen = temp->rssi;
    last_k_time = millis();
    Serial.println("Kitchen RSSI updated.");
  } 
  else if (temp->id == 'B') {
    rssi_bedroom = temp->rssi;
    last_b_time = millis();
    Serial.println("Bedroom RSSI updated.");
  } 
  else {
    Serial.println("Unknown slave ID received.");
  }

  Serial.println("---------------------------------");
}

// ---------------- BLE CALLBACK ----------------

class MyCallbacks : public BLEAdvertisedDeviceCallbacks {
  void onResult(BLEAdvertisedDevice d) {
    String addr = d.getAddress().toString().c_str();

    if (DEBUG_BLE_ALL) {
      Serial.print("BLE found: ");
      Serial.print(addr);
      Serial.print(" | RSSI: ");
      Serial.println(d.getRSSI());
    }

    if (addr.equalsIgnoreCase(TARGET_MAC)) {
      rssi_living = d.getRSSI();
      last_l_time = millis();

      Serial.print("TARGET BLE FOUND BY MASTER. Living RSSI: ");
      Serial.println(rssi_living);
    }
  }
};

// ---------------- TinyML / TFLite ----------------

const float SCALER_MEAN[3] = {
  -87.24034335f,
  -96.34406295f,
  -99.82188841f
};

const float SCALER_SCALE[3] = {
  21.73894478f,
  13.49597263f,
  14.97261807f
};

// class order:
// 0 = living_room
// 1 = kitchen
// 2 = bedroom
// 3 = feeding_area

constexpr int TENSOR_ARENA_SIZE = 8 * 1024;
uint8_t tensor_arena[TENSOR_ARENA_SIZE];

const tflite::Model* tfl_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

tflite::MicroMutableOpResolver<3> resolver;

float raw_confidence = 0.0f;
float stable_confidence = 0.0f;
// ---------------- Majority Vote ----------------

const int VOTE_SIZE = 5;
int vote_buffer[VOTE_SIZE];
int vote_count = 0;
int vote_index = 0;

void initVoteBuffer() {
  for (int i = 0; i < VOTE_SIZE; i++) {
    vote_buffer[i] = NO_SIGNAL_INDEX;
  }
}

void resetVoteBuffer() {
  for (int i = 0; i < VOTE_SIZE; i++) {
    vote_buffer[i] = NO_SIGNAL_INDEX;
  }
  vote_count = 0;
  vote_index = 0;
}

void addPredictionToVote(int prediction) {
  if (prediction == NO_SIGNAL_INDEX) return;

  vote_buffer[vote_index] = prediction;
  vote_index = (vote_index + 1) % VOTE_SIZE;

  if (vote_count < VOTE_SIZE) {
    vote_count++;
  }
}

int getMajorityPrediction() {
  if (vote_count == 0) {
    return NO_SIGNAL_INDEX;
  }

  int counts[4] = {0};

  for (int i = 0; i < VOTE_SIZE; i++) {
    int p = vote_buffer[i];
    if (p >= 0 && p < 4) {
      counts[p]++;
    }
  }

  int max_i = 0;
  for (int i = 1; i < 4; i++) {
    if (counts[i] > counts[max_i]) {
      max_i = i;
    }
  }

  return (counts[max_i] == 0) ? NO_SIGNAL_INDEX : max_i;
}

// ---------------- Behavioral Budgeting ----------------

const unsigned long FEEDING_CONFIRM_MS = 15000;
const unsigned long NO_FEEDING_THRESHOLD_MS = 12UL * 60UL * 60UL * 1000UL;
const unsigned long BUDGET_WINDOW_MS = 24UL * 60UL * 60UL * 1000UL;

bool feeding_candidate = false;
bool feeding_confirmed = false;

unsigned long feeding_candidate_start = 0;
unsigned long feeding_session_start = 0;
unsigned long total_feeding_time_ms = 0;
unsigned long last_feeding_seen_time = 0;
unsigned long budget_window_start = 0;

void resetBudgetWindowIfNeeded(unsigned long now) {
  if (now - budget_window_start >= BUDGET_WINDOW_MS) {
    total_feeding_time_ms = 0;
    budget_window_start = now;

    if (feeding_confirmed) {
      feeding_session_start = now;
    }
  }
}

void updateFeedingBehavior(int stable_prediction, unsigned long now) {
  resetBudgetWindowIfNeeded(now);

  if (stable_prediction == 3) {
    if (!feeding_candidate) {
      feeding_candidate = true;
      feeding_candidate_start = now;
    }

    if (!feeding_confirmed && now - feeding_candidate_start >= FEEDING_CONFIRM_MS) {
      feeding_confirmed = true;
      feeding_session_start = now;
      last_feeding_seen_time = now;
    }

    if (feeding_confirmed) {
      last_feeding_seen_time = now;
    }
  } 
  else {
    feeding_candidate = false;
    feeding_candidate_start = 0;

    if (feeding_confirmed) {
      total_feeding_time_ms += now - feeding_session_start;
      feeding_confirmed = false;
      feeding_session_start = 0;
    }
  }
}

unsigned long getCurrentFeedingTime(unsigned long now) {
  if (feeding_confirmed) {
    return total_feeding_time_ms + (now - feeding_session_start);
  }

  return total_feeding_time_ms;
}

bool isNoFeedingAnomaly(unsigned long now) {
  if (last_feeding_seen_time == 0) {
    return now - budget_window_start >= NO_FEEDING_THRESHOLD_MS;
  }

  return now - last_feeding_seen_time >= NO_FEEDING_THRESHOLD_MS;
}

unsigned long getMinutes(unsigned long ms) {
  return ms / 60000UL;
}

// ---------------- Helper Functions ----------------

const char* className(int index) {
  switch (index) {
    case 0: return "living_room";
    case 1: return "kitchen";
    case 2: return "bedroom";
    case 3: return "feeding_area";
    case NO_SIGNAL_INDEX: return "no_signal";
    default: return "unknown";
  }
}

bool unreliableSignalVector() {
  int valid_count = 0;

  if (rssi_living > -105) valid_count++;
  if (rssi_kitchen > -105) valid_count++;
  if (rssi_bedroom > -105) valid_count++;

  if (valid_count == 0) {
    return true;
  }

  if (valid_count == 1) {
    int strongest = max(rssi_living, max(rssi_kitchen, rssi_bedroom));

    if (strongest < -80) {
      return true;
    }
  }

  return false;
}

int predictState(float living, float kitchen, float bedroom) {
  float raw_input[3] = {
    living,
    kitchen,
    bedroom
  };

  for (int i = 0; i < 3; i++) {
    input->data.f[i] = (raw_input[i] - SCALER_MEAN[i]) / SCALER_SCALE[i];
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed!");
    return NO_SIGNAL_INDEX;
  }

  int max_index = 0;
  float max_value = output->data.f[0];

  for (int i = 1; i < 4; i++) {
    if (output->data.f[i] > max_value) {
      max_value = output->data.f[i];
      max_index = i;
    }
  }
  raw_confidence = max_value;

  return max_index;
}

void initTinyML() {
  resolver.AddFullyConnected();
  resolver.AddRelu();
  resolver.AddSoftmax();

  tfl_model = tflite::GetModel(tinyml_mlp_3_8_4_model_tflite);

  static tflite::MicroInterpreter static_interpreter(
    tfl_model,
    resolver,
    tensor_arena,
    TENSOR_ARENA_SIZE
  );

  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed!");
    while (true);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("TinyML model ready.");
}

// ---------------- WiFi and MQTT ----------------
void checkWiFiChannel() {
  int currentChannel = WiFi.channel();

  Serial.print("Master WiFi Channel: ");
  Serial.println(currentChannel);

  if (currentChannel != ESPNOW_CHANNEL) {
    Serial.println("WARNING: WiFi channel mismatch!");
    Serial.print("Master is on channel ");
    Serial.println(currentChannel);
    Serial.print("Slaves are configured for channel ");
    Serial.println(ESPNOW_CHANNEL);
    Serial.println("Fix router 2.4GHz WiFi channel to match ESPNOW_CHANNEL.");
  } else {
    Serial.println("WiFi channel OK. Master and slaves should be on the same channel.");
  }
}

void connectWiFi() {
  Serial.println("WiFi connecting...");

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.disconnect(true);
  delay(1000);

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  unsigned long startAttemptTime = millis();

  while (WiFi.status() != WL_CONNECTED && millis() - startAttemptTime < 20000) {
    delay(500);
    Serial.print(".");
    Serial.print(" status=");
    Serial.println(WiFi.status());
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println();
    Serial.println("WiFi connected.");

    Serial.print("IP: ");
    Serial.println(WiFi.localIP());

    Serial.print("Master MAC: ");
    Serial.println(WiFi.macAddress());

    checkWiFiChannel();
  } 
  else {
    Serial.println();
    Serial.println("WiFi connection FAILED!");
    Serial.print("Final WiFi status: ");
    Serial.println(WiFi.status());
  }
}

void reconnectMQTT() {
  if (mqttClient.connected()) {
    return;
  }

  Serial.print("MQTT connecting... ");

  String clientId = "ESP32PetTracker-";
  clientId += String(random(0xffff), HEX);

  if (mqttClient.connect(clientId.c_str())) {
    Serial.println("connected");
  } 
  else {
    Serial.print("failed, rc=");
    Serial.println(mqttClient.state());
  }
}

void publishMQTT(String payload) {
  if (!mqttClient.connected()) {
    reconnectMQTT();
  }

  mqttClient.loop();

  bool published = mqttClient.publish(MQTT_TOPIC, payload.c_str());

  if (published) {
    Serial.println("MQTT publish success");
  } 
  else {
    Serial.println("MQTT publish failed");
    Serial.print("MQTT state: ");
    Serial.println(mqttClient.state());
    Serial.print("Payload length: ");
    Serial.println(payload.length());
  }

  Serial.print("MQTT payload: ");
  Serial.println(payload);
}

// ---------------- Setup / Loop ----------------

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println();
  Serial.println("===== MASTER STARTING =====");

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);

  connectWiFi();

  mqttClient.setServer(MQTT_SERVER, MQTT_PORT);
  mqttClient.setBufferSize(1024);

  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW init failed!");
    return;
  }

  esp_now_register_recv_cb((esp_now_recv_cb_t)onDataRecv);
  Serial.println("ESP-NOW ready.");

  BLEDevice::init("");

  BLEScan* scanner = BLEDevice::getScan();
  scanner->setAdvertisedDeviceCallbacks(new MyCallbacks());
  scanner->setActiveScan(false);
  scanner->setInterval(160);
  scanner->setWindow(150);

  Serial.println("BLE scanner ready.");

  initTinyML();
  initVoteBuffer();

  window_start = millis();
  budget_window_start = millis();

  Serial.println("Master ready.");
  Serial.println("timestamp_ms,rssi_living,rssi_kitchen,rssi_bedroom,raw_prediction,stable_prediction,confidence,feeding_confirmed,feeding_time_24h_min,time_since_last_feeding_min,anomaly");
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi disconnected. Reconnecting...");
    connectWiFi();
  }

  if (!mqttClient.connected()) {
    reconnectMQTT();
  }

  mqttClient.loop();

  BLEScan* scanner = BLEDevice::getScan();
  scanner->start(BLE_SCAN_SECONDS, false);
  scanner->clearResults();

  if (millis() - window_start > WINDOW_MS) {
    unsigned long now = millis();

    if (now - last_l_time > TIMEOUT_MS) {
      rssi_living = NO_SIGNAL_RSSI;
    }

    if (now - last_k_time > TIMEOUT_MS) {
      rssi_kitchen = NO_SIGNAL_RSSI;
    }

    if (now - last_b_time > TIMEOUT_MS) {
      rssi_bedroom = NO_SIGNAL_RSSI;
    }

    int raw_prediction = NO_SIGNAL_INDEX;
    int stable_prediction = NO_SIGNAL_INDEX;

    bool unreliable = unreliableSignalVector();

    if (unreliable) {
      resetVoteBuffer();
      raw_confidence = 0.0f;
      stable_confidence = 0.0f;
    } 
    else {
      raw_prediction = predictState(
        rssi_living,
        rssi_kitchen,
        rssi_bedroom
      );

      addPredictionToVote(raw_prediction);
      stable_prediction = getMajorityPrediction();

      stable_confidence = raw_confidence;
    }

    updateFeedingBehavior(stable_prediction, now);

    unsigned long feeding_time_min = getMinutes(getCurrentFeedingTime(now));

    unsigned long time_since_last_feeding_min = 0;

    if (last_feeding_seen_time > 0) {
      time_since_last_feeding_min = getMinutes(now - last_feeding_seen_time);
    } 
    else {
      time_since_last_feeding_min = getMinutes(now - budget_window_start);
    }

    bool anomaly = isNoFeedingAnomaly(now);

    Serial.print(now);
    Serial.print(",");
    Serial.print(rssi_living);
    Serial.print(",");
    Serial.print(rssi_kitchen);
    Serial.print(",");
    Serial.print(rssi_bedroom);
    Serial.print(",");
    Serial.print(className(raw_prediction));
    Serial.print(",");
    Serial.print(className(stable_prediction));
    Serial.print(",");
    Serial.print(stable_confidence, 4);
    Serial.print(",");
    Serial.print(feeding_confirmed ? "true" : "false");
    Serial.print(",");
    Serial.print(feeding_time_min);
    Serial.print(",");
    Serial.print(time_since_last_feeding_min);
    Serial.print(",");
    Serial.println(anomaly ? "NO_FEEDING_VISIT_ANOMALY" : "normal");

    String payload = "{";
    payload += "\"timestamp_ms\":";
    payload += String(now);
    payload += ",";

    payload += "\"rssi_living\":";
    payload += String(rssi_living);
    payload += ",";

    payload += "\"rssi_kitchen\":";
    payload += String(rssi_kitchen);
    payload += ",";

    payload += "\"rssi_bedroom\":";
    payload += String(rssi_bedroom);
    payload += ",";

    payload += "\"raw_prediction\":\"";
    payload += className(raw_prediction);
    payload += "\",";

    payload += "\"stable_prediction\":\"";
    payload += className(stable_prediction);
    payload += "\",";

    payload += "\"confidence\":";
    payload += String(stable_confidence, 4);
    payload += ",";

    payload += "\"feeding_confirmed\":";
    payload += feeding_confirmed ? "true" : "false";
    payload += ",";

    payload += "\"feeding_time_24h_min\":";
    payload += String(feeding_time_min);
    payload += ",";

    payload += "\"time_since_last_feeding_min\":";
    payload += String(time_since_last_feeding_min);
    payload += ",";

    payload += "\"anomaly\":\"";
    payload += anomaly ? "NO_FEEDING_VISIT_ANOMALY" : "normal";
    payload += "\"";

    payload += "}";

    publishMQTT(payload);

    window_start = millis();

    Serial.println("----- CURRENT STATUS -----");
    Serial.print("Living RSSI: ");
    Serial.println(rssi_living);

    Serial.print("Kitchen RSSI: ");
    Serial.println(rssi_kitchen);

    Serial.print("Bedroom RSSI: ");
    Serial.println(rssi_bedroom);

    Serial.print("WiFi Channel: ");
    Serial.println(WiFi.channel());

    Serial.print("MQTT State: ");
    Serial.println(mqttClient.state());

    Serial.println("--------------------------");
  }

  delay(100);
}