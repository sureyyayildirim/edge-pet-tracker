#include <WiFi.h>
#include <esp_now.h>
#include <BLEDevice.h>
#include <BLEScan.h>

#include "model_data.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ---------------- BLE / ESP-NOW ----------------

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

void onDataRecv(const uint8_t *mac, const uint8_t *data, int len) {
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

class MyCallbacks : public BLEAdvertisedDeviceCallbacks {
  void onResult(BLEAdvertisedDevice d) {
    if (d.getAddress().toString() == TARGET_MAC) {
      rssi_living = d.getRSSI();
      l_ok = true;
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

const char* className(int index) {
  switch (index) {
    case 0: return "living_room";
    case 1: return "kitchen";
    case 2: return "bedroom";
    case 3: return "feeding_area";
    default: return "unknown";
  }
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
    return -1;
  }

  int max_index = 0;
  float max_value = output->data.f[0];

  for (int i = 1; i < 4; i++) {
    if (output->data.f[i] > max_value) {
      max_value = output->data.f[i];
      max_index = i;
    }
  }

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

// ---------------- Setup / Loop ----------------

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

  initTinyML();

  Serial.println("timestamp_ms,rssi_living,rssi_kitchen,rssi_bedroom,prediction");

  window_start = millis();
}

void loop() {
  BLEDevice::getScan()->start(1, false);

  if (millis() - window_start > WINDOW_MS) {

    if (!l_ok) {
      rssi_living = -110;
    }

    if (millis() - last_k_time > TIMEOUT_MS) {
      rssi_kitchen = -110;
    }

    if (millis() - last_b_time > TIMEOUT_MS) {
      rssi_bedroom = -110;
    }

    int prediction = predictState(
      rssi_living,
      rssi_kitchen,
      rssi_bedroom
    );

    Serial.print(",");
    Serial.print(rssi_living);
    Serial.print(",");
    Serial.print(rssi_kitchen);
    Serial.print(",");
    Serial.print(rssi_bedroom);
    Serial.print(",");
    Serial.println(className(prediction));

    l_ok = false;
    window_start = millis();
  }

  BLEDevice::getScan()->clearResults();
  delay(100);
}
