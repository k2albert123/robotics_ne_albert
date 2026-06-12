#define USE_US_TIMER

#include <WiFi.h>
#include <PubSubClient.h>
#include <ESP32Servo.h>

const char* WIFI_SSID = "Wokwi-GUEST";
const char* WIFI_PASSWORD = "";
// =========================
// MQTT Settings
// =========================
const char* MQTT_SERVER = "broker.hivemq.com";
const uint16_t MQTT_PORT = 1883;

const char* MQTT_TOPIC = "vision/Dieudonne/ne/movement";
const char* MQTT_CLIENT_ID_PREFIX = "teamalpha-face-servo";
const IPAddress MQTT_FALLBACK_IPS[] = {
  IPAddress(3, 126, 147, 153),
  IPAddress(3, 124, 122, 176),
  IPAddress(18, 197, 232, 142),
  IPAddress(3, 123, 123, 192),
  IPAddress(3, 120, 204, 188)
};
const uint8_t MQTT_FALLBACK_IP_COUNT =
    sizeof(MQTT_FALLBACK_IPS) / sizeof(MQTT_FALLBACK_IPS[0]);

// =========================
// Servo Configuration
// =========================
const uint8_t SERVO_PIN = 18; // Recommended ESP32 pin

const int SERVO_MIN_ANGLE = 0;
const int SERVO_MAX_ANGLE = 180;
const int SERVO_CENTER_ANGLE = 90;
const int SERVO_MIN_PULSE_US = 500;
const int SERVO_MAX_PULSE_US = 2400;

const float TRACK_STEP = 0.55f;
const float SCAN_STEP = 0.50f;

const unsigned long TRACK_INTERVAL_MS = 14;
const unsigned long SCAN_INTERVAL_MS = 24;
const unsigned long COMMAND_TIMEOUT_MS = 800;

const bool REVERSE_SERVO = true;

// =========================
// Command Types
// =========================
enum MovementCommand {
  CMD_IDLE,
  CMD_LEFT,
  CMD_RIGHT,
  CMD_CENTER,
  CMD_HOME,
  CMD_SCAN
};

// =========================
// Global Objects
// =========================
WiFiClient wifiClient;
PubSubClient mqttClient(wifiClient);
Servo panServo;

// =========================
// State Variables
// =========================
MovementCommand currentCommand = CMD_IDLE;

float servoAngle = SERVO_CENTER_ANGLE;
int sweepDirection = 1;

unsigned long lastMoveAt = 0;
unsigned long lastReconnectAttempt = 0;
unsigned long lastCommandAt = 0;
String mqttClientId = "";

// ======================================================
// Servo Functions
// ======================================================

int angleToPulse(float angle) {
  float normalized =
      (angle - SERVO_MIN_ANGLE) /
      float(SERVO_MAX_ANGLE - SERVO_MIN_ANGLE);

  return SERVO_MIN_PULSE_US +
      int((normalized * (SERVO_MAX_PULSE_US - SERVO_MIN_PULSE_US)) + 0.5f);
}

void setServoAngle(float angle) {
  if (angle < SERVO_MIN_ANGLE) {
    angle = SERVO_MIN_ANGLE;
  }

  if (angle > SERVO_MAX_ANGLE) {
    angle = SERVO_MAX_ANGLE;
  }

  servoAngle = angle;
  panServo.writeMicroseconds(angleToPulse(servoAngle));
}

void applyTrackingStep(int logicalDirection) {

  int direction =
      REVERSE_SERVO ? -logicalDirection : logicalDirection;

  setServoAngle(
      servoAngle + (direction * TRACK_STEP)
  );
}

// ======================================================
// Command Parsing
// ======================================================

MovementCommand parseCommand(String message) {

  message.trim();
  message.toUpperCase();

  // Allow CMD_LEFT style
  if (message.startsWith("CMD_")) {
    message = message.substring(4);
  }

  if (message == "LEFT") {
    return CMD_LEFT;
  }

  if (message == "RIGHT") {
    return CMD_RIGHT;
  }

  if (message == "CENTER") {
    return CMD_CENTER;
  }

  if (message == "HOME") {
    return CMD_HOME;
  }

  if (message == "SCAN") {
    return CMD_SCAN;
  }

  if (message == "IDLE") {
    return CMD_IDLE;
  }

  return CMD_IDLE;
}

// ======================================================
// MQTT Callback
// ======================================================

void mqttCallback(
    char* topic,
    byte* payload,
    unsigned int length
) {

  String message = "";

  for (unsigned int i = 0; i < length; i++) {
    message += (char)payload[i];
  }

  currentCommand = parseCommand(message);

  lastCommandAt = millis();

  Serial.print("[MQTT] Received: ");
  Serial.println(message);
}

// ======================================================
// Serial Input
// ======================================================

void handleSerial() {

  if (Serial.available() > 0) {

    String input =
        Serial.readStringUntil('\n');

    input.trim();

    MovementCommand newCmd =
        parseCommand(input);

    currentCommand = newCmd;

    lastCommandAt = millis();

    Serial.print("[SERIAL] Executing: ");
    Serial.println(input);
  }
}

// ======================================================
// Wi-Fi Connection
// ======================================================

void connectWiFi() {

  if (WiFi.status() == WL_CONNECTED) {
    return;
  }

  Serial.println("[WiFi] Connecting...");

  WiFi.mode(WIFI_STA);

  // Disconnect old attempts first
  WiFi.disconnect(true);

  delay(1000);

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  unsigned long startAttemptTime = millis();

  while (
      WiFi.status() != WL_CONNECTED &&
      millis() - startAttemptTime < 15000
  ) {

    delay(500);
    Serial.print(".");
  }

  if (WiFi.status() == WL_CONNECTED) {

    Serial.println();
    Serial.println("[WiFi] Connected!");

    Serial.print("[WiFi] IP Address: ");
    Serial.println(WiFi.localIP());
    Serial.print("[WiFi] Gateway: ");
    Serial.println(WiFi.gatewayIP());
    Serial.print("[WiFi] DNS: ");
    Serial.println(WiFi.dnsIP());
    Serial.print("[WiFi] RSSI: ");
    Serial.print(WiFi.RSSI());
    Serial.println(" dBm");

  } else {

    Serial.println();
    Serial.println("[WiFi] Connection Failed");
  }
}
// ======================================================
// MQTT Connection
// ======================================================

bool connectMqtt() {

  if (mqttClient.connected()) {
    return true;
  }

  if (WiFi.status() != WL_CONNECTED) {
    return false;
  }

  if (millis() - lastReconnectAttempt < 5000) {
    return false;
  }

  lastReconnectAttempt = millis();

  IPAddress brokerIp;
  bool dnsResolved = false;
  bool connectedProbe = false;

  Serial.print("[MQTT] Resolving ");
  Serial.print(MQTT_SERVER);
  Serial.print("...");
  if (WiFi.hostByName(MQTT_SERVER, brokerIp)) {
    dnsResolved = true;
    Serial.print(" ");
    Serial.println(brokerIp);
  } else {
    Serial.println(" DNS failed");
  }

  for (uint8_t attempt = 0; attempt <= MQTT_FALLBACK_IP_COUNT; attempt++) {
    if (!dnsResolved || attempt > 0) {
      uint8_t fallbackIndex = dnsResolved ? attempt - 1 : attempt;
      if (fallbackIndex >= MQTT_FALLBACK_IP_COUNT) {
        break;
      }
      brokerIp = MQTT_FALLBACK_IPS[fallbackIndex];
      Serial.print("[MQTT] DNS fallback IP ");
      Serial.println(brokerIp);
    }

    WiFiClient probe;
    Serial.print("[MQTT] TCP probe ");
    Serial.print(brokerIp);
    Serial.print(":");
    Serial.print(MQTT_PORT);
    Serial.print("...");
    if (probe.connect(brokerIp, MQTT_PORT)) {
      Serial.println(" ok");
      probe.stop();
      connectedProbe = true;
      break;
    }

    Serial.println(" failed");
    probe.stop();
  }

  if (!connectedProbe) {
    return false;
  }

  mqttClient.setServer(brokerIp, MQTT_PORT);
  Serial.print("[MQTT] Connecting as ");
  Serial.print(mqttClientId);
  Serial.print("...");

  bool connected =
      mqttClient.connect(mqttClientId.c_str());

  if (!connected) {

    Serial.print("Failed, rc=");
    Serial.println(mqttClient.state());

    return false;
  }

  Serial.println("Connected");

  mqttClient.subscribe(MQTT_TOPIC);

  Serial.print("[MQTT] Subscribed to: ");
  Serial.println(MQTT_TOPIC);

  return true;
}

// ======================================================
// Servo Logic
// ======================================================

void handleServo() {

  unsigned long now = millis();

  // Auto idle timeout
  if ((now - lastCommandAt) >
      COMMAND_TIMEOUT_MS) {

    currentCommand = CMD_IDLE;
  }

  // ----------------------
  // CENTER means the face is centered, so hold the current servo angle.
  // ----------------------
  if (currentCommand == CMD_CENTER || currentCommand == CMD_IDLE) {
    return;
  }

  // ----------------------
  // HOME
  // ----------------------
  if (currentCommand == CMD_HOME) {

    setServoAngle(SERVO_CENTER_ANGLE);

    currentCommand = CMD_IDLE;

    return;
  }

  // ----------------------
  // SCAN MODE
  // ----------------------
  if (currentCommand == CMD_SCAN) {

    if (
        now - lastMoveAt <
        SCAN_INTERVAL_MS
    ) {
      return;
    }

    lastMoveAt = now;

    setServoAngle(
        servoAngle +
        (sweepDirection * SCAN_STEP)
    );

    if (servoAngle >= SERVO_MAX_ANGLE) {
      sweepDirection = -1;
    }

    if (servoAngle <= SERVO_MIN_ANGLE) {
      sweepDirection = 1;
    }

    return;
  }

  // ----------------------
  // LEFT / RIGHT TRACKING
  // ----------------------
  if (
      now - lastMoveAt <
      TRACK_INTERVAL_MS
  ) {
    return;
  }

  lastMoveAt = now;

  if (currentCommand == CMD_LEFT) {

    applyTrackingStep(-1);

  } else if (
      currentCommand == CMD_RIGHT
  ) {

    applyTrackingStep(1);
  }
}

// ======================================================
// Setup
// ======================================================

void setup() {

  Serial.begin(115200);

  delay(500);

  Serial.println();
  Serial.println(
      "[SYS] ESP32 Face Servo Initializing..."
  );
  mqttClientId =
      String(MQTT_CLIENT_ID_PREFIX) +
      "-" +
      String((uint32_t)ESP.getEfuseMac(), HEX);
  Serial.print("[MQTT] Client ID: ");
  Serial.println(mqttClientId);

  // ----------------------
  // ESP32 Servo Setup
  // ----------------------
  ESP32PWM::allocateTimer(0);

  panServo.setPeriodHertz(50);

  panServo.attach(
      SERVO_PIN,
      SERVO_MIN_PULSE_US,
      SERVO_MAX_PULSE_US
  );

  setServoAngle(SERVO_CENTER_ANGLE);

  // ----------------------
  // MQTT Setup
  // ----------------------
  mqttClient.setServer(
      MQTT_SERVER,
      MQTT_PORT
  );

  mqttClient.setCallback(mqttCallback);

  // ----------------------
  // WiFi
  // ----------------------
  connectWiFi();

  lastCommandAt = millis();
}

// ======================================================
// Main Loop
// ======================================================

void loop() {

  // Wi-Fi reconnect
  if (WiFi.status() != WL_CONNECTED) {
    connectWiFi();
  }

  // MQTT reconnect
  if (!mqttClient.connected()) {
    connectMqtt();
  }

  mqttClient.loop();

  // Serial commands
  handleSerial();

  // Servo movement
  handleServo();

  delay(1);
}
