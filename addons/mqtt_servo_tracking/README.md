# MQTT Servo Face Tracking Addon

This addon runs the same face recognition and face-locking pipeline as the main app, then publishes MQTT messages for a servo-mounted ESP8266 camera and the browser dashboard.

The original `src/recognize.py` remains separate.

## Files

- `recognize_mqtt.py` - face recognition, face lock, MQTT movement commands, and dashboard status JSON.
- `esp8266/face_tracker_servo/face_tracker_servo.ino` - ESP8266 firmware that subscribes to movement commands.
- `esp8266/upload.ps1` - helper script for compiling and uploading with `arduino-cli`.
- `../../dashboard/index.html` - static MQTT dashboard.

## MQTT Topics

Default broker:

```text
157.173.101.159
```

Ports:

- `1883` - plain MQTT for Python and ESP8266.
- `9001` - MQTT over WebSockets for the browser dashboard.

Movement topic:

```text
vision/teamalpha/movement
```

Movement payloads:

- `LEFT` - locked face is left of frame center.
- `RIGHT` - locked face is right of frame center.
- `CENTER` - locked face is centered.
- `SEARCH` - locked face is missing, sweep the servo.
- `IDLE` - no active face lock.

Dashboard status topic:

```text
vision/teamalpha/status
```

Status payload shape:

```json
{
  "timestamp": 1781190000.0,
  "movement": "LEFT",
  "error_x": -124.5,
  "locked": true,
  "target": "Dieudonne",
  "locked_face_found": true,
  "faces": 1,
  "fps": 18.7,
  "threshold": 0.4,
  "provider": "CPU"
}
```

## Python Setup

From the repo root:

```bash
pip install -r requirements.txt
```

The default dependency is `onnxruntime`, which runs on CPU. No GPU is required.

Run the addon:

```bash
python addons/mqtt_servo_tracking/recognize_mqtt.py
```

Optional flags:

```bash
python addons/mqtt_servo_tracking/recognize_mqtt.py --mqtt-broker 157.173.101.159 --mqtt-port 1883 --mqtt-topic vision/teamalpha/movement --mqtt-status-topic vision/teamalpha/status --deadzone-px 80 --center-exit-hysteresis-px 30 --error-smooth-alpha 0.35 --search-delay-sec 0.8 --command-confirm-frames 2 --mqtt-min-interval 0.15 --mqtt-status-min-interval 0.25
```

Use `--disable-mqtt` to run the recognizer without publishing MQTT messages.

## Dashboard

Open:

```text
dashboard/index.html
```

The dashboard connects directly to MQTT over WebSockets:

```text
ws://157.173.101.159:9001
```

It subscribes to both the movement topic and the dashboard status topic. If your broker uses a different WebSocket port or path, change it in the dashboard input field and reconnect.

Plain MQTT port `1883` cannot be used directly by a browser.

## ESP8266 Setup

1. Open `esp8266/face_tracker_servo/face_tracker_servo.ino`.
2. Set:

   ```cpp
   const char* WIFI_SSID = "your-wifi";
   const char* WIFI_PASSWORD = "your-password";
   ```

3. Confirm the MQTT settings:

   ```cpp
   const char* MQTT_SERVER = "157.173.101.159";
   const uint16_t MQTT_PORT = 1883;
   const char* MQTT_TOPIC = "vision/teamalpha/movement";
   ```

4. Tune the servo constants:

   - `SERVO_PIN`
   - `SERVO_MIN_ANGLE`
   - `SERVO_MAX_ANGLE`
   - `SERVO_CENTER_ANGLE`
   - `TRACK_STEP`
   - `SEARCH_STEP`
   - `REVERSE_SERVO`

5. Install Arduino libraries:

   - `PubSubClient`
   - `Servo` from the ESP8266 core

6. Upload:

   ```powershell
   powershell -ExecutionPolicy Bypass -File addons/mqtt_servo_tracking/esp8266/upload.ps1 -Port COM5
   ```

For a different ESP8266 board:

```powershell
powershell -ExecutionPolicy Bypass -File addons/mqtt_servo_tracking/esp8266/upload.ps1 -Port COM5 -Fqbn esp8266:esp8266:d1_mini
```

## Tuning

- Increase `--deadzone-px` if the servo keeps moving near center.
- Increase `--center-exit-hysteresis-px` if the servo oscillates around center.
- Increase `--search-delay-sec` if short recognition drops trigger sweeping.
- Increase `--command-confirm-frames` for steadier movement at the cost of slower response.
- Increase `--mqtt-min-interval` if too many repeated movement commands are sent.

## Troubleshooting

- Python publishes but ESP8266 does not move: check ESP Serial Monitor, Wi-Fi credentials, broker, topic, and servo wiring.
- Dashboard stays offline: confirm MQTT over WebSockets is enabled at `ws://157.173.101.159:9001`.
- Recognizer is slow on CPU: lower camera resolution in `recognize_mqtt.py`.
- Known faces show as unknown: enroll more samples or increase the recognition threshold with `+`.
