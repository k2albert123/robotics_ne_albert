# Face Locking Recognition System

A real-time face recognition project for enrolling people, recognizing them from a camera, locking onto a selected face, logging face-lock actions, and optionally steering an ESP8266 servo camera over MQTT.

The project is CPU-first. It works with the normal `onnxruntime` package, so people without GPUs do not need CUDA, cuDNN, or `onnxruntime-gpu`.

## Project Layout

- `src/enroll.py` - capture face samples and build the face database.
- `src/recognize.py` - run local face recognition and face locking.
- `src/rebuild_db.py` - rebuild `data/db/face_db.npz` from existing enrollment crops.
- `addons/mqtt_servo_tracking/recognize_mqtt.py` - face locking plus MQTT movement/status publishing.
- `addons/mqtt_servo_tracking/esp8266/face_tracker_servo/face_tracker_servo.ino` - ESP8266 servo firmware.
- `dashboard/index.html` - static MQTT dashboard for live movement and lock status.
- `logs/` - action history files created during face-lock sessions.

## Requirements

Use Python 3.10, 3.11, 3.12, or 3.13. Python 3.14 is not recommended because some computer-vision packages may not have wheels for it yet.

Install dependencies from the repo root:

```bash
pip install -r requirements.txt
```

The included requirements use:

```text
onnxruntime
```

That is the CPU ONNX Runtime package. If only CPU ONNX Runtime is installed, the recognizer automatically uses `CPUExecutionProvider`.

Required model files:

- `models/embedder_arcface.onnx`
- `models/face_landmarker.task`

## Quick Start

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Put the required model files in `models/`.

3. Enroll a person:

   ```bash
   python -m src.enroll
   ```

   Controls:

   - `SPACE` - capture one sample.
   - `a` - toggle auto-capture.
   - `s` - save enrollment to the database.
   - `q` - quit.

4. Rebuild the database if you already have crops in `data/enroll/`:

   ```bash
   python -m src.rebuild_db
   ```

5. Run local recognition:

   ```bash
   python -m src.recognize
   ```

   Controls:

   - `+` or `=` - increase the recognition distance threshold.
   - `-` - decrease the threshold.
   - `r` - reload the face database.
   - `d` - toggle debug overlay.
   - `l` - lock or unlock the selected recognized face.
   - `q` - quit.

## CPU And GPU Notes

For CPU-only users, keep `onnxruntime` in `requirements.txt`. No separate CPU-only Python files are needed.

When the app starts, it checks ONNX Runtime providers. If only CPU is available, it selects CPU automatically. If GPU-capable ONNX Runtime packages are installed, it prompts for a provider and keeps CPU as a fallback.

Optional GPU setups:

- NVIDIA CUDA: use `onnxruntime-gpu` in a GPU-specific environment.
- Windows DirectML: use `onnxruntime-directml` in a GPU-specific environment.

Avoid installing multiple ONNX Runtime variants into the same environment unless you know they are compatible.

## Face Locking

During recognition, press `l` when a known face is selected. The app locks onto that identity, tracks movement, and records actions such as:

- Face locked or unlocked.
- Head moved left or right.
- Smile or blink events detected from landmarks.
- Face temporarily lost or reacquired.

History files are written to `logs/` as:

```text
[Name]_history_[timestamp].txt
```

Example line:

```text
2026-01-31 13:20:20.225528 - HEAD_RIGHT: Moved right by 31.9px
```

## MQTT Servo Addon

The MQTT addon keeps the original recognizer separate and publishes servo commands for the ESP8266.

Run it from the repo root:

```bash
python addons/mqtt_servo_tracking/recognize_mqtt.py
```

Default MQTT settings:

- Broker: `broker.hivemq.com`
- MQTT port: `1883`
- Browser WebSocket URL: `ws://broker.hivemq.com:8000/mqtt`
- Movement topic: `vision/Dieudonne/ne/movement`
- Status topic: `vision/Dieudonne/ne/status`

Movement payloads on `vision/Dieudonne/ne/movement`:

- `LEFT` - locked face is left of frame center.
- `RIGHT` - locked face is right of frame center.
- `CENTER` - locked face is centered; the servo holds its current angle.
- `SEARCH` - locked face is missing, sweep the servo.
- `IDLE` - no active face lock.

The firmware also accepts `HOME` for manual recentering; the Python tracker does not publish it automatically.

Dashboard JSON is published on `vision/Dieudonne/ne/status`, including movement, lock state, target name, face count, horizontal error, FPS, threshold, and provider.

Useful addon flags:

```bash
python addons/mqtt_servo_tracking/recognize_mqtt.py --mqtt-broker broker.hivemq.com --mqtt-topic vision/Dieudonne/ne/movement --mqtt-status-topic vision/Dieudonne/ne/status --camera-width 1280 --camera-height 720 --max-faces 3 --detect-every 2 --recognize-every 3 --deadzone-px 80 --center-exit-hysteresis-px 30 --command-hold-sec 0.25 --search-delay-sec 0.8 --reacquire-hold-sec 0.30 --command-confirm-frames 2 --mqtt-min-interval 0.15 --mqtt-status-min-interval 0.25
```

## Dashboard

Open:

```text
dashboard/index.html
```

The dashboard is a static HTML file. It uses MQTT over WebSockets and defaults to:

```text
ws://broker.hivemq.com:8000/mqtt
```

It listens to:

- `vision/Dieudonne/ne/movement`
- `vision/Dieudonne/ne/status`

The JSON status topic is authoritative for the displayed command. The raw movement topic is used only as a fallback if status messages stop arriving, which prevents delayed MQTT movement messages from making the dashboard flicker between commands.

The page includes editable connection fields, so you can change the WebSocket URL or topics without editing the file.

The dashboard connects via MQTT over WebSockets at `ws://broker.hivemq.com:8000/mqtt`. Plain MQTT port `1883` is for Python and ESP8266/ESP32 clients, not browsers.

## ESP8266 Servo Setup

1. Open `addons/mqtt_servo_tracking/esp8266/face_tracker_servo/face_tracker_servo.ino`.
2. Set `WIFI_SSID` and `WIFI_PASSWORD`.
3. Confirm:

   ```cpp
   MQTT_SERVER = "broker.hivemq.com";
   MQTT_TOPIC = "vision/Dieudonne/ne/movement";
   ```

4. Adjust servo settings for your hardware:

   - `SERVO_PIN`
   - `SERVO_MIN_ANGLE`
   - `SERVO_MAX_ANGLE`
   - `REVERSE_SERVO`

5. Install Arduino libraries:

   - `PubSubClient`
   - `Servo` from the ESP8266 core

6. Upload with `arduino-cli`:

   ```powershell
   powershell -ExecutionPolicy Bypass -File addons/mqtt_servo_tracking/esp8266/upload.ps1 -Port COM5
   ```

If your board is not NodeMCU v2, pass a different FQBN:

```powershell
powershell -ExecutionPolicy Bypass -File addons/mqtt_servo_tracking/esp8266/upload.ps1 -Port COM5 -Fqbn esp8266:esp8266:d1_mini
```

## Tuning Tips

- CPU-first defaults use `640x480`, `--max-faces 3`, `--detect-every 2`, and `--recognize-every 3`.
- Run with `--profile` to show frame, detection, and recognition timing in the recognition window.
- Increase `--detect-every` or `--recognize-every` if CPU usage is still too high.
- Lower `--max-faces` or use `--locked-max-faces 1` for smoother locked-face tracking.
- Increase `--deadzone-px` if the servo moves while the face is already centered.
- Increase `--command-hold-sec` if the dashboard or servo still reacts to short command blips.
- Increase `--search-delay-sec` if brief recognition drops trigger `SEARCH` too quickly.
- Increase `--command-confirm-frames` if `LEFT` and `RIGHT` flicker.
- Lower the recognition threshold if false positives happen.
- Raise the recognition threshold if known faces are not accepted.

## Troubleshooting

- Empty database: run `python -m src.enroll` or `python -m src.rebuild_db`.
- Camera not available: check the camera index in the recognizer code if your webcam is not device `1`.
- Dashboard offline: confirm the broker exposes MQTT over WebSockets at `ws://broker.hivemq.com:8000/mqtt`.
- ESP8266 not moving: confirm Wi-Fi credentials, broker address, topic, and Serial Monitor output.
- CPU-only machine: keep `onnxruntime`; do not install `onnxruntime-gpu`.
