# 🛡 ThreatVision – Live Threat Detection System

A real-time AI-powered surveillance dashboard that uses **YOLOv8** to detect threatening objects (knives, scissors) in a live camera feed, displays a rich web UI, saves timestamped screenshots, and fires desktop notifications.

---

## 📸 System Overview

```
USB Camera / IP Stream
        │
        ▼
 OpenCV Frame Capture  ──►  YOLOv8 Nano Inference
        │                          │
        │                   Threat Detected?
        │                     YES ──────────►  Red bounding box drawn
        │                                  ►  Screenshot saved to /alerts
        │                                  ►  Desktop notification sent
        │                                  ►  Alert logged to CSV
        ▼
  Flask MJPEG Stream  ──►  Browser Dashboard
                              (live feed + stats + alert log)
```

---

## 🗂 Project Structure

```
SURVIELLANCE SYSTEM/
├── app.py              # Flask web server & REST API
├── detector.py         # Core YOLO detection engine (runs in background thread)
├── config.yaml         # All runtime settings – edit this to customize
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Dashboard UI (dark theme, live updates)
├── alerts/             # Auto-created; timestamped alert screenshots go here
│   └── alert_log.csv   # Running CSV log of every alert
└── README.md           # This file
```

---

## ⚙ Configuration (`config.yaml`)

All settings are in `config.yaml` — no code changes needed.

| Key | Default | Description |
|---|---|---|
| `camera.source` | `0` | USB camera index (`0` = first camera). Use an RTSP URL for IP cameras. |
| `detection.model` | `yolov8n.pt` | YOLOv8 weights. Nano is fastest; swap for `yolov8s.pt` etc. |
| `detection.threat_classes` | `[knife, scissors]` | COCO class names to flag as threats. |
| `detection.confidence_threshold` | `0.40` | Minimum confidence (0–1) to count a detection. |
| `detection.alert_cooldown` | `5` | Seconds between saving consecutive alerts for the same class. |
| `alerts.output_dir` | `alerts` | Directory for saved screenshots (relative to project root). |
| `server.host` | `127.0.0.1` | Server bind address. Use `0.0.0.0` for LAN access. |
| `server.port` | `5000` | Web server port. |

### Switching to an IP Camera

Change `camera.source` in `config.yaml`:
```yaml
camera:
  source: "rtsp://admin:password@192.168.1.100:554/stream1"
```

---

## 🚀 Setup & Running

### 1. Prerequisites

- **Python 3.9+** (3.10 / 3.11 recommended)
- A webcam (USB index 0) or an accessible RTSP stream
- Windows 10+ (for desktop notifications via `plyer`)

### 2. Install Dependencies

```powershell
# In the project folder
pip install -r requirements.txt
```

> **First run**: `ultralytics` will automatically download `yolov8n.pt` (~6 MB) from the internet the first time it is loaded.

### 3. Start the Server

```powershell
python app.py
```

You will see:
```
HH:MM:SS [INFO] Loading YOLO model: yolov8n.pt
HH:MM:SS [INFO] Threat detector ready. Watching for: {'knife', 'scissors'}
HH:MM:SS [INFO] Camera opened: 0
 * Running on http://127.0.0.1:5000
```

### 4. Open the Dashboard

Navigate to: **http://127.0.0.1:5000**

The camera feed starts automatically. Use the **Start / Stop** buttons to control detection.

---

## 🖥 Dashboard Features

| Feature | Description |
|---|---|
| 📹 Live Feed | MJPEG stream with bounding boxes drawn directly on frames |
| 🔴 Red boxes | Drawn around detected threats (knife / scissors) |
| 🟢 Green boxes | Non-threatening detected objects |
| ⚠ HUD overlay | Timestamp and "THREAT DETECTED" flashed on frame |
| 📊 Stats cards | Frames processed, threats found, alerts saved, uptime |
| 🚨 Alert log | Scrollable list of all alerts with thumbnail previews |
| 🔔 Web toast | Browser toast notification on each new threat |
| 🔕 Desktop notification | OS-level notification via `plyer` |

---

## 🔌 REST API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Dashboard HTML |
| `GET` | `/video_feed` | MJPEG live stream |
| `GET` | `/api/status` | JSON: frames, threats, alerts, uptime, is_running |
| `GET` | `/api/alerts` | JSON: paginated alert list |
| `POST` | `/api/control` | `{"action": "start"}` or `{"action": "stop"}` |
| `GET` | `/alerts/<file>` | Serve saved screenshot images |

---

## 📂 Alert Files

Every time a threat is detected (respecting the cooldown), the system:

1. Saves a JPEG to `alerts/alert_<class>_<timestamp>.jpg`
2. Appends a row to `alerts/alert_log.csv` with columns:
   - `timestamp` – ISO 8601 time
   - `class` – `knife` or `scissors`
   - `confidence` – model confidence (0–1)
   - `screenshot` – filename of saved image

---

## 🛠 Troubleshooting

| Problem | Solution |
|---|---|
| `Cannot open video source: 0` | Camera index wrong; try `1` or `2` in `config.yaml`, or check camera permissions |
| Low detection accuracy | Lower `confidence_threshold` to `0.25`, or use a larger model (`yolov8s.pt`) |
| No desktop notifications | Install `plyer`: `pip install plyer`. On Windows, check notification settings. |
| Slow frame rate | Use `yolov8n.pt` (default) or set a lower resolution in the camera driver |
| Port already in use | Change `server.port` in `config.yaml` |
| ImportError for ultralytics | Run `pip install ultralytics --upgrade` |

---

## 🧠 Adding More Threat Classes

Edit `config.yaml`:
```yaml
detection:
  threat_classes:
    - "knife"
    - "scissors"
    - "gun"        # add any COCO class name
    - "baseball bat"
```

For a full list of COCO class names, see: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml

---

## 📜 License

MIT — free to use, modify, and distribute.
