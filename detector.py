"""
detector.py
-----------
Core detection engine.
  - Opens the configured video source with OpenCV
  - Runs YOLOv8 inference on every frame
  - Flags threat classes (knife / scissors)
  - Saves timestamped alert screenshots
  - Fires a desktop notification (Windows toast / Linux libnotify)
  - Exposes a thread-safe JPEG frame generator for the Flask stream
"""

from __future__ import annotations

import csv
import io
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────
# Desktop notification helper (cross-platform best-effort)
# ──────────────────────────────────────────────────────────────
def _notify(title: str, message: str) -> None:
    """Fire a desktop notification without crashing the main thread."""
    try:
        import platform
        system = platform.system()
        if system == "Windows":
            # plyer is the preferred backend; fall back to win10toast
            try:
                from plyer import notification
                notification.notify(
                    title=title,
                    message=message,
                    app_name="Threat Detector",
                    timeout=5,
                )
            except Exception:
                try:
                    from win10toast import ToastNotifier
                    ToastNotifier().show_toast(title, message, duration=5, threaded=True)
                except Exception:
                    # Last resort: Windows balloon via ctypes
                    import ctypes
                    ctypes.windll.user32.MessageBeep(0xFFFFFFFF)
        elif system == "Darwin":
            os.system(f'osascript -e \'display notification "{message}" with title "{title}"\'')
        else:
            os.system(f'notify-send "{title}" "{message}"')
    except Exception as exc:
        logging.warning("Desktop notification failed: %s", exc)


# ──────────────────────────────────────────────────────────────
# Detector
# ──────────────────────────────────────────────────────────────
class ThreatDetector:
    """Runs in a background thread; provides JPEG frames + alert log."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        self.cfg = self._load_config(config_path)
        self._lock = threading.Lock()
        self._latest_frame: bytes | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._alert_log: list[dict] = []          # in-memory alert history
        self._last_alert_time: dict[str, float] = {}  # class → epoch time
        self._stats = {
            "frames_processed": 0,
            "threats_detected": 0,
            "alerts_saved": 0,
            "uptime_start": None,
        }

        # Ensure alerts directory exists
        self.alerts_dir = Path(self.cfg["alerts"]["output_dir"])
        self.alerts_dir.mkdir(parents=True, exist_ok=True)

        # CSV log file
        self._csv_path = self.alerts_dir / "alert_log.csv"
        if not self._csv_path.exists():
            with open(self._csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["timestamp", "class", "confidence", "screenshot"])
                writer.writeheader()

        logging.info("Loading YOLO model: %s", self.cfg["detection"]["model"])
        self.model = YOLO(self.cfg["detection"]["model"])
        self.threat_classes: set[str] = set(self.cfg["detection"]["threat_classes"])
        self.conf_threshold: float = float(self.cfg["detection"]["confidence_threshold"])
        self.alert_cooldown: float = float(self.cfg["detection"]["alert_cooldown"])
        self.inference_imgsz: int = int(self.cfg["detection"].get("inference_imgsz", 640))

        # CLAHE for contrast enhancement (helps detect low-contrast knives)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Raw detection log for debug panel (last 50 unique class sightings)
        self._raw_detections: list[dict] = []

        logging.info(
            "Threat detector ready. Watching for: %s (conf ≥ %.2f)",
            self.threat_classes, self.conf_threshold,
        )

    # ── config ────────────────────────────────────────────────
    @staticmethod
    def _load_config(path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    # ── public API ────────────────────────────────────────────
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._stats["uptime_start"] = time.time()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logging.info("Detection thread started.")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logging.info("Detection thread stopped.")

    def get_jpeg_frame(self) -> bytes | None:
        with self._lock:
            return self._latest_frame

    def get_alerts(self) -> list[dict]:
        with self._lock:
            return list(self._alert_log)

    def get_stats(self) -> dict:
        with self._lock:
            stats = dict(self._stats)
        if stats["uptime_start"]:
            stats["uptime_seconds"] = int(time.time() - stats["uptime_start"])
        else:
            stats["uptime_seconds"] = 0
        return stats

    def is_running(self) -> bool:
        return self._running

    def get_raw_detections(self) -> list[dict]:
        """Return last seen detections across ALL classes (for debug panel)."""
        with self._lock:
            return list(self._raw_detections)

    # ── internal capture + inference loop ────────────────────
    def _capture_loop(self) -> None:
        source = self.cfg["camera"]["source"]
        # Convert numeric string to int if needed
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logging.error("Cannot open video source: %s", source)
            self._running = False
            return

        logging.info("Camera opened: %s", source)

        while self._running:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Frame grab failed – retrying in 1 s...")
                time.sleep(1)
                cap.release()
                cap = cv2.VideoCapture(source)
                continue

            annotated = self._process_frame(frame)

            # Encode to JPEG
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with self._lock:
                self._latest_frame = buf.tobytes()
                self._stats["frames_processed"] += 1

        cap.release()
        logging.info("Camera released.")

    # ── frame processing ──────────────────────────────────────
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE contrast enhancement to improve detection of thin objects."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self._clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        # Preprocess for better knife/thin-object visibility
        enhanced = self._preprocess(frame)
        results = self.model(
            enhanced,
            conf=self.conf_threshold,
            imgsz=self.inference_imgsz,
            verbose=False,
        )
        annotated = frame.copy()
        now = time.time()
        threats_in_frame: list[dict] = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                conf = float(box.conf[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if cls_name in self.threat_classes:
                    # Red box + label
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    label = f"! {cls_name.upper()} {conf:.0%}"
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(annotated, (x1, y1 - lh - 10), (x1 + lw + 4, y1), (0, 0, 255), -1)
                    cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    threats_in_frame.append({"class": cls_name, "confidence": conf,
                                             "x1": x1, "y1": y1, "x2": x2, "y2": y2})
                else:
                    # Green box for non-threats
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
                    label = f"{cls_name} {conf:.0%}"
                    cv2.putText(annotated, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)

                # Track all raw detections for debug panel
                raw_record = {
                    "class": cls_name,
                    "confidence": round(conf, 3),
                    "is_threat": cls_name in self.threat_classes,
                    "time": datetime.now().strftime("%H:%M:%S"),
                }
                with self._lock:
                    self._raw_detections.insert(0, raw_record)
                    self._raw_detections = self._raw_detections[:50]

        # Overlay HUD
        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(annotated, ts, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
        if threats_in_frame:
            cv2.putText(annotated, "! THREAT DETECTED !", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
            with self._lock:
                self._stats["threats_detected"] += 1

        # Handle alerts (cooldown per class)
        for threat in threats_in_frame:
            cls = threat["class"]
            last = self._last_alert_time.get(cls, 0)
            if now - last >= self.alert_cooldown:
                self._last_alert_time[cls] = now
                self._save_alert(annotated, threat)

        return annotated

    # ── alert persistence ─────────────────────────────────────
    def _save_alert(self, frame: np.ndarray, threat: dict) -> None:
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"alert_{threat['class']}_{ts_str}.jpg"
        filepath = self.alerts_dir / filename
        cv2.imwrite(str(filepath), frame)

        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "class": threat["class"],
            "confidence": round(threat["confidence"], 3),
            "screenshot": filename,
        }

        with self._lock:
            self._alert_log.insert(0, record)          # newest first
            if len(self._alert_log) > 200:             # cap in-memory list
                self._alert_log = self._alert_log[:200]
            self._stats["alerts_saved"] += 1

        # Append to CSV
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(record.keys()))
            writer.writerow(record)

        logging.info("Alert saved: %s (%.0f%% conf)", filename, threat["confidence"] * 100)

        # Desktop notification (non-blocking)
        threading.Thread(
            target=_notify,
            args=(
                f"⚠ Threat Detected: {threat['class'].upper()}",
                f"Confidence: {threat['confidence']:.0%}\nSaved: {filename}",
            ),
            daemon=True,
        ).start()


# ── MJPEG generator (used by Flask) ──────────────────────────
def mjpeg_generator(detector: ThreatDetector) -> Generator[bytes, None, None]:
    """Yield multipart JPEG frames for an HTTP MJPEG stream."""
    placeholder = _make_placeholder()
    while True:
        frame = detector.get_jpeg_frame()
        if frame is None:
            frame = placeholder
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )
        time.sleep(0.033)   # ~30 fps cap


def _make_placeholder() -> bytes:
    """Return a small 'Connecting…' JPEG when the camera isn't ready."""
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)
    cv2.putText(img, "Connecting to camera...", (100, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 200, 255), 2)
    cv2.putText(img, "Please wait", (230, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()
