"""
detector.py
-----------
Core detection engine.
  - Supports USB webcam, phone via USB (DroidCam / Android 14 Webcam mode),
    RTSP/HTTP IP camera streams, and local video files
  - Runtime source switching without server restart
  - Auto camera index scanning
  - Runs YOLOv8 inference on every frame
  - Flags threat classes (knife / scissors)
  - Saves timestamped alert screenshots
  - Fires a desktop notification (Windows toast / Linux libnotify)
  - Exposes a thread-safe JPEG frame generator for the Flask stream
"""

from __future__ import annotations

import csv
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

from face_engine import FaceEngine

# ──────────────────────────────────────────────────────────────
# Desktop notification helper (cross-platform best-effort)
# ──────────────────────────────────────────────────────────────
def _notify(title: str, message: str) -> None:
    """Fire a desktop notification without crashing the main thread."""
    try:
        import platform
        system = platform.system()
        if system == "Windows":
            try:
                from plyer import notification
                notification.notify(
                    title=title, message=message,
                    app_name="Threat Detector", timeout=5,
                )
            except Exception:
                try:
                    from win10toast import ToastNotifier
                    ToastNotifier().show_toast(title, message, duration=5, threaded=True)
                except Exception:
                    import ctypes
                    ctypes.windll.user32.MessageBeep(0xFFFFFFFF)
        elif system == "Darwin":
            os.system(f'osascript -e \'display notification "{message}" with title "{title}"\'')
        else:
            os.system(f'notify-send "{title}" "{message}"')
    except Exception as exc:
        logging.warning("Desktop notification failed: %s", exc)


# ──────────────────────────────────────────────────────────────
# Camera scanner utility
# ──────────────────────────────────────────────────────────────
def scan_cameras(max_index: int = 6) -> list[dict]:
    """
    Probe camera indices 0..max_index and return a list of available devices.
    Each entry: {"index": int, "label": str, "available": bool}
    """
    results = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)   # CAP_DSHOW is faster on Windows
        if cap.isOpened():
            ret, _ = cap.read()
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            label = f"Camera {idx}  ({w}×{h})"
            results.append({"index": idx, "label": label, "available": ret})
            cap.release()
        else:
            results.append({"index": idx, "label": f"Camera {idx}  (not found)", "available": False})
    return results


# ──────────────────────────────────────────────────────────────
# Detector
# ──────────────────────────────────────────────────────────────
class ThreatDetector:
    """Runs in a background thread; provides JPEG frames + alert log."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        self.config_path = config_path
        self.cfg = self._load_config(config_path)
        self._lock = threading.Lock()
        self._source_lock = threading.Lock()
        self._latest_frame: bytes | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._alert_log: list[dict] = []
        self._last_alert_time: dict[str, float] = {}
        self._stats = {
            "frames_processed": 0,
            "threats_detected": 0,
            "alerts_saved": 0,
            "uptime_start": None,
        }
        self._current_source: str | int = self.cfg["camera"]["source"]
        self._pending_source: str | int | None = None   # set to trigger a switch
        self._source_label: str = str(self._current_source)
        self._connection_error: str = ""

        # Ensure directories exist
        self.alerts_dir = Path(self.cfg["alerts"]["output_dir"])
        self.alerts_dir.mkdir(parents=True, exist_ok=True)
        self.videos_dir = Path(self.cfg.get("videos", {}).get("input_dir", "videos"))
        self.videos_dir.mkdir(parents=True, exist_ok=True)

        # CSV log
        self._csv_path = self.alerts_dir / "alert_log.csv"
        if not self._csv_path.exists():
            with open(self._csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=["timestamp", "class", "confidence", "screenshot"]).writeheader()

        logging.info("Loading YOLO model: %s", self.cfg["detection"]["model"])
        self.model = YOLO(self.cfg["detection"]["model"])
        self.threat_classes: set[str] = set(self.cfg["detection"]["threat_classes"])
        self.conf_threshold: float = float(self.cfg["detection"]["confidence_threshold"])
        self.alert_cooldown: float = float(self.cfg["detection"]["alert_cooldown"])
        self.inference_imgsz: int = int(self.cfg["detection"].get("inference_imgsz", 640))
        self._loop_video: bool = bool(self.cfg["camera"].get("loop_video", True))

        # CLAHE for contrast enhancement
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Raw detection log for debug panel
        self._raw_detections: list[dict] = []

        # Face detection engine
        self._face_engine = FaceEngine(self.cfg)
        self._face_frame_skip = int(self.cfg.get("face_detection", {}).get("run_every_n_frames", 2))
        self._face_frame_counter = 0
        self._last_faces: list[dict] = []   # cached result from last face-detection frame
        self._stats["faces_detected"] = 0
        self._last_combined_alert: float = 0.0
        self.target_alert_cooldown = float(self.cfg.get("target_recognition", {}).get("alert_cooldown", 5))

        logging.info(
            "Threat detector ready. Watching for: %s (conf >= %.2f)",
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

    def switch_source(self, source: str | int) -> dict:
        """Hot-swap the video source without restarting the server."""
        # Normalise
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        # For video files, resolve relative to videos/ dir if not absolute
        if isinstance(source, str) and not source.startswith(("rtsp://", "http://", "https://")):
            p = Path(source)
            if not p.is_absolute() and not p.exists():
                candidate = self.videos_dir / p.name
                if candidate.exists():
                    source = str(candidate)

        with self._source_lock:
            self._pending_source = source
        logging.info("Source switch requested: %s", source)

        # If not running, also update config so next start uses it
        if not self._running:
            self._current_source = source
            self._source_label = str(source)

        return {"source": str(source)}

    def get_current_source(self) -> str:
        return str(self._current_source)

    def get_source_label(self) -> str:
        return self._source_label

    def get_connection_error(self) -> str:
        return self._connection_error

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
        with self._lock:
            return list(self._raw_detections)

    def set_face_detection(self, enabled: bool) -> None:
        """Toggle face detection at runtime."""
        self._face_engine.set_enabled(enabled)
        logging.info("Face detection %s", "enabled" if enabled else "disabled")

    def set_privacy_blur(self, blur: bool) -> None:
        """Toggle privacy blur at runtime."""
        self._face_engine.set_privacy_blur(blur)
        logging.info("Privacy blur %s", "enabled" if blur else "disabled")

    def get_face_backend(self) -> str:
        return self._face_engine.get_backend()

    def is_face_detection_enabled(self) -> bool:
        return self._face_engine.enabled

    def is_privacy_blur_enabled(self) -> bool:
        return self._face_engine.privacy_blur

    def get_target_names(self) -> list[str]:
        return self._face_engine.get_target_names()

    def reload_target_faces(self) -> None:
        self._face_engine.load_target_faces()

    def list_video_files(self) -> list[dict]:
        """Return video files available in the /videos folder."""
        exts = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".flv", ".webm"}
        files = []
        for p in sorted(self.videos_dir.iterdir()):
            if p.suffix.lower() in exts:
                files.append({
                    "name": p.name,
                    "path": str(p),
                    "size_mb": round(p.stat().st_size / 1_048_576, 1),
                })
        return files

    # ── internal capture + inference loop ─────────────────────
    def _open_source(self, source: str | int) -> cv2.VideoCapture:
        """Open a VideoCapture for any source type."""
        if isinstance(source, int):
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            # Apply resolution hints
            w = self.cfg["camera"].get("width", 0)
            h = self.cfg["camera"].get("height", 0)
            if w and h:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        else:
            # RTSP / HTTP stream or local file — let OpenCV auto-detect backend
            cap = cv2.VideoCapture(source)
        return cap

    def _is_video_file(self, source: str | int) -> bool:
        if isinstance(source, int):
            return False
        exts = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".flv", ".webm"}
        return Path(str(source)).suffix.lower() in exts

    def _capture_loop(self) -> None:
        source = self._current_source
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        cap = self._open_source(source)

        if not cap.isOpened():
            msg = f"Cannot open source: {source}"
            logging.error(msg)
            with self._source_lock:
                self._connection_error = msg
            self._running = False
            return

        self._current_source = source
        self._source_label = str(source)
        self._connection_error = ""
        logging.info("Source opened: %s", source)

        while self._running:
            # ── Hot-swap source if requested ──────────────────
            with self._source_lock:
                pending = self._pending_source
                if pending is not None:
                    self._pending_source = None

            if pending is not None:
                logging.info("Switching source to: %s", pending)
                cap.release()
                cap = self._open_source(pending)
                if cap.isOpened():
                    source = pending
                    self._current_source = source
                    self._source_label = str(source)
                    self._connection_error = ""
                    logging.info("Switched to: %s", source)
                else:
                    msg = f"Cannot open new source: {pending}"
                    logging.error(msg)
                    self._connection_error = msg
                    # Re-open old source
                    cap = self._open_source(source)
                    if not cap.isOpened():
                        self._running = False
                        break

            ret, frame = cap.read()

            if not ret:
                # End of video file → loop or stop
                if self._is_video_file(source):
                    if self._loop_video:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        logging.info("Video file ended.")
                        self._running = False
                        break
                else:
                    logging.warning("Frame grab failed – retrying in 1 s...")
                    time.sleep(1)
                    cap.release()
                    cap = self._open_source(source)
                    continue

            annotated = self._process_frame(frame)

            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with self._lock:
                self._latest_frame = buf.tobytes()
                self._stats["frames_processed"] += 1

        cap.release()
        logging.info("Source released.")

    # ── frame processing ──────────────────────────────────────
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE contrast enhancement to improve detection of thin objects."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self._clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
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
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    label = f"! {cls_name.upper()} {conf:.0%}"
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(annotated, (x1, y1 - lh - 10), (x1 + lw + 4, y1), (0, 0, 255), -1)
                    cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    threats_in_frame.append({"class": cls_name, "confidence": conf,
                                             "x1": x1, "y1": y1, "x2": x2, "y2": y2})
                else:
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
                    label = f"{cls_name} {conf:.0%}"
                    cv2.putText(annotated, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)

                raw_record = {
                    "class": cls_name, "confidence": round(conf, 3),
                    "is_threat": cls_name in self.threat_classes,
                    "time": datetime.now().strftime("%H:%M:%S"),
                }
                with self._lock:
                    self._raw_detections.insert(0, raw_record)
                    self._raw_detections = self._raw_detections[:50]

        # ── Face detection pass ───────────────────────────────
        self._face_frame_counter += 1
        run_face = (self._face_frame_counter % max(1, self._face_frame_skip) == 0)
        if run_face:
            annotated, faces = self._face_engine.detect_and_draw(
                annotated, threats_present=bool(threats_in_frame)
            )
            self._last_faces = faces
            if faces:
                with self._lock:
                    self._stats["faces_detected"] = self._stats.get("faces_detected", 0) + len(faces)
        else:
            # Re-draw last known face boxes without re-running detection
            if self._last_faces and self._face_engine.draw_boxes and self._face_engine.enabled:
                for face in self._last_faces:
                    col = (0, 0, 255) if threats_in_frame else self._face_engine.BOX_COLOUR
                    cv2.rectangle(annotated, (face["x1"], face["y1"]),
                                  (face["x2"], face["y2"]), col, 2)

        # ── Face count HUD ────────────────────────────────────
        n_faces = len(self._last_faces)
        if n_faces > 0 and self._face_engine.enabled:
            face_txt = f"Faces: {n_faces}" + (" [BLUR]" if self._face_engine.privacy_blur else "")
            cv2.putText(annotated, face_txt,
                        (annotated.shape[1] - 180, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 180, 0), 2)
                        
            # Check for target faces
            for face in self._last_faces:
                tname = face.get("target_name")
                if tname:
                    key = f"target_{tname}"
                    last = self._last_alert_time.get(key, 0)
                    if now - last >= self.target_alert_cooldown:
                        self._last_alert_time[key] = now
                        self._save_target_alert(annotated, tname, face["confidence"])

        # ── HUD overlay ───────────────────────────────────────
        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        src_label = f"SRC: {str(self._current_source)[:30]}"
        cv2.putText(annotated, ts, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
        cv2.putText(annotated, src_label, (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
        if threats_in_frame:
            cv2.putText(annotated, "! THREAT DETECTED !", (10, 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
            with self._lock:
                self._stats["threats_detected"] += 1

        for threat in threats_in_frame:
            cls = threat["class"]
            last = self._last_alert_time.get(cls, 0)
            if now - last >= self.alert_cooldown:
                self._last_alert_time[cls] = now
                self._save_alert(annotated, threat)

        # ── Combined face + threat alert ──────────────────────
        face_cfg = self.cfg.get("face_detection", {})
        if (threats_in_frame and self._last_faces
                and face_cfg.get("alert_on_face_with_threat", True)
                and face_cfg.get("save_combined_alert", True)
                and now - self._last_combined_alert >= self.alert_cooldown):
            self._last_combined_alert = now
            self._save_combined_alert(annotated, threats_in_frame, self._last_faces)

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
            self._alert_log.insert(0, record)
            if len(self._alert_log) > 200:
                self._alert_log = self._alert_log[:200]
            self._stats["alerts_saved"] += 1

        with open(self._csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=list(record.keys())).writerow(record)

        logging.info("Alert saved: %s (%.0f%% conf)", filename, threat["confidence"] * 100)

        threading.Thread(
            target=_notify,
            args=(
                f"Threat Detected: {threat['class'].upper()}",
                f"Confidence: {threat['confidence']:.0%}\nSaved: {filename}",
            ),
            daemon=True,
        ).start()

    def _save_target_alert(self, frame: np.ndarray, target_name: str, confidence: float) -> None:
        """Save a screenshot when a recognized target face is detected."""
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"target_{target_name}_{ts_str}.jpg"
        filepath = self.alerts_dir / filename
        cv2.imwrite(str(filepath), frame)

        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "class": f"TARGET ({target_name})",
            "confidence": round(confidence, 3),
            "screenshot": filename,
        }
        with self._lock:
            self._alert_log.insert(0, record)
            self._stats["alerts_saved"] += 1

        with open(self._csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=list(record.keys())).writerow(record)

        logging.info("Target recognized: %s", filename)

        threading.Thread(
            target=_notify,
            args=(
                f"Target Found: {target_name.upper()}",
                f"Saved: {filename}",
            ),
            daemon=True,
        ).start()

    def _save_combined_alert(self, frame: np.ndarray, threats: list[dict], faces: list[dict]) -> None:
        """Save a combined screenshot when a face and weapon appear together."""
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        cls_names = "+".join(sorted({t["class"] for t in threats}))
        filename = f"alert_COMBINED_{cls_names}_{len(faces)}face_{ts_str}.jpg"
        filepath = self.alerts_dir / filename
        cv2.imwrite(str(filepath), frame)

        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "class": f"COMBINED ({cls_names} + {len(faces)} face(s))",
            "confidence": round(max(t["confidence"] for t in threats), 3),
            "screenshot": filename,
        }
        with self._lock:
            self._alert_log.insert(0, record)
            self._stats["alerts_saved"] += 1

        with open(self._csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=list(record.keys())).writerow(record)

        logging.info("Combined alert saved: %s", filename)

        threading.Thread(
            target=_notify,
            args=(
                "ARMED PERSON DETECTED",
                f"Weapon + face in frame!\n{cls_names} detected with {len(faces)} face(s)",
            ),
            daemon=True,
        ).start()


# ── MJPEG generator (used by Flask) ──────────────────────────
def mjpeg_generator(detector: ThreatDetector) -> Generator[bytes, None, None]:
    placeholder = _make_placeholder()
    while True:
        frame = detector.get_jpeg_frame()
        if frame is None:
            frame = placeholder
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )
        time.sleep(0.033)


def _make_placeholder() -> bytes:
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)
    cv2.putText(img, "Connecting to source...", (100, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 200, 255), 2)
    cv2.putText(img, "Please wait", (230, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()
