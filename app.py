"""
app.py
------
Flask web application for the live threat detection dashboard.

Routes:
  GET  /                   → Main dashboard HTML
  GET  /video_feed         → MJPEG live stream
  GET  /api/status         → JSON system status + current source info
  GET  /api/alerts         → JSON alert history (paginated)
  GET  /api/detections     → JSON raw live detections (all classes)
  GET  /api/cameras        → JSON list of available camera indices (0-5)
  GET  /api/videos         → JSON list of video files in /videos folder
  POST /api/source         → Switch video source at runtime
  POST /api/control        → Start / stop the detector
  GET  /alerts/<file>      → Serve saved alert screenshots
  POST /api/upload         → Upload a video file from phone (multipart)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml
from flask import (Flask, Response, jsonify, render_template,
                   request, send_from_directory)
from werkzeug.utils import secure_filename

from detector import ThreatDetector, mjpeg_generator, scan_cameras

# ── logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# ── config ────────────────────────────────────────────────────
CONFIG_PATH = os.environ.get("CONFIG_PATH", "config.yaml")
with open(CONFIG_PATH) as f:
    _cfg = yaml.safe_load(f)

VIDEOS_DIR = Path(_cfg.get("videos", {}).get("input_dir", "videos"))
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".flv", ".webm"}

# ── Flask app ─────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024 * 1024  # 4 GB upload limit
detector = ThreatDetector(CONFIG_PATH)

# ── routes ────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", config=_cfg)


@app.route("/video_feed")
def video_feed():
    return Response(
        mjpeg_generator(detector),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/status")
def api_status():
    stats = detector.get_stats()
    stats["is_running"] = detector.is_running()
    stats["current_source"] = detector.get_current_source()
    stats["source_label"] = detector.get_source_label()
    stats["connection_error"] = detector.get_connection_error()
    stats["face_enabled"] = detector.is_face_detection_enabled()
    stats["face_backend"] = detector.get_face_backend()
    stats["privacy_blur"] = detector.is_privacy_blur_enabled()
    return jsonify(stats)


@app.route("/api/alerts")
def api_alerts():
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 20))
    alerts = detector.get_alerts()
    start = (page - 1) * per_page
    return jsonify({
        "total": len(alerts),
        "page": page,
        "per_page": per_page,
        "alerts": alerts[start: start + per_page],
    })


@app.route("/api/detections")
def api_detections():
    return jsonify(detector.get_raw_detections())


@app.route("/api/face", methods=["GET", "POST"])
def api_face():
    """
    GET  /api/face         -> current face detection state
    POST /api/face         -> toggle settings
      Body: { "enabled": true/false }       # toggle face detection
            { "privacy_blur": true/false }  # toggle privacy blur
    """
    if request.method == "POST":
        data = request.json or {}
        if "enabled" in data:
            detector.set_face_detection(bool(data["enabled"]))
        if "privacy_blur" in data:
            detector.set_privacy_blur(bool(data["privacy_blur"]))

    return jsonify({
        "enabled": detector.is_face_detection_enabled(),
        "privacy_blur": detector.is_privacy_blur_enabled(),
        "backend": detector.get_face_backend(),
    })

@app.route("/api/cameras")
def api_cameras():
    """Probe and return available camera indices (0-5)."""
    cameras = scan_cameras(max_index=6)
    return jsonify(cameras)


@app.route("/api/videos")
def api_videos():
    """List video files available in the /videos folder."""
    return jsonify(detector.list_video_files())


@app.route("/api/source", methods=["POST"])
def api_source():
    """
    Hot-swap the video source.
    Body: { "source": "0" }           ← camera index (string or int)
           { "source": "videos/x.mp4" } ← video file
           { "source": "rtsp://..." }   ← IP camera
    """
    data = request.json or {}
    source = data.get("source", "")
    if source == "":
        return jsonify({"error": "source field is required"}), 400

    result = detector.switch_source(source)

    # Start the detector if it wasn't running
    if not detector.is_running():
        detector.start()

    return jsonify(result)


@app.route("/api/control", methods=["POST"])
def api_control():
    action = (request.json or {}).get("action", "")
    if action == "start":
        if not detector.is_running():
            detector.start()
        return jsonify({"status": "started"})
    elif action == "stop":
        if detector.is_running():
            detector.stop()
        return jsonify({"status": "stopped"})
    return jsonify({"error": "Unknown action"}), 400


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """
    Upload a video file transferred from the phone.
    Saves to /videos folder, then optionally auto-starts analysis.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(f.filename)
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    save_path = VIDEOS_DIR / filename
    f.save(str(save_path))
    size_mb = round(save_path.stat().st_size / 1_048_576, 1)
    logging.info("Uploaded video: %s (%.1f MB)", filename, size_mb)

    # Auto-switch to this file if requested
    auto_analyze = request.form.get("auto_analyze", "false").lower() == "true"
    if auto_analyze:
        detector.switch_source(str(save_path))
        if not detector.is_running():
            detector.start()

    return jsonify({
        "filename": filename,
        "path": str(save_path),
        "size_mb": size_mb,
        "auto_analyze": auto_analyze,
    })


@app.route("/alerts/<path:filename>")
def serve_alert(filename: str):
    alerts_dir = Path(_cfg["alerts"]["output_dir"]).resolve()
    return send_from_directory(alerts_dir, filename)


@app.route("/videos/<path:filename>")
def serve_video(filename: str):
    return send_from_directory(VIDEOS_DIR.resolve(), filename)


# ── main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    detector.start()
    server_cfg = _cfg.get("server", {})
    app.run(
        host=server_cfg.get("host", "127.0.0.1"),
        port=server_cfg.get("port", 5000),
        debug=server_cfg.get("debug", False),
        threaded=True,
        use_reloader=False,
    )
