"""
app.py
------
Flask web application for the live threat detection dashboard.

Routes:
  GET  /              → Main dashboard HTML
  GET  /video_feed    → MJPEG live stream
  GET  /api/status    → JSON system status
  GET  /api/alerts    → JSON alert history
  POST /api/control   → Start / stop the detector
  GET  /alerts/<file> → Serve saved alert screenshots
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml
from flask import Flask, Response, jsonify, render_template, request, send_from_directory

from detector import ThreatDetector, mjpeg_generator

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

# ── Flask app ─────────────────────────────────────────────────
app = Flask(__name__)
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
    """Raw detections across all classes — useful for tuning confidence threshold."""
    return jsonify(detector.get_raw_detections())


@app.route("/api/control", methods=["POST"])
def api_control():
    action = request.json.get("action", "")
    if action == "start":
        if not detector.is_running():
            detector.start()
        return jsonify({"status": "started"})
    elif action == "stop":
        if detector.is_running():
            detector.stop()
        return jsonify({"status": "stopped"})
    return jsonify({"error": "Unknown action"}), 400


@app.route("/alerts/<path:filename>")
def serve_alert(filename: str):
    alerts_dir = Path(_cfg["alerts"]["output_dir"]).resolve()
    return send_from_directory(alerts_dir, filename)


# ── main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    detector.start()
    server_cfg = _cfg.get("server", {})
    app.run(
        host=server_cfg.get("host", "127.0.0.1"),
        port=server_cfg.get("port", 5000),
        debug=server_cfg.get("debug", False),
        threaded=True,
        use_reloader=False,   # prevents double-start of the detector thread
    )
