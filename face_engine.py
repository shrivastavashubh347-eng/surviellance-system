"""
face_engine.py
--------------
Lightweight face detection engine using MediaPipe (primary)
with automatic fallback to OpenCV Haar cascades if MediaPipe
is not installed or fails to initialize.

Features:
  - MediaPipe BlazeFace: ~10ms/frame on CPU, very accurate
  - OpenCV Haar cascade fallback: ~20ms/frame, no extra install
  - Privacy blur (pixelate / gaussian blur of face regions)
  - Combined threat+face alerting
  - Runtime toggle (enable/disable without restart)
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FaceEngine:
    """
    Detects faces in frames. Tries MediaPipe first, falls back to OpenCV.
    Designed to be called once per frame (or every N frames).
    """

    # ── colour palette ────────────────────────────────────────
    BOX_COLOUR  = (255, 180, 0)   # bright blue/cyan
    TEXT_COLOUR = (255, 255, 255)
    LABEL_BG    = (180, 100, 0)

    def __init__(self, cfg: dict) -> None:
        self.cfg         = cfg.get("face_detection", {})
        self.enabled     = bool(self.cfg.get("enabled", True))
        self.min_conf    = float(self.cfg.get("min_confidence", 0.6))
        self.draw_boxes  = bool(self.cfg.get("draw_boxes", True))
        self.privacy_blur = bool(self.cfg.get("privacy_blur", False))
        self.alert_on_face_with_threat = bool(self.cfg.get("alert_on_face_with_threat", True))

        self._backend: str = "none"
        self._mp_detector = None

        if self.enabled:
            self._init_backend()

    # ── backend initialisation ────────────────────────────────
    def _init_backend(self) -> None:
        """Try MediaPipe first, fall back to OpenCV Haar."""
        try:
            import mediapipe as mp
            self._mp_face = mp.solutions.face_detection
            self._mp_detector = self._mp_face.FaceDetection(
                model_selection=0,           # model 0 = short-range (<2 m), faster
                min_detection_confidence=self.min_conf,
            )
            self._backend = "mediapipe"
            logger.info("Face detection: MediaPipe BlazeFace initialised")
        except Exception as exc:
            logger.warning("MediaPipe not available (%s) – falling back to OpenCV Haar", exc)
            self._init_opencv_fallback()

    def _init_opencv_fallback(self) -> None:
        """Load OpenCV's built-in frontal-face Haar cascade."""
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._haar = cv2.CascadeClassifier(cascade_path)
            if self._haar.empty():
                raise RuntimeError("Haar cascade file not found")
            self._backend = "opencv"
            logger.info("Face detection: OpenCV Haar cascade initialised")
        except Exception as exc:
            logger.error("Face detection unavailable: %s", exc)
            self._backend = "none"

    # ── public API ────────────────────────────────────────────
    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        if enabled and self._backend == "none":
            self._init_backend()

    def set_privacy_blur(self, blur: bool) -> None:
        self.privacy_blur = blur

    def get_backend(self) -> str:
        return self._backend

    def detect_and_draw(
        self,
        frame: np.ndarray,
        threats_present: bool = False,
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Run face detection on `frame`. Returns (annotated_frame, face_list).
        face_list items: {"x1","y1","x2","y2","confidence"}
        """
        if not self.enabled or self._backend == "none":
            return frame, []

        h, w = frame.shape[:2]
        faces: list[dict] = []

        try:
            if self._backend == "mediapipe":
                faces = self._detect_mediapipe(frame, w, h)
            elif self._backend == "opencv":
                faces = self._detect_opencv(frame)
        except Exception as exc:
            logger.warning("Face detection error: %s", exc)
            return frame, []

        annotated = frame.copy()

        for face in faces:
            x1, y1, x2, y2 = face["x1"], face["y1"], face["x2"], face["y2"]
            conf = face["confidence"]

            # ── Privacy blur ──────────────────────────────────
            if self.privacy_blur:
                roi = annotated[y1:y2, x1:x2]
                if roi.size > 0:
                    # Pixelate: shrink then enlarge
                    small = cv2.resize(roi, (max(1, (x2-x1)//8), max(1, (y2-y1)//8)))
                    blurred = cv2.resize(small, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
                    annotated[y1:y2, x1:x2] = blurred

            if not self.draw_boxes:
                continue

            # ── Threat + face combined glow ───────────────────
            box_col = (0, 0, 255) if threats_present else self.BOX_COLOUR
            thickness = 3 if threats_present else 2

            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_col, thickness)

            label = f"FACE {conf:.0%}"
            if threats_present:
                label = "! ARMED FACE"

            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
            bg_col = (0, 0, 200) if threats_present else self.LABEL_BG
            cv2.rectangle(annotated, (x1, y1 - lh - 10), (x1 + lw + 4, y1), bg_col, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, self.TEXT_COLOUR, 2)

        return annotated, faces

    # ── backends ──────────────────────────────────────────────
    def _detect_mediapipe(self, frame: np.ndarray, w: int, h: int) -> list[dict]:
        """Run MediaPipe BlazeFace detection."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._mp_detector.process(rgb)
        faces = []
        if not results.detections:
            return faces
        for det in results.detections:
            score = det.score[0] if det.score else 0.0
            if score < self.min_conf:
                continue
            bbox = det.location_data.relative_bounding_box
            x1 = max(0, int(bbox.xmin * w))
            y1 = max(0, int(bbox.ymin * h))
            x2 = min(w, int((bbox.xmin + bbox.width) * w))
            y2 = min(h, int((bbox.ymin + bbox.height) * h))
            if x2 > x1 and y2 > y1:
                faces.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": score})
        return faces

    def _detect_opencv(self, frame: np.ndarray) -> list[dict]:
        """Run OpenCV Haar cascade face detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        detections = self._haar.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )
        faces = []
        if len(detections) == 0:
            return faces
        for (x, y, fw, fh) in detections:
            faces.append({
                "x1": int(x), "y1": int(y),
                "x2": int(x + fw), "y2": int(y + fh),
                "confidence": 0.85,   # Haar doesn't give confidence; use a fixed value
            })
        return faces
