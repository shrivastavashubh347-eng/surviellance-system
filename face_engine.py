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

import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FaceEngine:
    """
    Detects faces and recognizes target individuals.
    Primary backend: OpenCV YuNet (detection) + SFace (recognition).
    Fallback: MediaPipe (detection only) or Haar cascades.
    """

    # ── colour palette ────────────────────────────────────────
    BOX_COLOUR  = (255, 180, 0)   # bright blue/cyan
    TEXT_COLOUR = (255, 255, 255)
    LABEL_BG    = (180, 100, 0)
    TARGET_COLOUR = (0, 255, 0)   # green for recognized targets
    TARGET_BG   = (0, 200, 0)

    def __init__(self, cfg: dict) -> None:
        self.cfg         = cfg.get("face_detection", {})
        self.target_cfg  = cfg.get("target_recognition", {})
        self.enabled     = bool(self.cfg.get("enabled", True))
        self.min_conf    = float(self.cfg.get("min_confidence", 0.6))
        self.draw_boxes  = bool(self.cfg.get("draw_boxes", True))
        self.privacy_blur = bool(self.cfg.get("privacy_blur", False))
        
        # Target face settings
        self.similarity_thresh = float(self.target_cfg.get("similarity_threshold", 0.364))
        self.targets: dict[str, np.ndarray] = {}  # name -> 128D embedding
        
        self._backend: str = "none"
        self._mp_detector = None
        self._yunet = None
        self._sface = None

        if self.enabled:
            self._init_backend()
            self.load_target_faces()

    # ── backend initialisation ────────────────────────────────
    def _init_backend(self) -> None:
        """Try YuNet+SFace first, then MediaPipe, then OpenCV Haar."""
        yunet_path = "models/face_detection_yunet_2023mar.onnx"
        sface_path = "models/face_recognition_sface_2021dec.onnx"

        if os.path.exists(yunet_path) and os.path.exists(sface_path):
            try:
                # We initialize with a dummy size; it will be set dynamically per frame
                self._yunet = cv2.FaceDetectorYN.create(yunet_path, "", (320, 320), self.min_conf, 0.3, 5000)
                self._sface = cv2.FaceRecognizerSF.create(sface_path, "")
                self._backend = "yunet_sface"
                logger.info("Face detection: YuNet + SFace initialised for recognition")
                return
            except Exception as exc:
                logger.warning("YuNet/SFace failed: %s", exc)

        try:
            import mediapipe as mp
            self._mp_face = mp.solutions.face_detection
            self._mp_detector = self._mp_face.FaceDetection(
                model_selection=0,
                min_detection_confidence=self.min_conf,
            )
            self._backend = "mediapipe"
            logger.info("Face detection: MediaPipe BlazeFace initialised (No recognition)")
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

    # ── target face management ────────────────────────────────
    def load_target_faces(self, directory: str = "target_faces") -> None:
        """Load images from directory, extract faces, and save embeddings."""
        self.targets.clear()
        if self._backend != "yunet_sface" or not self._sface:
            logger.info("Target recognition disabled: backend is %s", self._backend)
            return

        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            return

        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if not os.path.isfile(filepath):
                continue
                
            img = cv2.imread(filepath)
            if img is None:
                continue

            # Detect face
            h, w = img.shape[:2]
            self._yunet.setInputSize((w, h))
            _, faces = self._yunet.detect(img)
            
            if faces is None or len(faces) == 0:
                logger.warning("No face found in target image: %s", filename)
                continue
                
            # Use the most prominent face (first one)
            face = faces[0][:-1]
            
            # Align and extract feature
            aligned = self._sface.alignCrop(img, face)
            feature = self._sface.feature(aligned)
            
            # Use filename without extension as the target name
            name = Path(filename).stem
            self.targets[name] = feature
            logger.info("Loaded target face: %s", name)

    def get_target_names(self) -> list[str]:
        return list(self.targets.keys())

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
        Run face detection/recognition. Returns (annotated_frame, face_list).
        face_list items: {"x1", "y1", "x2", "y2", "confidence", "target_name": Optional[str]}
        """
        if not self.enabled or self._backend == "none":
            return frame, []

        h, w = frame.shape[:2]
        faces: list[dict] = []

        try:
            if self._backend == "yunet_sface":
                faces = self._detect_yunet_sface(frame, w, h)
            elif self._backend == "mediapipe":
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
            target_name = face.get("target_name")

            # ── Privacy blur ──────────────────────────────────
            if self.privacy_blur:
                roi = annotated[y1:y2, x1:x2]
                if roi.size > 0:
                    small = cv2.resize(roi, (max(1, (x2-x1)//8), max(1, (y2-y1)//8)))
                    blurred = cv2.resize(small, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
                    annotated[y1:y2, x1:x2] = blurred

            if not self.draw_boxes:
                continue

            # ── Styling based on Target / Threat ──────────────
            is_target = target_name is not None
            
            if is_target:
                box_col = self.TARGET_COLOUR
                bg_col = self.TARGET_BG
                thickness = 3
                label = f"TARGET: {target_name.upper()}"
            elif threats_present:
                box_col = (0, 0, 255)
                bg_col = (0, 0, 200)
                thickness = 3
                label = "! ARMED FACE"
            else:
                box_col = self.BOX_COLOUR
                bg_col = self.LABEL_BG
                thickness = 2
                label = f"FACE {conf:.0%}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_col, thickness)

            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
            cv2.rectangle(annotated, (x1, y1 - lh - 10), (x1 + lw + 4, y1), bg_col, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, self.TEXT_COLOUR, 2)

        return annotated, faces

    # ── backends ──────────────────────────────────────────────
    def _detect_yunet_sface(self, frame: np.ndarray, w: int, h: int) -> list[dict]:
        """Run YuNet detection + SFace recognition."""
        self._yunet.setInputSize((w, h))
        _, detections = self._yunet.detect(frame)
        
        faces = []
        if detections is None:
            return faces
            
        for det in detections:
            conf = det[-1]
            if conf < self.min_conf:
                continue
                
            x1, y1, fw, fh = map(int, det[0:4])
            x2, y2 = x1 + fw, y1 + fh
            
            # Ensure bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            face_info = {
                "x1": x1, "y1": y1, "x2": x2, "y2": y2, 
                "confidence": conf,
                "target_name": None
            }
            
            # Recognition
            if self.targets:
                aligned = self._sface.alignCrop(frame, det[:-1])
                feature = self._sface.feature(aligned)
                
                best_match = None
                highest_sim = 0.0
                
                for name, target_feature in self.targets.items():
                    sim = self._sface.match(feature, target_feature, cv2.FaceRecognizerSF_FR_COSINE)
                    if sim > highest_sim and sim >= self.similarity_thresh:
                        highest_sim = sim
                        best_match = name
                        
                if best_match:
                    face_info["target_name"] = best_match
                    
            faces.append(face_info)
            
        return faces

    def _detect_mediapipe(self, frame: np.ndarray, w: int, h: int) -> list[dict]:
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
                faces.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": score, "target_name": None})
        return faces

    def _detect_opencv(self, frame: np.ndarray) -> list[dict]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        detections = self._haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        faces = []
        if len(detections) == 0:
            return faces
        for (x, y, fw, fh) in detections:
            faces.append({
                "x1": int(x), "y1": int(y), "x2": int(x + fw), "y2": int(y + fh),
                "confidence": 0.85, "target_name": None
            })
        return faces
