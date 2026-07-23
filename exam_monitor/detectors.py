# =====================================================
#  detectors.py — Three detection functions:
#    1. detect_face   — is a face present in the frame?
#    2. detect_gaze   — is the student looking away?
#    3. detect_phone  — is a phone visible? (YOLO)
#
#  Each function is stateless and independent so they
#  are easy to unit-test and swap out individually.
#
#  MediaPipe 0.10.x Tasks API is used throughout.
#  The legacy `mp.solutions` namespace was removed.
# =====================================================

import logging

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

from config import CFG
from model_loader import models

logger = logging.getLogger("exam_monitor")


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITY: image decode + resize
# ─────────────────────────────────────────────────────────────────────────────

def decode_image(raw_bytes: bytes) -> np.ndarray:
    """
    Decode raw JPEG/PNG bytes into a NumPy BGR array.
    Raises ValueError if the image cannot be decoded.
    """
    arr   = np.frombuffer(raw_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Image could not be decoded. Send a valid JPEG or PNG.")
    return frame


def resize_frame(frame: np.ndarray, target_width: int) -> np.ndarray:
    """
    Resize a frame to a fixed width while maintaining aspect ratio.
    Returns the frame unchanged if it is already narrower than target_width.
    """
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / w
    return cv2.resize(frame, (target_width, int(h * scale)))


# ─────────────────────────────────────────────────────────────────────────────
#  1. FACE DETECTION  (MediaPipe Tasks — FaceDetector)
# ─────────────────────────────────────────────────────────────────────────────

def detect_face(rgb_frame: np.ndarray) -> bool:
    """
    Detect whether at least one face is visible in the frame.

    Uses the MediaPipe Tasks FaceDetector (blaze_face_short_range model).
    Returns True if one or more detections are found, False otherwise.
    """
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result   = models["face_detector"].detect(mp_image)
    return len(result.detections) > 0


# ─────────────────────────────────────────────────────────────────────────────
#  2. GAZE / LOOK-AWAY DETECTION  (MediaPipe Tasks — FaceLandmarker)
# ─────────────────────────────────────────────────────────────────────────────

def detect_gaze(rgb_frame: np.ndarray) -> bool:
    """
    Estimate whether the student is looking away from the screen.

    Strategy:
      - Extract nose tip (landmark 1) and eye corners (landmarks 33, 263)
        from the first detected face.
      - Compute the horizontal distance between the nose and the midpoint
        of the two eye corners.
      - If this deviation exceeds GAZE_DEVIATION_LIMIT the student is
        classified as looking away.

    Returns True  = looking away
            False = looking at screen (or landmarks not found)
    """
    mp_image    = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result      = models["face_landmarker"].detect(mp_image)

    if not result.face_landmarks:
        # No landmarks detected — cannot determine gaze
        return False

    landmarks = result.face_landmarks[0]   # first face only

    # Landmark indices (MediaPipe 478-point model):
    #   1  = nose tip
    #   33 = left eye outer corner
    #  263 = right eye outer corner
    nose      = landmarks[1]
    left_eye  = landmarks[33]
    right_eye = landmarks[263]

    eye_center_x = (left_eye.x + right_eye.x) / 2
    deviation    = abs(nose.x - eye_center_x)

    looking_away = deviation > CFG["GAZE_DEVIATION_LIMIT"]

    if looking_away:
        logger.debug("Gaze deviation: %.3f (limit: %.3f)", deviation, CFG["GAZE_DEVIATION_LIMIT"])

    return looking_away


# ─────────────────────────────────────────────────────────────────────────────
#  3. PHONE DETECTION  (YOLOv8)
# ─────────────────────────────────────────────────────────────────────────────

def detect_phone(frame_bgr: np.ndarray) -> tuple[bool, list[dict]]:
    """
    Detect mobile phones in the frame using YOLOv8 (COCO class 67).

    Returns:
        phone_found   — True if at least one phone was detected
        boxes         — list of dicts, each containing:
                          label, x1, y1, x2, y2, confidence
    """
    boxes: list[dict] = []

    yolo_results = models["yolo"](
        frame_bgr,
        conf    = CFG["YOLO_CONF"],     # minimum confidence threshold
        imgsz   = CFG["YOLO_IMGSZ"],    # inference resolution
        verbose = False,                 # suppress per-frame console output
    )

    phone_found = False

    for r in yolo_results:
        for box in r.boxes:
            if int(box.cls[0]) == CFG["PHONE_CLASS_ID"]:
                phone_found = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append({
                    "label":      "phone",
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "confidence": round(float(box.conf[0]), 3),
                })
                logger.debug("Phone detected at (%d,%d)-(%d,%d)", x1, y1, x2, y2)

    return phone_found, boxes