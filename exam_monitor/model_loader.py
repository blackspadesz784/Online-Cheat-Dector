# =====================================================
#  model_loader.py — ML model lifecycle management
#
#  All models are loaded exactly once when the server
#  starts (FastAPI lifespan startup) and released when
#  it shuts down.
#
#  MediaPipe 0.10.x removed the legacy `mp.solutions`
#  API. We now use the MediaPipe Tasks API:
#    - FaceDetector  (replaces FaceDetection)
#    - FaceLandmarker (replaces FaceMesh)
#
#  Both task model files are downloaded automatically
#  on first run and cached locally.
# =====================================================

import logging
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from fastapi import FastAPI
from ultralytics import YOLO

import database

logger = logging.getLogger("exam_monitor")

# Global model shelf — shared across all requests (read-only from other modules)
models: dict = {}

# ── MediaPipe Task model URLs ────────────────────────────────────────────────
_FACE_DETECTOR_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
_FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

_FACE_DETECTOR_MODEL_PATH  = Path("models/blaze_face_short_range.tflite")
_FACE_LANDMARKER_MODEL_PATH = Path("models/face_landmarker.task")


def _download_model(url: str, dest: Path) -> None:
    """Download a MediaPipe model file if it does not already exist locally."""
    if dest.exists():
        logger.info("Model already cached: %s", dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading model: %s → %s", url, dest)
    urllib.request.urlretrieve(url, dest)
    logger.info("Downloaded: %s (%.1f KB)", dest.name, dest.stat().st_size / 1024)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan hook.

    Everything before `yield` runs at server startup.
    Everything after `yield` runs at server shutdown.
    """
    logger.info("Server starting — loading ML models and connecting to DB ...")

    # ── YOLOv8 (phone detection) ──────────────────────────────────────────────
    models["yolo"] = YOLO("yolov8n.pt")
    logger.info("YOLOv8 loaded")

    # ── Download MediaPipe task model files if needed ─────────────────────────
    _download_model(_FACE_DETECTOR_MODEL_URL,  _FACE_DETECTOR_MODEL_PATH)
    _download_model(_FACE_LANDMARKER_MODEL_URL, _FACE_LANDMARKER_MODEL_PATH)

    # ── MediaPipe Face Detector (Tasks API) ───────────────────────────────────
    face_detector_opts = mp_vision.FaceDetectorOptions(
        base_options      = mp_python.BaseOptions(
            model_asset_path=str(_FACE_DETECTOR_MODEL_PATH)
        ),
        running_mode      = mp_vision.RunningMode.IMAGE,
        min_detection_confidence = 0.5,
    )
    models["face_detector"] = mp_vision.FaceDetector.create_from_options(face_detector_opts)
    logger.info("MediaPipe FaceDetector (Tasks API) loaded")

    # ── MediaPipe Face Landmarker (Tasks API) — replaces FaceMesh ────────────
    face_landmarker_opts = mp_vision.FaceLandmarkerOptions(
        base_options   = mp_python.BaseOptions(
            model_asset_path=str(_FACE_LANDMARKER_MODEL_PATH)
        ),
        running_mode   = mp_vision.RunningMode.IMAGE,
        num_faces      = 1,
        min_face_detection_confidence = 0.5,
        min_tracking_confidence       = 0.5,
    )
    models["face_landmarker"] = mp_vision.FaceLandmarker.create_from_options(face_landmarker_opts)
    logger.info("MediaPipe FaceLandmarker (Tasks API) loaded")

    # ── MongoDB (async) ───────────────────────────────────────────────────────
    await database.init_db()

    logger.info("All models and DB ready — accepting requests")

    yield   # ← server handles requests between here and shutdown

    # ── Cleanup ───────────────────────────────────────────────────────────────
    models["face_detector"].close()
    models["face_landmarker"].close()
    await database.close_db()
    logger.info("Shutdown complete — models released, DB connection closed")