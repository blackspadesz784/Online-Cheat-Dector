# =====================================================
#  routes.py — All API endpoints
#
#  Endpoints:
#    POST /analyze-frame/             — main detection endpoint
#    POST /reset-session/{id}         — clear a student session
#    GET  /session/{id}               — inspect one session (debug)
#    GET  /session/{id}/log           — full MongoDB frame log
#    GET  /sessions/                  — list all active sessions
#    GET  /health                     — server + model status
# =====================================================

import asyncio
import logging
import time
import uuid
from functools import partial
from typing import Optional

import cv2
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

import database
from config import CFG
from schemas import AnalysisResult, BoundingBox
from session_store import get_session, reset_session, get_all_sessions
from detectors import decode_image, resize_frame, detect_face, detect_gaze, detect_phone

logger = logging.getLogger("exam_monitor")

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
#  INTERNAL HELPER — runs CPU-bound inference in a thread-pool executor
#  so the async event loop is never blocked by MediaPipe / YOLO.
# ─────────────────────────────────────────────────────────────────────────────

def _run_inference(frame_bgr, frame_rgb, frame_idx: int) -> tuple[bool, bool, bool, list[dict]]:
    """
    Synchronous function that performs all ML inference for one frame.

    This is executed inside asyncio's default ThreadPoolExecutor via
    run_in_executor, so it does NOT block the event loop.

    Returns:
        face_found    — True if a face was detected
        looking_away  — True if gaze deviation exceeds threshold
        phone_found   — True if a cell phone was detected
        bounding_boxes — list of phone bounding-box dicts
    """
    # Step 1: Face detection (MediaPipe)
    face_found = detect_face(frame_rgb)

    # Step 2: Gaze detection — only when a face is present
    looking_away = False
    if face_found:
        looking_away = detect_gaze(frame_rgb)

    # Step 3: Phone detection (YOLO) — only on every Nth frame to save CPU
    phone_found:    bool      = False
    bounding_boxes: list[dict] = []

    if frame_idx % CFG["YOLO_FRAME_SKIP"] == 0:
        phone_found, bounding_boxes = detect_phone(frame_bgr)

    return face_found, looking_away, phone_found, bounding_boxes


# ─────────────────────────────────────────────────────────────────────────────
#  POST /analyze-frame/
#  Main endpoint — send one JPEG frame, receive detection JSON
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/analyze-frame/",
    response_model = AnalysisResult,
    summary        = "Analyze a single webcam frame",
    description    = (
        "Send a JPEG frame (from a React webcam) and an optional session_id. "
        "Receive face detection, gaze analysis, phone detection, "
        "and a cumulative cheat score."
    ),
)
async def analyze_frame(
    file:       UploadFile       = File(...,        description="JPEG frame from webcam"),
    session_id: Optional[str]    = Form(default=None, description="UUID — omit on first call"),
):
    # ── Session management ────────────────────────────────────────────────────
    # Generate a new session ID if this is the student's first request
    if not session_id:
        session_id = str(uuid.uuid4())

    state = get_session(session_id)
    state["frame_count"] += 1
    state["last_seen"]    = time.time()
    frame_idx             = state["frame_count"]

    # ── Read and decode the uploaded JPEG ────────────────────────────────────
    raw_bytes = await file.read()
    try:
        frame_bgr = decode_image(raw_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # ── Resize to a fixed width for faster inference ──────────────────────────
    frame_bgr = resize_frame(frame_bgr, CFG["INFER_WIDTH"])

    # ── Convert BGR → RGB (required by MediaPipe) ─────────────────────────────
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # ── Run all ML inference in a thread-pool executor ────────────────────────
    # Using run_in_executor prevents the synchronous MediaPipe / YOLO calls
    # from blocking the async event loop during concurrent requests.
    loop = asyncio.get_event_loop()
    face_found, looking_away, phone_found, bounding_boxes = await loop.run_in_executor(
        None,
        partial(_run_inference, frame_bgr, frame_rgb, frame_idx),
    )

    # ── Update the session cheat score ────────────────────────────────────────
    _update_cheat_score(state, face_found, looking_away, phone_found)

    cheat_score    = state["cheat_score"]
    cheating_alert = cheat_score >= CFG["CHEAT_ALERT_THRESHOLD"]
    ts             = time.time()

    # ── Persist the frame event to MongoDB (fire-and-forget) ─────────────────
    # We do not await this so the response is returned immediately.
    # Errors are caught inside the coroutine and logged — they won't crash
    # the request.
    asyncio.ensure_future(
        _persist_frame(
            session_id, frame_idx,
            face_found, looking_away, phone_found,
            cheat_score, cheating_alert, ts,
        )
    )

    # ── Structured log line ───────────────────────────────────────────────────
    logger.info(
        "session=%-36s | frame=%4d | face=%-5s | away=%-5s | "
        "phone=%-5s | score=%2d | alert=%s",
        session_id, frame_idx,
        face_found, looking_away, phone_found,
        cheat_score, cheating_alert,
    )

    # ── Build and return the JSON response ────────────────────────────────────
    return AnalysisResult(
        session_id     = session_id,
        face_detected  = face_found,
        looking_away   = looking_away,
        phone_detected = phone_found,
        cheat_score    = cheat_score,
        cheating_alert = cheating_alert,
        bounding_boxes = [BoundingBox(**b) for b in bounding_boxes],
        timestamp      = ts,
        frame_index    = frame_idx,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  INTERNAL — cheat score logic (pure function operating on session state)
# ─────────────────────────────────────────────────────────────────────────────

def _update_cheat_score(
    state:        dict,
    face_found:   bool,
    looking_away: bool,
    phone_found:  bool,
) -> int:
    """
    Apply penalty rules to the mutable session state dict.

    Penalties fire exactly once when a counter reaches its threshold,
    not on every subsequent frame, to avoid runaway score inflation.

    Returns: score delta added this frame (0 or more).
    """
    delta = 0

    # Rule 1 — No face detected
    if face_found:
        state["no_face_frames"] = 0     # reset counter when face returns
    else:
        state["no_face_frames"] += 1
        if state["no_face_frames"] == CFG["NO_FACE_THRESHOLD"]:
            state["cheat_score"] += CFG["SCORE_NO_FACE"]
            delta += CFG["SCORE_NO_FACE"]
            logger.warning(
                "NO FACE | session=%s | frames=%d | score→%d",
                state["session_id"], state["no_face_frames"], state["cheat_score"],
            )

    # Rule 2 — Student is looking away (only relevant when face is visible)
    if face_found:
        if looking_away:
            state["look_away_frames"] += 1
            if state["look_away_frames"] == CFG["LOOK_AWAY_THRESHOLD"]:
                state["cheat_score"] += CFG["SCORE_LOOK_AWAY"]
                delta += CFG["SCORE_LOOK_AWAY"]
                logger.warning(
                    "LOOKING AWAY | session=%s | frames=%d | score→%d",
                    state["session_id"], state["look_away_frames"], state["cheat_score"],
                )
        else:
            state["look_away_frames"] = 0   # reset when student looks back

    # Rule 3 — Phone detected
    if phone_found:
        state["phone_frames"] += 1
        if state["phone_frames"] == CFG["PHONE_THRESHOLD"]:
            state["cheat_score"] += CFG["SCORE_PHONE"]
            delta += CFG["SCORE_PHONE"]
            logger.warning(
                "PHONE DETECTED | session=%s | frames=%d | score→%d",
                state["session_id"], state["phone_frames"], state["cheat_score"],
            )
    else:
        state["phone_frames"] = 0           # reset when phone disappears

    return delta


async def _persist_frame(
    session_id: str, frame_index: int,
    face_detected: bool, looking_away: bool, phone_detected: bool,
    cheat_score: int, cheating_alert: bool, timestamp: float,
) -> None:
    """
    Fire-and-forget coroutine that inserts a frame_event document to MongoDB.
    Errors are logged but never propagate to the caller.
    """
    try:
        await database.log_frame_event(
            session_id     = session_id,
            frame_index    = frame_index,
            face_detected  = face_detected,
            looking_away   = looking_away,
            phone_detected = phone_detected,
            cheat_score    = cheat_score,
            cheating_alert = cheating_alert,
            timestamp      = timestamp,
        )
    except Exception as exc:
        logger.error("MongoDB insert failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
#  POST /reset-session/{session_id}
#  Clear a student's session (exam restart, proctor intervention)
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/reset-session/{session_id}",
    summary = "Reset (clear) a student session",
)
async def api_reset_session(session_id: str):
    found = reset_session(session_id)
    return {
        "status":     "reset" if found else "not_found",
        "session_id": session_id,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  GET /session/{session_id}
#  Inspect the in-memory state of one session (debug / admin)
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/session/{session_id}",
    summary = "Inspect in-memory state of a single session",
)
async def api_get_session(session_id: str):
    all_sessions = get_all_sessions()
    if session_id not in all_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return all_sessions[session_id]


# ─────────────────────────────────────────────────────────────────────────────
#  GET /session/{session_id}/log
#  Full MongoDB event log for one session (chronological)
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/session/{session_id}/log",
    summary = "Retrieve full MongoDB frame log for a session",
)
async def api_get_session_log(session_id: str):
    try:
        events = await database.get_session_log(session_id)
    except Exception as exc:
        logger.error("MongoDB query failed: %s", exc)
        raise HTTPException(status_code=503, detail="Database unavailable")

    return {
        "session_id":  session_id,
        "event_count": len(events),
        "events":      events,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  GET /sessions/
#  List all currently active in-memory sessions (admin dashboard)
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/sessions/",
    summary = "List all active in-memory sessions",
)
async def api_list_sessions():
    all_sessions = get_all_sessions()

    # Return a lightweight summary — not the full mutable state dict
    summaries = [
        {
            "session_id":  sid,
            "frame_count": data["frame_count"],
            "cheat_score": data["cheat_score"],
            "last_seen":   data["last_seen"],
            "created_at":  data["created_at"],
        }
        for sid, data in all_sessions.items()
    ]

    return {
        "active_sessions": len(summaries),
        "sessions":        summaries,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  GET /health
#  Quick liveness probe — confirms server is up and models are loaded
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/health", summary="Server and model health check")
async def health():
    from model_loader import models
    return {
        "status":        "ok",
        "models_loaded": list(models.keys()),
        "active_sessions": len(get_all_sessions()),
    }