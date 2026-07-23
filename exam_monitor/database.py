# =====================================================
#  database.py — Async MongoDB integration (motor)
#
#  Provides three public functions used by the rest of
#  the application:
#
#    init_db()           — called once at server startup
#                          to create indexes and verify
#                          the connection.
#
#    log_frame_event()   — insert one document per
#                          analyzed webcam frame.
#
#    get_session_log()   — retrieve all frame events for
#                          a given session (admin / debug).
#
#  The MongoDB client is module-level so it is shared
#  across every request (motor handles its own
#  connection pool internally).
#
#  To switch from a local instance to MongoDB Atlas,
#  update MONGO_URI in .env — no code changes needed.
# =====================================================

import logging
from typing import Any

import motor.motor_asyncio
from pymongo import ASCENDING, DESCENDING

from config import CFG

logger = logging.getLogger("exam_monitor")

# ── Motor client — created once, reused for the app lifetime ─────────────────
# motor.motor_asyncio.AsyncIOMotorClient is thread-safe and manages its own
# connection pool; do NOT create a new client per request.
_client: motor.motor_asyncio.AsyncIOMotorClient | None = None
_db     = None
_col    = None   # collection: frame_events


def _get_collection():
    """Return the frame_events collection, raising if not yet initialised."""
    if _col is None:
        raise RuntimeError(
            "Database not initialised. Call await init_db() at startup."
        )
    return _col


# ── Public API ────────────────────────────────────────────────────────────────

async def init_db() -> None:
    """
    Connect to MongoDB and ensure required indexes exist.

    Called once inside the FastAPI lifespan startup block.
    Index on (session_id, timestamp DESC) speeds up per-session
    log queries significantly.
    """
    global _client, _db, _col

    uri     = CFG["MONGO_URI"]
    db_name = CFG["MONGO_DB_NAME"]

    logger.info("Connecting to MongoDB | uri=%s | db=%s", uri, db_name)

    _client = motor.motor_asyncio.AsyncIOMotorClient(uri)
    _db     = _client[db_name]
    _col    = _db["frame_events"]

    # Create compound index for fast per-session queries sorted by time
    await _col.create_index(
        [("session_id", ASCENDING), ("timestamp", DESCENDING)],
        name       = "idx_session_timestamp",
        background = True,
    )
    # Sparse index on cheat_score for potential alert queries
    await _col.create_index(
        [("cheat_score", DESCENDING)],
        name       = "idx_cheat_score",
        background = True,
    )

    # Ping the server to verify the connection is live
    await _client.admin.command("ping")
    logger.info("MongoDB connection verified — collection: frame_events")


async def log_frame_event(
    session_id:     str,
    frame_index:    int,
    face_detected:  bool,
    looking_away:   bool,
    phone_detected: bool,
    cheat_score:    int,
    cheating_alert: bool,
    timestamp:      float,
) -> str:
    """
    Insert a single frame analysis result into the frame_events collection.

    Returns the inserted document's string ID (useful for debugging).
    Each document represents one webcam frame processed by the API.
    """
    col = _get_collection()

    doc: dict[str, Any] = {
        "session_id":     session_id,
        "frame_index":    frame_index,
        "face_detected":  face_detected,
        "looking_away":   looking_away,
        "phone_detected": phone_detected,
        "cheat_score":    cheat_score,
        "cheating_alert": cheating_alert,
        "timestamp":      timestamp,
    }

    result = await col.insert_one(doc)
    logger.debug(
        "DB insert | session=%s | frame=%d | _id=%s",
        session_id, frame_index, result.inserted_id,
    )
    return str(result.inserted_id)


async def get_session_log(session_id: str) -> list[dict]:
    """
    Retrieve all frame_events documents for a given session,
    sorted by timestamp ascending (oldest frame first).

    Returns a list of plain dicts (ObjectId converted to string).
    Used by the GET /session/{session_id}/log endpoint.
    """
    col     = _get_collection()
    cursor  = col.find(
        {"session_id": session_id},
        {"_id": 0},           # exclude internal MongoDB _id from response
        sort=[("timestamp", ASCENDING)],
    )
    docs = await cursor.to_list(length=None)   # fetch all documents
    return docs


async def close_db() -> None:
    """
    Close the motor client gracefully.

    Called inside the FastAPI lifespan shutdown block.
    """
    global _client
    if _client is not None:
        _client.close()
        logger.info("MongoDB connection closed")
