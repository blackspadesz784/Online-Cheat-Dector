# =====================================================
#  session_store.py — Har student ka alag session
#  Abhi in-memory dictionary use ho rahi hai.
#  Production mein isko Redis se replace karo.
# =====================================================

import time
import logging

logger = logging.getLogger("exam_monitor")

# Key   = session_id (UUID string)
# Value = us student ke saare counters
_store: dict[str, dict] = {}


def get_session(session_id: str) -> dict:
    """
    Session milti hai toh return karo,
    nahi milti toh fresh banake return karo.
    """
    if session_id not in _store:
        _store[session_id] = {
            "session_id":      session_id,
            "cheat_score":     0,
            "no_face_frames":  0,
            "look_away_frames": 0,
            "phone_frames":    0,
            "frame_count":     0,
            "created_at":      time.time(),
            "last_seen":       time.time(),
        }
        logger.info("🆕 Nayi session bani: %s", session_id)

    return _store[session_id]


def reset_session(session_id: str) -> bool:
    """Session delete karo (exam restart pe use karo). Returns True agar mili."""
    if session_id in _store:
        del _store[session_id]
        logger.info("🔄 Session reset: %s", session_id)
        return True
    return False


def get_all_sessions() -> dict:
    """Saari active sessions dekho (admin dashboard ke liye)."""
    return _store


# ─────────────────────────────────────────────────────
#  Redis ke saath replace karna ho toh:
# ─────────────────────────────────────────────────────
#
#  import redis, json
#  r = redis.Redis(host="redis", port=6379, decode_responses=True)
#
#  def get_session(session_id):
#      data = r.hgetall(f"session:{session_id}")
#      if not data:
#          data = {"cheat_score": 0, "no_face_frames": 0, ...}
#          r.hset(f"session:{session_id}", mapping=data)
#          r.expire(f"session:{session_id}", 7200)   # 2 ghante TTL
#      return {k: int(v) for k, v in data.items()}
#
# ─────────────────────────────────────────────────────