# =====================================================
#  config.py — Centralised configuration
#
#  All values are read from environment variables.
#  Defaults are provided so the app works without a .env
#  file (useful for CI / Docker ENV flags).
#
#  Load order:
#    1. Values in .env (loaded by main.py at startup)
#    2. Real environment variables (override .env)
#    3. Hardcoded defaults below (fallback)
# =====================================================

import os

CFG: dict = {
    # ── Detection Thresholds ──────────────────────────────────────────────
    # Consecutive frames without a face before adding a no-face penalty
    "NO_FACE_THRESHOLD":   int(os.getenv("NO_FACE_THRESHOLD",   "15")),
    # Consecutive frames looking away before adding a look-away penalty
    "LOOK_AWAY_THRESHOLD": int(os.getenv("LOOK_AWAY_THRESHOLD", "12")),
    # Consecutive frames with a phone visible before adding a phone penalty
    "PHONE_THRESHOLD":     int(os.getenv("PHONE_THRESHOLD",      "8")),

    # ── Gaze Sensitivity ──────────────────────────────────────────────────
    # Horizontal distance (normalised 0–1) between nose tip and eye-center
    # that classifies the student as "looking away"
    "GAZE_DEVIATION_LIMIT": float(os.getenv("GAZE_DEVIATION_LIMIT", "0.06")),

    # ── YOLO Settings ─────────────────────────────────────────────────────
    # Minimum YOLO confidence to accept a detection
    "YOLO_CONF":      float(os.getenv("YOLO_CONF",      "0.4")),
    # Input resolution for YOLO inference (lower = faster)
    "YOLO_IMGSZ":     int(os.getenv("YOLO_IMGSZ",     "320")),
    # Only run YOLO on every Nth frame to save CPU
    "YOLO_FRAME_SKIP": int(os.getenv("YOLO_FRAME_SKIP", "5")),
    # COCO dataset class ID for "cell phone"
    "PHONE_CLASS_ID":  int(os.getenv("PHONE_CLASS_ID",  "67")),

    # ── Image Processing ──────────────────────────────────────────────────
    # Resize incoming frames to this width before running inference
    "INFER_WIDTH": int(os.getenv("INFER_WIDTH", "640")),

    # ── Cheat Score Penalties ─────────────────────────────────────────────
    "SCORE_NO_FACE":   int(os.getenv("SCORE_NO_FACE",   "1")),
    "SCORE_LOOK_AWAY": int(os.getenv("SCORE_LOOK_AWAY", "1")),
    "SCORE_PHONE":     int(os.getenv("SCORE_PHONE",     "2")),

    # ── Alert Threshold ───────────────────────────────────────────────────
    # Fire a cheating alert when cumulative cheat_score reaches this value
    "CHEAT_ALERT_THRESHOLD": int(os.getenv("CHEAT_ALERT_THRESHOLD", "10")),

    # ── MongoDB ───────────────────────────────────────────────────────────
    "MONGO_URI":     os.getenv("MONGO_URI",     "mongodb://localhost:27017"),
    "MONGO_DB_NAME": os.getenv("MONGO_DB_NAME", "exam_monitor"),

    # ── Logging ───────────────────────────────────────────────────────────
    "LOG_LEVEL":        os.getenv("LOG_LEVEL",        "INFO"),
    "LOG_FILE":         os.getenv("LOG_FILE",         "logs/exam_monitor.log"),
    "LOG_MAX_BYTES":    int(os.getenv("LOG_MAX_BYTES",    "5242880")),  # 5 MB
    "LOG_BACKUP_COUNT": int(os.getenv("LOG_BACKUP_COUNT", "3")),
}