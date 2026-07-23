# =====================================================
#  main.py — Application entry point
#
#  Run with:
#      uvicorn main:app --reload            (development)
#      uvicorn main:app --host 0.0.0.0 \   (production)
#                       --port 8000 \
#                       --workers 1
#
#  Workers must be 1 because MediaPipe and YOLO models
#  are held in module-level globals.  For horizontal
#  scaling, deploy multiple containers behind a
#  load-balancer instead.
# =====================================================

from dotenv import load_dotenv

# Load .env before anything else so os.getenv() calls
# inside config.py pick up the file values.
load_dotenv()

import logging  # noqa: E402 — must come after load_dotenv

from fastapi import FastAPI                       # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from logger_setup import setup_logging            # noqa: E402
from model_loader import lifespan                 # noqa: E402
from routes import router                         # noqa: E402

# Initialise logging first so every subsequent import can use the logger
setup_logging()

logger = logging.getLogger("exam_monitor")

# ── FastAPI application ───────────────────────────────────────────────────────
# lifespan manages ML model loading on startup and cleanup on shutdown.
app = FastAPI(
    title       = "Online Exam Cheating Detection API",
    description = (
        "Real-time cheating detection using YOLOv8 + MediaPipe.\n\n"
        "Send JPEG frames from a React frontend; receive JSON with "
        "face, gaze, and phone analysis results."
    ),
    version     = "2.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# ── CORS — allow the React dev server and any deployed frontend ───────────────
# In production replace ["*"] with your specific domain(s).
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],    # e.g. ["https://your-exam-platform.com"]
    allow_methods     = ["*"],
    allow_headers     = ["*"],
    allow_credentials = False,
)

# ── Mount all API routes ──────────────────────────────────────────────────────
app.include_router(router)

logger.info("FastAPI app configured — waiting for uvicorn to start lifespan")