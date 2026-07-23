# =====================================================
#  Dockerfile — Production container for exam_monitor
#
#  Build:   docker build -t exam-monitor .
#  Run:     docker run -p 8000:8000 \
#               -e MONGO_URI=mongodb://host.docker.internal:27017 \
#               exam-monitor
#
#  Notes:
#  - Uses python:3.10-slim for a minimal image footprint
#  - opencv-python-headless avoids all GUI / X11 deps
#  - yolov8n.pt is downloaded at first run; mount a volume
#    or COPY it in if you want to avoid the download:
#      COPY yolov8n.pt /app/yolov8n.pt
# =====================================================

FROM python:3.10-slim

# Set a stable, non-root working directory
WORKDIR /app

# ── System dependencies required by OpenCV and MediaPipe ─────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1-mesa-glx \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first to leverage Docker layer caching —
# this layer only rebuilds when requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source code ───────────────────────────────────────────────────
COPY . .

# ── Runtime configuration ─────────────────────────────────────────────────────
# Disable Python output buffering so logs appear immediately in docker logs
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose the port uvicorn will listen on
EXPOSE 8000

# ── Start the server ──────────────────────────────────────────────────────────
# Single worker — models are in module-level globals.
# Scale by running multiple containers behind a load-balancer.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
