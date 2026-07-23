# Online Exam Cheating Detection API

Real-time cheating detection backend built with **FastAPI**, **YOLOv8**, and **MediaPipe**.  
Accepts JPEG frames from a React webcam, returns face/gaze/phone analysis as JSON, and persists all events to **MongoDB**.

---

## Architecture Overview

```
React frontend
     │  POST /analyze-frame/ (JPEG frame)
     ▼
FastAPI (main.py)
     │
     ├── routes.py          — endpoint handlers
     │     ├── detectors.py — MediaPipe + YOLO inference (thread-pool)
     │     ├── cheat_logic  — scoring rules (inline in routes.py)
     │     └── session_store.py — in-memory session counters
     │
     ├── model_loader.py    — models loaded once at startup (lifespan)
     ├── database.py        — async MongoDB via motor
     ├── config.py          — all settings from .env / env vars
     └── logger_setup.py    — console + rotating file logging
```

---

## Quick Start

### 1. Prerequisites

| Tool | Version |
|---|---|
| Python | 3.10+ |
| MongoDB | 6.0+ (local or Atlas) |

### 2. Clone and set up the environment

```bash
git clone <repo-url>
cd exam_monitor

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env .env.local   # keep .env as the template
# Edit .env — at minimum set MONGO_URI if not using localhost
```

Key variables:

| Variable | Default | Description |
|---|---|---|
| `MONGO_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `MONGO_DB_NAME` | `exam_monitor` | Database name |
| `CHEAT_ALERT_THRESHOLD` | `10` | Score that triggers a cheating alert |
| `YOLO_FRAME_SKIP` | `5` | Run YOLO only every N frames |
| `GAZE_DEVIATION_LIMIT` | `0.06` | Sensitivity of gaze detection |
| `LOG_LEVEL` | `INFO` | DEBUG / INFO / WARNING / ERROR |
| `LOG_FILE` | `logs/exam_monitor.log` | Rotating log file path |

### 4. Run the development server

```bash
uvicorn main:app --reload
```

API is now available at **http://localhost:8000**  
Interactive docs: **http://localhost:8000/docs**

---

## API Endpoints

### `POST /analyze-frame/`

Analyze one webcam frame.

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | `UploadFile` (JPEG) | ✅ | Webcam frame |
| `session_id` | `string` | ❌ | UUID — omit on first call; use the returned ID on subsequent calls |

**Response** — `application/json`

```json
{
  "session_id":     "550e8400-e29b-41d4-a716-446655440000",
  "face_detected":  true,
  "looking_away":   false,
  "phone_detected": false,
  "cheat_score":    3,
  "cheating_alert": false,
  "bounding_boxes": [],
  "timestamp":      1714032000.123,
  "frame_index":    42
}
```

---

### `POST /reset-session/{session_id}`

Clear a student's in-memory session (exam restart / proctor intervention).

---

### `GET /session/{session_id}`

Return the current in-memory counters for one session.

---

### `GET /session/{session_id}/log`

Return all frame events stored in MongoDB for a session (chronological order).

---

### `GET /sessions/`

List all currently active in-memory sessions with summary stats.  
Useful for admin / proctor dashboards.

---

### `GET /health`

Server liveness check — returns loaded model names and active session count.

---

## Running with Docker

```bash
# Build the image
docker build -t exam-monitor .

# Run with a local MongoDB instance
docker run -p 8000:8000 \
  -e MONGO_URI=mongodb://host.docker.internal:27017 \
  exam-monitor

# Run with MongoDB Atlas
docker run -p 8000:8000 \
  -e MONGO_URI="mongodb+srv://user:pass@cluster.mongodb.net" \
  exam-monitor
```

---

## React Frontend Integration

```javascript
// Minimal example — capture frame and send to API
const analyzeFrame = async (blob, sessionId) => {
  const form = new FormData();
  form.append("file", blob, "frame.jpg");
  if (sessionId) form.append("session_id", sessionId);

  const res  = await fetch("http://localhost:8000/analyze-frame/", {
    method: "POST",
    body:   form,
  });
  return res.json();   // { face_detected, looking_away, phone_detected, cheat_score, ... }
};
```

Store the `session_id` from the first response and send it with every subsequent frame.

---

## MongoDB Document Schema

Collection: `frame_events`

```json
{
  "session_id":     "uuid",
  "frame_index":    42,
  "face_detected":  true,
  "looking_away":   false,
  "phone_detected": false,
  "cheat_score":    3,
  "cheating_alert": false,
  "timestamp":      1714032000.123
}
```

Indexes:
- `(session_id ASC, timestamp DESC)` — fast per-session queries
- `(cheat_score DESC)` — fast alert queries

---

## Performance Notes

| Optimization | Detail |
|---|---|
| Models loaded once | YOLO + MediaPipe initialized in FastAPI lifespan, never reloaded |
| Frame resizing | All frames resized to `INFER_WIDTH` (default 640px) before inference |
| YOLO frame-skip | YOLO runs only every `YOLO_FRAME_SKIP` frames (default: every 5th) |
| Thread-pool executor | All CPU-bound inference runs in `asyncio.run_in_executor` — event loop never blocks |
| Fire-and-forget DB | MongoDB insert is `ensure_future` — does not add latency to the API response |
| Single worker | `--workers 1` is intentional; models are module-level globals. Scale via multiple containers. |

---

## Project Structure

```
exam_monitor/
├── main.py            # FastAPI app, CORS, startup wiring
├── model_loader.py    # ML model + DB lifecycle (lifespan hook)
├── routes.py          # All API endpoints + cheat scoring
├── detectors.py       # Face / gaze / phone detection functions
├── session_store.py   # In-memory session state (swap for Redis in production)
├── database.py        # Async MongoDB via motor
├── schemas.py         # Pydantic response models
├── config.py          # All settings from environment variables
├── logger_setup.py    # Console + rotating file logging
├── requirements.txt   # Python dependencies
├── Dockerfile         # Production container
├── .env               # Environment variable template (do not commit secrets)
├── .gitignore
└── logs/              # Auto-created at startup
    └── exam_monitor.log
```
