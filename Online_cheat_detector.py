import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

# ===============================
# INITIALIZE MEDIAPIPE MODELS
# ===============================

# Face Detection
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

# Face Mesh (for gaze / looking away)
mp_mesh = mp.solutions.face_mesh
face_mesh = mp_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.7
)

# ===============================
# YOLOv8 MODEL (PHONE DETECTION)
# ===============================
yolo = YOLO("yolov8n.pt")

# ===============================
# CAMERA
# ===============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not opening")
    exit()

# ===============================
# VARIABLES
# ===============================
cheat_score = 0

no_face_frames = 0
look_away_frames = 0
phone_frames = 0

FRAME_THRESHOLD_NO_FACE = 15
FRAME_THRESHOLD_LOOK_AWAY = 12
FRAME_THRESHOLD_PHONE = 8

frame_count = 0

# ===============================
# MAIN LOOP
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ===============================
    # FACE DETECTION
    # ===============================
    face_result = face_detector.process(rgb)

    if face_result.detections:
        no_face_frames = 0
        cv2.putText(frame, "FACE DETECTED", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        no_face_frames += 1
        if no_face_frames == FRAME_THRESHOLD_NO_FACE:
            cheat_score += 1
            cv2.putText(frame, "NO FACE DETECTED", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # ===============================
    # LOOKING AWAY DETECTION
    # ===============================
    if face_result.detections:
        mesh_result = face_mesh.process(rgb)

        if mesh_result.multi_face_landmarks:
            landmarks = mesh_result.multi_face_landmarks[0]

            nose = landmarks.landmark[1]
            left_eye = landmarks.landmark[33]
            right_eye = landmarks.landmark[263]

            eye_center_x = (left_eye.x + right_eye.x) / 2

            if abs(nose.x - eye_center_x) > 0.06:
                look_away_frames += 1
                cv2.putText(frame, "LOOKING AWAY", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                look_away_frames = 0

            if look_away_frames == FRAME_THRESHOLD_LOOK_AWAY:
                cheat_score += 1

    # ===============================
    # PHONE DETECTION (RUN YOLO EVERY 5 FRAMES)
    # ===============================
    phone_detected = False

    if frame_count % 5 == 0:
        yolo_results = yolo(frame, conf=0.4, imgsz=320, verbose=False)

        for r in yolo_results:
            for box in r.boxes:
                if int(box.cls[0]) == 67:  # Mobile phone
                    phone_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  (0, 0, 255), 2)
                    cv2.putText(frame, "PHONE DETECTED",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 0, 255), 2)

    if phone_detected:
        phone_frames += 1
    else:
        phone_frames = 0

    if phone_frames == FRAME_THRESHOLD_PHONE:
        cheat_score += 2

    # ===============================
    # DISPLAY CHEAT SCORE
    # ===============================
    cv2.putText(frame, f"Cheat Score: {cheat_score}",
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 0), 2)

    if cheat_score >= 10:
        cv2.putText(frame, "CHEATING ALERT!",
                    (180, 230),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (0, 0, 255), 4)

    # ===============================
    # SHOW WINDOW
    # ===============================
    cv2.imshow("Online Exam Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

# ===============================
# CLEAN EXIT
# ===============================
cap.release()
cv2.destroyAllWindows()
