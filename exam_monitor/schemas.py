# =====================================================
#  schemas.py — API ke response ka structure
#  Frontend ko exactly yahi JSON milega
# =====================================================

from pydantic import BaseModel


class BoundingBox(BaseModel):
    label:      str    # e.g. "phone"
    x1:         int
    y1:         int
    x2:         int
    y2:         int
    confidence: float  # YOLO ka confidence score


class AnalysisResult(BaseModel):
    session_id:     str
    face_detected:  bool
    looking_away:   bool
    phone_detected: bool
    cheat_score:    int
    cheating_alert: bool              # True jab score >= threshold
    bounding_boxes: list[BoundingBox] # phone ke boxes (draw karo frontend pe)
    timestamp:      float             # Unix timestamp
    frame_index:    int               # is session ka frame number