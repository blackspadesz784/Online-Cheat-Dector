# =====================================================
#  cheat_logic.py — Cheat score calculate karna
#  Original ke frame-based counters ko yahan session
#  state ke saath manage kiya gaya hai
# =====================================================

import logging
from config import CFG

logger = logging.getLogger("exam_monitor")


def update_cheat_score(
    state: dict,
    face_found:   bool,
    looking_away: bool,
    phone_found:  bool,
) -> int:
    """
    Teen conditions check karo aur session state update karo.
    Jab bhi koi threshold poora ho, score mein penalty add hoti hai.

    Returns: is frame mein kitna score badha (0 ya zyada)
    """
    delta = 0  # is frame mein kitni penalty lagi

    # ─────────────────────────────────────────────
    #  CONDITION 1: Face nahi dikh raha
    # ─────────────────────────────────────────────
    if face_found:
        state["no_face_frames"] = 0  # reset — face wapas aa gaya
    else:
        state["no_face_frames"] += 1

        # Exactly threshold pe penalty lagao (bar bar nahi)
        if state["no_face_frames"] == CFG["NO_FACE_THRESHOLD"]:
            state["cheat_score"] += CFG["SCORE_NO_FACE"]
            delta += CFG["SCORE_NO_FACE"]
            logger.warning(
                "⚠️  [%s] NO FACE — %d frames se | score → %d",
                state["session_id"], state["no_face_frames"], state["cheat_score"],
            )

    # ─────────────────────────────────────────────
    #  CONDITION 2: Kahi aur dekh raha hai
    #  (sirf tab check karo jab face dikh raha ho)
    # ─────────────────────────────────────────────
    if face_found:
        if looking_away:
            state["look_away_frames"] += 1

            if state["look_away_frames"] == CFG["LOOK_AWAY_THRESHOLD"]:
                state["cheat_score"] += CFG["SCORE_LOOK_AWAY"]
                delta += CFG["SCORE_LOOK_AWAY"]
                logger.warning(
                    "⚠️  [%s] LOOKING AWAY — %d frames se | score → %d",
                    state["session_id"], state["look_away_frames"], state["cheat_score"],
                )
        else:
            state["look_away_frames"] = 0  # reset — wapas screen dekha

    # ─────────────────────────────────────────────
    #  CONDITION 3: Phone dikh raha hai
    # ─────────────────────────────────────────────
    if phone_found:
        state["phone_frames"] += 1

        if state["phone_frames"] == CFG["PHONE_THRESHOLD"]:
            state["cheat_score"] += CFG["SCORE_PHONE"]
            delta += CFG["SCORE_PHONE"]
            logger.warning(
                "⚠️  [%s] PHONE DETECTED — %d frames se | score → %d",
                state["session_id"], state["phone_frames"], state["cheat_score"],
            )
    else:
        state["phone_frames"] = 0  # reset

    return delta