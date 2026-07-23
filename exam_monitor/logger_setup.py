# =====================================================
#  logger_setup.py — Centralised logging configuration
#
#  Call setup_logging() once in main.py before the app
#  starts. Every other module then uses:
#      logger = logging.getLogger("exam_monitor")
#
#  Two handlers are attached:
#    1. StreamHandler  — coloured output to the terminal
#    2. RotatingFileHandler — persistent log file that
#       rolls over when it reaches LOG_MAX_BYTES
# =====================================================

import logging
import os
from logging.handlers import RotatingFileHandler

from config import CFG


def setup_logging() -> None:
    """
    Configure the root 'exam_monitor' logger.

    Creates the log directory if it does not exist, then
    attaches both a console handler and a rotating file
    handler using the values from CFG.
    """
    log_level = getattr(logging, CFG["LOG_LEVEL"].upper(), logging.INFO)
    log_file  = CFG["LOG_FILE"]

    # Ensure the logs/ directory exists before opening the file handler
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Shared formatter — same format for both handlers
    fmt = logging.Formatter(
        fmt     = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ───────────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(fmt)

    # ── Rotating file handler ─────────────────────────────────────────────
    # When the file reaches LOG_MAX_BYTES it is renamed to .log.1, .log.2, …
    # up to LOG_BACKUP_COUNT copies, then the oldest is deleted.
    file_handler = RotatingFileHandler(
        filename    = log_file,
        maxBytes    = CFG["LOG_MAX_BYTES"],
        backupCount = CFG["LOG_BACKUP_COUNT"],
        encoding    = "utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(fmt)

    # ── Attach both handlers to the named logger ──────────────────────────
    logger = logging.getLogger("exam_monitor")
    logger.setLevel(log_level)

    # Avoid duplicate handlers if setup_logging() is accidentally called twice
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    logger.info(
        "Logging initialised | level=%s | file=%s",
        CFG["LOG_LEVEL"], log_file,
    )
