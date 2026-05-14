"""
logger.py — Structured logging for Medical Document Intelligence System
"""

import sys
from pathlib import Path
from loguru import logger

from src.utils.config import LOGS_DIR


def setup_logger(name: str = "mdi") -> "logger":
    """Configure and return a named logger instance."""
    log_file = LOGS_DIR / f"{name}.log"

    logger.remove()  # Remove default handler

    # Console handler — colourised, readable
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        level="INFO",
        colorize=True,
    )

    # Rotating file handler — JSON-structured for dashboards
    logger.add(
        str(log_file),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="14 days",
        compression="zip",
        serialize=False,
    )

    return logger


# Default application logger
log = setup_logger("mdi")
