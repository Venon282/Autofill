"""Utility helpers for consistent and colorized logging configuration across the project."""

from __future__ import annotations
import logging
import sys
from typing import Optional

_LOGGING_INITIALIZED = False


class ColorFormatter(logging.Formatter):
    """Formatter that adds colors and short logger names."""

    COLORS = {
        "DEBUG": "\033[94m",   # bleu
        "INFO": "\033[92m",    # vert
        "WARNING": "\033[93m", # jaune
        "ERROR": "\033[91m",   # rouge
        "CRITICAL": "\033[95m" # magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        record.shortname = record.name.split(".")[-1]
        color = self.COLORS.get(record.levelname, "")
        levelname_color = f"{color}{record.levelname}{self.RESET}"
        message = super().format(record)
        return message.replace(record.levelname, levelname_color, 1)


def setup_logging(level: int = logging.INFO, format: Optional[str] = None) -> None:
    """Configure the global logging settings once."""
    global _LOGGING_INITIALIZED

    if _LOGGING_INITIALIZED:
        return

    fmt = format or "%(levelname)s - %(shortname)s - %(message)s"
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColorFormatter(fmt))
    logging.basicConfig(level=level, handlers=[handler])

    _LOGGING_INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger with consistent style."""
    setup_logging()
    return logging.getLogger(name)
