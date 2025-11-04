"""Utility helpers for consistent and colorized logging configuration across the project."""

from __future__ import annotations
import logging
import sys
from typing import Optional

_LOGGING_INITIALIZED = False


class ColorFormatter(logging.Formatter):
    """Formatter that adds colors and short logger names."""

    COLORS = {
        "DEBUG": "\033[94m",
        "INFO": "\033[92m",
        "WARNING": "\033[93m",
        "ERROR": "\033[91m",
        "CRITICAL": "\033[95m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        if "shortname" not in record.__dict__:
            record.shortname = record.name.split(".")[-1]
        color = self.COLORS.get(record.levelname, "")
        levelname_color = f"{color}{record.levelname}{self.RESET}"
        message = super().format(record)
        return message.replace(record.levelname, levelname_color, 1)


def setup_logging(level: int = logging.INFO, fmt: Optional[str] = None) -> None:
    """Configure global logging once."""
    global _LOGGING_INITIALIZED
    if _LOGGING_INITIALIZED:
        return
    log_format = fmt or "%(levelname)s - %(shortname)s - %(message)s"
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColorFormatter(log_format))
    logging.basicConfig(level=level, handlers=[handler])
    _LOGGING_INITIALIZED = True


def get_logger(name: str, custom_name: Optional[str] = None) -> logging.Logger:
    """Return a configured logger; optionally override displayed shortname."""
    setup_logging()
    base_logger = logging.getLogger(name)
    if not custom_name:
        return base_logger
    return logging.LoggerAdapter(base_logger, extra={"shortname": custom_name})  # type: ignore[return-value]
