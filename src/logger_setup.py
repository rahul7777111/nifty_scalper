"""
Centralized logging configuration for NiftyScalper.

Replaces ad-hoc print() statements with structured logging.
"""

import logging
import os
import sys
from pathlib import Path
from datetime import datetime

def setup_logging(
    level: str = "INFO",
    log_dir: str = "logs",
    app_name: str = "niftyscalper",
) -> logging.Logger:
    """
    Configure root logger with console + rotating file handlers.

    Returns the application logger instance.
    """

    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Map text level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Root logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    # Avoid adding multiple handlers if already configured
    if logger.handlers:
        return logging.getLogger(app_name)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(numeric_level)
    console_fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    console.setFormatter(console_fmt)
    logger.addHandler(console)

    # File handler (daily log)
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = Path(log_dir) / f"{app_name}_{today}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(numeric_level)
    file_fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s [%(filename)s:%(lineno)d]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    app_logger = logging.getLogger(app_name)
    app_logger.info("Logging initialized (level=%s)", level)
    return app_logger


def get_logger(name: str) -> logging.Logger:
    """Return a named logger for a module."""
    return logging.getLogger(f"niftyscalper.{name}")