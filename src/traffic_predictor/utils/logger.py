"""Structured logging via loguru."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logger(level: str = "INFO", log_file: str | None = None) -> None:
    logger.remove()
    logger.add(sys.stderr, level=level.upper(), colorize=True,
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_file, level=level.upper(), rotation="10 MB", retention="7 days")


def get_logger(name: str):
    return logger.bind(name=name)
