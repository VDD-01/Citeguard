"""Logging utility for CiteGuard-RAG.

Provides clear, structured logging for pipeline execution, retrieval scores,
validation decisions, and abstentions without exposing sensitive secrets.
"""

import logging
import sys
from typing import Optional


def setup_logger(name: str = "citeguard", level: Optional[str] = None) -> logging.Logger:
    """Set up and configure a structured logger.

    Args:
        name: Name of the logger.
        level: Optional log level ('DEBUG', 'INFO', 'WARNING', 'ERROR').

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if level:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    elif not logger.level:
        logger.setLevel(logging.INFO)

    return logger
