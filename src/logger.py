"""Structured logging configuration."""
import logging
import sys
from src.config import settings


def setup_logger(name: str = "jeffsvideoscan") -> logging.Logger:
    """Configures and returns a structured logger."""
    logger = logging.getLogger(name)
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logger()
