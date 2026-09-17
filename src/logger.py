"""Structured logging configuration and in-memory log retention."""
import collections
from datetime import datetime, timezone
import logging
import sys
from typing import Any, Dict, List, Optional
from src.config import settings


class MemoryLogHandler(logging.Handler):
    """Circular in-memory log buffer that retains the most recent log records for failure diagnostics."""

    def __init__(self, capacity: int = 1000):
        super().__init__()
        self.capacity = capacity
        self.buffer: collections.deque[Dict[str, Any]] = collections.deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            formatted = self.format(record)
            now_iso = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
            entry: Dict[str, Any] = {
                "timestamp": now_iso,
                "level": record.levelname,
                "name": record.name,
                "message": record.getMessage(),
                "formatted": formatted,
            }
            if record.exc_info:
                import traceback
                entry["traceback"] = "".join(traceback.format_exception(*record.exc_info))
            self.buffer.append(entry)
        except Exception:
            self.handleError(record)

    def get_logs(
        self,
        limit: int = 100,
        level: Optional[str] = None,
        search: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        records = list(self.buffer)
        if level:
            target_level = level.upper()
            records = [r for r in records if r["level"] == target_level]
        if search:
            query = search.lower()
            records = [r for r in records if query in r["message"].lower() or query in r.get("formatted", "").lower()]
        return records[-limit:]

    def clear(self) -> None:
        self.buffer.clear()


_memory_handler = MemoryLogHandler(capacity=1000)


def setup_logger(name: str = "jeffsvideoscan") -> logging.Logger:
    """Configures and returns a structured logger with console output and in-memory retention."""
    logger_instance = logging.getLogger(name)
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logger_instance.setLevel(level)

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Add stream handler if not already present
    has_stream = any(isinstance(h, logging.StreamHandler) and not isinstance(h, MemoryLogHandler) for h in logger_instance.handlers)
    if not has_stream:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger_instance.addHandler(stream_handler)

    # Add memory handler if not already present
    has_memory = any(isinstance(h, MemoryLogHandler) for h in logger_instance.handlers)
    if not has_memory:
        _memory_handler.setLevel(logging.DEBUG)
        _memory_handler.setFormatter(formatter)
        logger_instance.addHandler(_memory_handler)

    return logger_instance


def get_system_logs(
    limit: int = 100,
    level: Optional[str] = None,
    search: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieves recent logs from the in-memory ring buffer."""
    return _memory_handler.get_logs(limit=limit, level=level, search=search)


def clear_system_logs() -> None:
    """Clears retained in-memory logs."""
    _memory_handler.clear()


logger = setup_logger()
