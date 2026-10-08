import json
import logging
import time
from pathlib import Path
from typing import Any, Dict


LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "agent_events.log"


logger = logging.getLogger(
    "ai_quant_research"
)

logger.setLevel(logging.INFO)


if not logger.handlers:

    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8"
    )

    formatter = logging.Formatter(
        "%(message)s"
    )

    file_handler.setFormatter(
        formatter
    )

    logger.addHandler(
        file_handler
    )


def log_event(
    event: str,
    agent: str,
    status: str = "INFO",
    metadata: Dict[str, Any] | None = None
):
    """
    Record a structured agent event.
    """

    log_record = {
        "timestamp": time.time(),
        "event": event,
        "agent": agent,
        "status": status,
        "metadata": metadata or {}
    }

    logger.info(
        json.dumps(
            log_record,
            default=str
        )
    )


def log_agent_start(
    agent: str,
    metadata: Dict[str, Any] | None = None
):
    log_event(
        event="agent_started",
        agent=agent,
        status="STARTED",
        metadata=metadata
    )


def log_agent_end(
    agent: str,
    duration: float,
    metadata: Dict[str, Any] | None = None
):
    metadata = metadata or {}

    metadata["duration_seconds"] = duration

    log_event(
        event="agent_completed",
        agent=agent,
        status="SUCCESS",
        metadata=metadata
    )


def log_agent_error(
    agent: str,
    error: Exception,
    duration: float | None = None
):
    metadata = {
        "error": str(error)
    }

    if duration is not None:
        metadata["duration_seconds"] = duration

    log_event(
        event="agent_failed",
        agent=agent,
        status="ERROR",
        metadata=metadata
    )