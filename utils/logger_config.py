"""Centralized structlog configuration for the Demand-Forecasting project."""
from __future__ import annotations

import logging
import os
import sys
from contextvars import ContextVar
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Dict
from structlog.processors import CallsiteParameter

import structlog

from config.config_loader import ConfigLoader

_CONFIGURED = False
_REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="-")
_CONFIG_LOADER = ConfigLoader()


def set_request_id(request_id: str) -> None:
    """Bind a request ID for the current context (Prefect run, CLI invocation, etc.)."""

    _REQUEST_ID.set(request_id)


def _get_log_level() -> int:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    return getattr(logging, level_name, logging.INFO)


def _ensure_log_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _build_handlers(log_path: Path) -> list[logging.Handler]:
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(sort_keys=True)
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    rotating_handler = TimedRotatingFileHandler(
        filename=_ensure_log_dir(log_path), when="midnight", backupCount=14, encoding="utf-8"
    )
    rotating_handler.setFormatter(formatter)
    return [console_handler, rotating_handler]


def _add_request_id(_: logging.Logger, __: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    event_dict.setdefault("request_id", _REQUEST_ID.get())
    return event_dict


def configure_logging() -> None:
    """Configure structlog + stdlib logging once per interpreter session."""

    global _CONFIGURED
    if _CONFIGURED:
        return

    config = _CONFIG_LOADER.load_config()
    log_path: Path = Path(config["paths"]["log_file"])
    log_level = _get_log_level()

    handlers = _build_handlers(log_path)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(log_level)
    for handler in handlers:
        root_logger.addHandler(handler)

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        _add_request_id,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", key="timestamp"),
        structlog.processors.CallsiteParameterAdder(
            parameters={
                CallsiteParameter.FILENAME,
                CallsiteParameter.FUNC_NAME,
                CallsiteParameter.LINENO,
            }
        ),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ]

    structlog.configure(
        processors=shared_processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _CONFIGURED = True


# Configure logging immediately for module importers --------------------------
configure_logging()
