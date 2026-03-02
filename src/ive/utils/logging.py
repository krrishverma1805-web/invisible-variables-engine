"""
Structured Logging Configuration.

Configures structlog for the IVE application. In development mode,
logs are rendered as coloured, human-readable key=value pairs.
In production (is_production=True), logs are rendered as JSON for
ingestion by log aggregators (Datadog, CloudWatch, ELK, etc.).

Usage:
    from ive.utils.logging import configure_logging
    configure_logging(level="INFO", json_logs=True)

    import structlog
    log = structlog.get_logger(__name__)
    log.info("event.name", key="value")
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """
    Configure structlog with appropriate processors for the environment.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_logs: If True, emit JSON; if False, emit pretty console output.
    """
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if json_logs:
        processors: list[structlog.types.Processor] = [
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib logging so SQLAlchemy / uvicorn logs flow through
    logging.basicConfig(
        format="%(message)s",
        level=logging.getLevelName(level.upper()),
        stream=sys.stdout,
    )
