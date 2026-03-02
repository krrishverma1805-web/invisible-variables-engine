"""
Logging configuration for IVE.
Uses structlog with stdlib logging backend.
"""

import logging

import structlog


def setup_logging(log_level: str = "INFO", json_format: bool = False) -> None:
    """Configure structured logging for the application."""

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper(), logging.INFO),
    )

    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if json_format:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str):
    """Return a structured logger."""
    return structlog.get_logger(name)


def bind_context(**kwargs):
    """Bind context variables to the current request."""
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context():
    """Clear context variables."""
    structlog.contextvars.clear_contextvars()


def log_request(
    method: str, path: str, status_code: int, duration_ms: float, client_ip: str
) -> None:
    """Log an HTTP request with structured data."""
    logger = get_logger("ive.request")
    logger.info(
        "request_finished",
        method=method,
        path=path,
        status_code=status_code,
        duration_ms=round(duration_ms, 2),
        client_ip=client_ip,
    )
