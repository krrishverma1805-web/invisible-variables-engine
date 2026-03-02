"""
Structured Logging — Invisible Variables Engine.

Configures ``structlog`` as the application-wide logging library.  Provides:

* :func:`setup_logging`   — configure structlog + stdlib logging bridge
* :func:`get_logger`      — return a bound logger for a module
* :func:`bind_context`    — add key-value pairs to the current async/thread context
* :func:`clear_context`   — clear all context variables
* :func:`log_request`     — emit a structured HTTP request log line
* :func:`log_duration`    — decorator that logs a function's execution time

All log records include ``service_name="ive"`` and ``version="0.1.0"`` via the
``add_app_info`` custom processor so that log aggregators (Datadog, Grafana Loki,
CloudWatch) can filter by service without extra config.

Usage::

    from ive.utils.logging import setup_logging, get_logger, bind_context, log_duration

    setup_logging(log_level="INFO", json_format=True)
    logger = get_logger(__name__)

    logger.info("experiment_started", experiment_id="abc-123", dataset_rows=50_000)

    bind_context(experiment_id="abc-123")
    logger.info("phase_started", phase="model")  # includes experiment_id automatically

    @log_duration
    def train_model() -> None:
        ...  # logs "train_model completed in 2.34s"
"""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any, TypeVar

import structlog
import structlog.contextvars
from structlog.types import EventDict, WrappedLogger

# ─── App metadata injected into every log record ──────────────────────────────
_APP_NAME: str = "ive"
_APP_VERSION: str = "0.1.0"

# ─── Type alias for functions wrapped by log_duration ─────────────────────────
F = TypeVar("F", bound=Callable[..., Any])

# ─── Module-level logger (available immediately after setup_logging) ───────────
log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Custom processors
# ---------------------------------------------------------------------------

def add_app_info(_: WrappedLogger, __: str, event_dict: EventDict) -> EventDict:
    """Inject ``service_name`` and ``version`` into every log record.

    This processor runs in the processor chain so every emitted event
    automatically carries service identity without the caller needing to
    pass it explicitly.

    Args:
        _: The wrapped logger instance (unused).
        __: The method name ("info", "warning", etc.) (unused).
        event_dict: The mutable log event dictionary.

    Returns:
        The event dict with ``service_name`` and ``version`` added.
    """
    event_dict["service_name"] = _APP_NAME
    event_dict["version"] = _APP_VERSION
    return event_dict


def _stdlib_to_structlog_level(_: WrappedLogger, method: str, event_dict: EventDict) -> EventDict:
    """Normalise stdlib log level names emitted by the structlog stdlib bridge."""
    level = event_dict.pop("level", method)
    event_dict["level"] = level.upper() if isinstance(level, str) else level
    return event_dict


# ---------------------------------------------------------------------------
# Public setup
# ---------------------------------------------------------------------------

def setup_logging(
    log_level: str = "INFO",
    json_format: bool = True,
) -> None:
    """Configure structlog and the stdlib logging bridge for the entire process.

    Must be called **once** at application startup — in ``main.py`` for the API,
    and in the Celery app setup for workers.  Subsequent calls are idempotent
    (structlog is reconfigured, which is fine in tests).

    Two rendering modes are supported:

    * **JSON** (``json_format=True``): Emits newline-delimited JSON.  Use in
      staging / production where a log aggregator (Loki, CloudWatch, Datadog)
      collects stdout.

    * **Console** (``json_format=False``): Emits coloured, human-readable
      ``key=value`` lines.  Use in development for readability.

    Args:
        log_level: Minimum log level.  One of ``"DEBUG"``, ``"INFO"``,
            ``"WARNING"``, ``"ERROR"``, ``"CRITICAL"``.
        json_format: If ``True``, emit JSON.  If ``False``, emit coloured
            console output.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # --- Shared processor pipeline (both modes) ----------------------------
    shared_processors: list[Any] = [
        # Pull context vars (experiment_id, request_id, etc.) into the event
        structlog.contextvars.merge_contextvars,
        # Add the log level string
        structlog.processors.add_log_level,
        # ISO-8601 timestamp
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        # Render any attached stack info
        structlog.processors.StackInfoRenderer(),
        # Format exc_info tuples into a readable traceback string
        structlog.processors.ExceptionRenderer(),
        # Decode bytes → str
        structlog.processors.UnicodeDecoder(),
        # Inject service metadata
        add_app_info,
    ]

    # --- Select final renderer based on format mode ------------------------
    if json_format:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(
            colors=True,
            exception_formatter=structlog.dev.plain_traceback,
        )

    structlog.configure(
        processors=[
            *shared_processors,
            # Must be last: prepares the event dict for the renderer
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # --- Stdlib logging bridge so uvicorn / SQLAlchemy / etc. flow through --
    stdlib_formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            # Rename stdlib "level" to match structlog conventions
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    root_handler = logging.StreamHandler()
    root_handler.setFormatter(stdlib_formatter)

    root_logger = logging.getLogger()
    # Remove any existing handlers to avoid duplicate output
    root_logger.handlers.clear()
    root_logger.addHandler(root_handler)
    root_logger.setLevel(numeric_level)

    # Quiet down noisy third-party libraries in production
    logging.getLogger("sqlalchemy.engine").setLevel(
        logging.DEBUG if log_level == "DEBUG" else logging.WARNING
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("celery").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Logger factory
# ---------------------------------------------------------------------------

def get_logger(name: str) -> structlog.BoundLogger:
    """Return a bound structlog logger for the given module name.

    This is the primary way to obtain a logger throughout the codebase::

        logger = get_logger(__name__)
        logger.info("model_trained", folds=5, metric=0.92)

    The returned logger is bound to ``{"logger": name}`` so every record
    emitted includes the originating module.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A :class:`structlog.BoundLogger` pre-bound with the module name.
    """
    return structlog.get_logger(name)


# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------

def bind_context(**kwargs: Any) -> None:
    """Bind key-value pairs to the current request / task context.

    All subsequent log calls in the same async task or thread will
    automatically include these values without explicit passing.

    Typical usage in FastAPI middleware or Celery task preamble::

        bind_context(experiment_id="abc-123", request_id="req-xyz")
        logger.info("processing")  # automatically includes experiment_id and request_id

    Args:
        **kwargs: Arbitrary key-value pairs to add to the log context.
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """Clear all context variables from the current async / thread context.

    Call at the end of each request (in middleware teardown) or Celery
    task completion to prevent context leaking into the next request::

        @app.middleware("http")
        async def logging_middleware(request, call_next):
            bind_context(request_id=str(uuid4()))
            try:
                return await call_next(request)
            finally:
                clear_context()
    """
    structlog.contextvars.clear_contextvars()


@contextmanager
def logging_context(**kwargs: Any) -> Generator[None, None, None]:
    """Context manager that binds and then clears logging context variables.

    Useful within a ``with`` block to scope context to a specific operation::

        with logging_context(experiment_id="abc-123"):
            logger.info("phase_started", phase="model")
        # experiment_id no longer in context here

    Args:
        **kwargs: Key-value pairs to bind for the duration of the block.

    Yields:
        Nothing — used purely for side effects.
    """
    bind_context(**kwargs)
    try:
        yield
    finally:
        # Only unset the keys we set, preserving any outer context
        structlog.contextvars.unbind_contextvars(*kwargs.keys())


# ---------------------------------------------------------------------------
# Request logging
# ---------------------------------------------------------------------------

def log_request(
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    **extra: Any,
) -> None:
    """Emit a structured HTTP access log record.

    Designed to be called from a FastAPI middleware after the response is
    complete::

        log_request("GET", "/api/v1/datasets", 200, 18.5, client_ip="10.0.0.1")

    Args:
        method: HTTP method (``"GET"``, ``"POST"``, etc.)
        path: Request path (without query string to avoid logging sensitive params).
        status_code: HTTP response status code.
        duration_ms: Total request duration in milliseconds.
        **extra: Additional fields to include (e.g. ``client_ip``, ``user_agent``).
    """
    level = "info" if status_code < 400 else ("warning" if status_code < 500 else "error")
    _logger = get_logger("ive.access")
    getattr(_logger, level)(
        "http_request",
        http_method=method,
        http_path=path,
        http_status=status_code,
        duration_ms=round(duration_ms, 2),
        **extra,
    )


# ---------------------------------------------------------------------------
# Performance logging decorator
# ---------------------------------------------------------------------------

def log_duration(_func: F | None = None, *, logger_name: str | None = None) -> F:
    """Decorator that logs the wall-clock execution time of a function.

    Works with both regular and ``async`` functions.

    Usage::

        @log_duration
        def train_model() -> None:
            ...  # logs: "train_model completed in 2.34s"

        @log_duration(logger_name="ive.pipeline")
        async def run_phase(ctx: PipelineContext) -> None:
            ...

    Args:
        _func: The function to wrap (when decorator is used without arguments).
        logger_name: Optional logger name override.  Defaults to
            ``"ive.performance"``.

    Returns:
        The wrapped function with timing instrumentation.
    """
    import asyncio

    def decorator(func: F) -> F:
        _log = get_logger(logger_name or "ive.performance")
        fn_name = f"{func.__module__}.{func.__qualname__}"

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                    elapsed = time.perf_counter() - start
                    _log.info(
                        "function_completed",
                        function=fn_name,
                        duration_s=round(elapsed, 4),
                    )
                    return result
                except Exception as exc:
                    elapsed = time.perf_counter() - start
                    _log.error(
                        "function_failed",
                        function=fn_name,
                        duration_s=round(elapsed, 4),
                        error=str(exc),
                    )
                    raise

            return async_wrapper  # type: ignore[return-value]

        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    elapsed = time.perf_counter() - start
                    _log.info(
                        "function_completed",
                        function=fn_name,
                        duration_s=round(elapsed, 4),
                    )
                    return result
                except Exception as exc:
                    elapsed = time.perf_counter() - start
                    _log.error(
                        "function_failed",
                        function=fn_name,
                        duration_s=round(elapsed, 4),
                        error=str(exc),
                    )
                    raise

            return sync_wrapper  # type: ignore[return-value]

    # Support both @log_duration and @log_duration(logger_name="...")
    if _func is not None:
        return decorator(_func)
    return decorator  # type: ignore[return-value]
