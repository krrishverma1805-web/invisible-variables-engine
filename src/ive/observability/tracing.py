"""OpenTelemetry tracing scaffolding (plan §117 + §150).

Opt-in via ``settings.enable_tracing``; when off, every entry point in this
module short-circuits to a no-op so production deployments without OTel pay
zero overhead. When on, we install:

    * a global ``TracerProvider`` with a Resource carrying ``service.name``,
    * an OTLP-HTTP exporter (or stdout when no endpoint is configured),
    * auto-instrumentation for FastAPI, httpx, asyncpg, SQLAlchemy, Celery,
      whichever of those packages are actually present at runtime.

Each experiment lifecycle gets a single root span ``ive.experiment`` whose
trace ID propagates into every downstream span (Groq HTTP, DB writes,
Celery chained tasks).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:  # pragma: no cover
    from opentelemetry.trace import Span, Tracer

log = structlog.get_logger(__name__)

_INSTALLED = False
_TRACER: Tracer | None = None  # type: ignore[assignment]


def install_tracing() -> bool:
    """Install OTel tracing once per process. Returns True when active.

    Idempotent — repeated calls are no-ops. Failures (missing packages,
    misconfiguration) are logged and swallowed; the rest of the application
    continues without tracing.
    """
    global _INSTALLED, _TRACER
    if _INSTALLED:
        return _TRACER is not None

    from ive.config import get_settings

    settings = get_settings()
    if not settings.enable_tracing:
        _INSTALLED = True
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )
    except ImportError:
        log.warning(
            "ive.tracing.disabled",
            reason="opentelemetry-sdk not installed; install the `tracing` extra",
        )
        _INSTALLED = True
        return False

    resource = Resource.create(
        {
            "service.name": settings.otel_service_name,
            "service.version": settings.app_version,
            "deployment.environment": settings.env.value,
        }
    )
    provider = TracerProvider(resource=resource)

    exporter: Any
    if settings.otel_exporter_otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            exporter = OTLPSpanExporter(
                endpoint=f"{settings.otel_exporter_otlp_endpoint.rstrip('/')}/v1/traces"
            )
        except ImportError:
            log.warning(
                "ive.tracing.otlp_unavailable",
                reason="opentelemetry-exporter-otlp-proto-http missing; "
                "falling back to stdout exporter",
            )
            exporter = ConsoleSpanExporter()
    else:
        exporter = ConsoleSpanExporter()

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _TRACER = trace.get_tracer("ive")

    _install_auto_instrumentation()

    log.info(
        "ive.tracing.installed",
        service_name=settings.otel_service_name,
        otlp_endpoint=settings.otel_exporter_otlp_endpoint or "<stdout>",
    )
    _INSTALLED = True
    return True


def _install_auto_instrumentation() -> None:
    """Best-effort auto-instrumentation. Each block is independent so a
    missing package only disables that one integration."""
    with suppress(ImportError, Exception):
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor().instrument()
    with suppress(ImportError, Exception):
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().instrument()
    with suppress(ImportError, Exception):
        from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor

        AsyncPGInstrumentor().instrument()
    with suppress(ImportError, Exception):
        from opentelemetry.instrumentation.sqlalchemy import (
            SQLAlchemyInstrumentor,
        )

        SQLAlchemyInstrumentor().instrument()
    with suppress(ImportError, Exception):
        from opentelemetry.instrumentation.celery import CeleryInstrumentor

        CeleryInstrumentor().instrument()


@contextmanager
def trace_span(
    name: str, attributes: dict[str, Any] | None = None
) -> Iterator[Any]:
    """Open a span when tracing is active; otherwise no-op.

    Usage::

        with trace_span("ive.experiment", {"experiment_id": str(eid)}):
            await pipeline.run_experiment(eid)
    """
    if _TRACER is None:
        yield None
        return
    span: Span
    with _TRACER.start_as_current_span(name) as span:  # type: ignore[assignment]
        if attributes:
            for k, v in attributes.items():
                with suppress(Exception):
                    span.set_attribute(k, v)
        try:
            yield span
        except Exception as exc:
            with suppress(Exception):
                span.record_exception(exc)
                from opentelemetry.trace import Status, StatusCode

                span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise


def reset_for_tests() -> None:  # pragma: no cover -- test hook
    """Reset module state so tests can re-install tracing."""
    global _INSTALLED, _TRACER
    _INSTALLED = False
    _TRACER = None
