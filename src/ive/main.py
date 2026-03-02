"""
IVE FastAPI Application Factory.

Creates and configures the FastAPI application instance with all middleware,
routers, and lifecycle event handlers.

Usage (uvicorn):
    uvicorn ive.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from ive.api.middleware.auth import APIKeyMiddleware
from ive.api.middleware.error_handler import register_exception_handlers
from ive.api.middleware.rate_limit import setup_rate_limiter
from ive.api.v1.router import api_v1_router
from ive.api.websocket.progress import router as ws_router
from ive.config import get_settings
from ive.db.database import close_db, init_db
from ive.utils.logging import (
    bind_context,
    get_logger,
    log_request,
    setup_logging,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Request ID Middleware
# ---------------------------------------------------------------------------


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Inject a unique ``X-Request-ID`` header into every request/response.

    If the client sends an ``X-Request-ID`` header it is re-used (useful for
    distributed tracing); otherwise a fresh UUID4 is generated.

    The ID is also stored in ``request.state.request_id`` and bound into the
    structlog context so every log emitted during the request includes it.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        bind_context(request_id=request_id)

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


# ---------------------------------------------------------------------------
# Request Logging Middleware
# ---------------------------------------------------------------------------


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every HTTP request with method, path, status, and duration."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000

        # Skip health checks and docs from cluttering logs
        path = request.url.path
        if not path.startswith(("/api/v1/health", "/docs", "/redoc", "/openapi.json")):
            log_request(
                method=request.method,
                path=path,
                status_code=response.status_code,
                duration_ms=duration_ms,
                client_ip=request.client.host if request.client else "unknown",
            )

        return response


# ---------------------------------------------------------------------------
# Lifespan (startup + shutdown events)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown."""
    settings = get_settings()

    setup_logging(
        log_level=settings.log_level,
        json_format=settings.is_production,
    )

    log.info(
        "ive.startup",
        env=settings.env.value,
        version=settings.app_version,
        debug=settings.debug,
    )

    # Sentry (optional)
    if settings.sentry_dsn:
        try:
            import sentry_sdk

            sentry_sdk.init(
                dsn=settings.sentry_dsn,
                environment=settings.env.value,
                release=f"ive@{settings.app_version}",
            )
            log.info("ive.sentry.initialised")
        except ImportError:
            log.warning("ive.sentry.skipped", reason="sentry-sdk not installed")

    # Database — graceful degradation: if DB is unavailable, the app still
    # boots (health endpoint returns status="degraded").
    try:
        await init_db()
        log.info("ive.db.ready")
    except Exception as exc:
        log.error("ive.db.init_failed", error=str(exc))

    yield  # Application is running

    log.info("ive.shutdown")
    try:
        await close_db()
    except Exception:
        pass  # best-effort during shutdown


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        A fully configured :class:`fastapi.FastAPI` instance ready to serve.
    """
    settings = get_settings()

    app = FastAPI(
        title="Invisible Variables Engine",
        description=(
            "Production-grade data science system that discovers hidden "
            "latent variables in datasets by analyzing systematic model "
            "prediction errors."
        ),
        version="0.1.0",
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None,
        openapi_url="/openapi.json" if settings.is_development else None,
        lifespan=lifespan,
    )

    # ── 1. CORS (outermost — first added) ──────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=(["*"] if settings.is_development else ["https://ive.example.com"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── 2. Request ID ──────────────────────────────────────────────────────
    app.add_middleware(RequestIDMiddleware)

    # ── 3. Request Logging ─────────────────────────────────────────────────
    app.add_middleware(RequestLoggingMiddleware)

    # ── 4. Rate Limiting ───────────────────────────────────────────────────
    setup_rate_limiter(app)

    # ── 5. API Key Authentication ──────────────────────────────────────────
    app.add_middleware(APIKeyMiddleware)

    # ── 6. Exception handlers ──────────────────────────────────────────────
    register_exception_handlers(app)

    # ── Routers ────────────────────────────────────────────────────────────
    app.include_router(api_v1_router, prefix="/api/v1")
    app.include_router(ws_router, prefix="/ws")

    # ── Root redirect → health ─────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        """Redirect root to the health check endpoint."""
        return RedirectResponse(url="/api/v1/health")

    return app


# Module-level app instance for uvicorn / gunicorn
app = create_app()
