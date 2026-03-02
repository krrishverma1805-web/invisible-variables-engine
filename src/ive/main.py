"""
IVE FastAPI Application Factory.

Creates and configures the FastAPI application instance with all middleware,
routers, and lifecycle event handlers. Use the `create_app()` factory function
to get the configured application.

Usage (uvicorn):
    uvicorn ive.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

from ive.api.middleware.auth import APIKeyMiddleware
from ive.api.middleware.error_handler import register_exception_handlers
from ive.api.middleware.rate_limit import RateLimitMiddleware
from ive.api.v1.router import api_v1_router
from ive.api.websocket.progress import router as ws_router
from ive.config import get_settings
from ive.db.database import close_db, init_db
from ive.utils.logging import configure_logging

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan (startup + shutdown events)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage application startup and shutdown.

    Startup:
        - Configure structured logging
        - Initialise DB connection pool
        - Optionally initialise Sentry

    Shutdown:
        - Close DB connection pool
        - Flush any pending log buffers
    """
    settings = get_settings()
    configure_logging(level=settings.log_level, json_logs=settings.is_production)

    log.info("ive.startup", env=settings.env, version="0.1.0")

    # TODO: Initialise Sentry if settings.sentry_dsn is set
    # if settings.sentry_dsn:
    #     import sentry_sdk
    #     sentry_sdk.init(dsn=settings.sentry_dsn, ...)

    await init_db()
    log.info("ive.db.ready")

    yield  # Application is running

    log.info("ive.shutdown")
    await close_db()


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        A fully configured FastAPI application ready to serve requests.
    """
    settings = get_settings()

    app = FastAPI(
        title="Invisible Variables Engine",
        description=(
            "Discovers hidden latent variables in datasets by analysing "
            "systematic model prediction errors."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        default_response_class=ORJSONResponse,
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------
    # CORS
    # ------------------------------------------------------------------
    # TODO: Tighten allowed origins in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not settings.is_production else ["https://your-domain.com"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Custom middleware (order matters — outermost first)
    # ------------------------------------------------------------------
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(APIKeyMiddleware)

    # ------------------------------------------------------------------
    # Exception handlers
    # ------------------------------------------------------------------
    register_exception_handlers(app)

    # ------------------------------------------------------------------
    # Routers
    # ------------------------------------------------------------------
    app.include_router(api_v1_router, prefix="/api/v1")
    app.include_router(ws_router, prefix="/ws")

    return app


# Module-level app instance for uvicorn / gunicorn
app = create_app()
