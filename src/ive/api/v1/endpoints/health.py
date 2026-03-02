"""
Health Check Endpoints — Invisible Variables Engine.

Provides liveness and readiness probes for Kubernetes / Docker health checks.
Both endpoints are exempt from API key authentication.

Routes:
    GET /health         — liveness: always 200 if the process is running
    GET /health/ready   — readiness: 200 if DB and Redis are reachable
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ive.api.v1.dependencies import get_db
from ive.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter()

_SERVICE = "ive-engine"
_VERSION = "0.1.0"


@router.get(
    "/health",
    summary="Liveness probe",
    description=(
        "Basic health check — always returns **200** if the API process is "
        "running.  Does **not** check database or Redis connectivity."
    ),
    tags=["Health"],
    response_class=JSONResponse,
)
async def health_check() -> JSONResponse:
    """Return 200 immediately.  Used by load balancers as a liveness probe."""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "service": _SERVICE,
            "version": _VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


@router.get(
    "/health/ready",
    summary="Readiness probe",
    description=(
        "Deep health check that verifies **PostgreSQL** and **Redis** "
        "connectivity.  Returns **200** when all dependencies are reachable, "
        "**503** otherwise."
    ),
    tags=["Health"],
    response_class=JSONResponse,
)
async def readiness_check(
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    """Verify database and Redis are reachable.

    Args:
        db: Async database session (injected by ``get_db`` dependency).

    Returns:
        JSON with per-dependency status and overall readiness.
    """
    checks: dict[str, str] = {}

    # ── PostgreSQL ────────────────────────────────────────────────────
    try:
        await db.execute(text("SELECT 1"))
        checks["database"] = "healthy"
    except Exception as exc:
        log.warning("health.db_unhealthy", error=str(exc))
        checks["database"] = f"unhealthy: {exc}"

    # ── Redis ─────────────────────────────────────────────────────────
    try:
        import redis.asyncio as aioredis

        from ive.config import get_settings
        settings = get_settings()
        r = aioredis.from_url(settings.redis_url, socket_connect_timeout=2)
        await r.ping()
        await r.aclose()
        checks["redis"] = "healthy"
    except ImportError:
        checks["redis"] = "unavailable: redis-py not installed"
    except Exception as exc:
        log.warning("health.redis_unhealthy", error=str(exc))
        checks["redis"] = f"unhealthy: {exc}"

    all_healthy = all(v == "healthy" for v in checks.values())
    status_code = 200 if all_healthy else 503

    log.debug(
        "health.readiness",
        status_code=status_code,
        checks=checks,
    )
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if all_healthy else "not_ready",
            "checks": checks,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
