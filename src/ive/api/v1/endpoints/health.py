"""
Health Check Endpoints.

Provides liveness (/health) and readiness (/health/ready) probes.
These endpoints are excluded from authentication middleware.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter
from pydantic import BaseModel

log = structlog.get_logger(__name__)

router = APIRouter()


class HealthResponse(BaseModel):
    """Liveness probe response."""
    status: str
    version: str


class ReadinessResponse(BaseModel):
    """Readiness probe response."""
    status: str
    db: str
    redis: str


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health() -> HealthResponse:
    """Return OK if the application process is running."""
    return HealthResponse(status="ok", version="0.1.0")


@router.get("/health/ready", response_model=ReadinessResponse, summary="Readiness probe")
async def readiness() -> ReadinessResponse:
    """
    Check if the application is ready to serve traffic.

    Verifies:
        - PostgreSQL connection
        - Redis connection

    TODO:
        - Import and call db.database.check_db_connection()
        - Import and call a Redis ping utility
        - Return 503 if either dependency is unavailable
    """
    # TODO: Replace with real health checks
    db_status = "ok"    # await check_db_connection()
    redis_status = "ok" # await check_redis_connection()

    overall = "ready" if db_status == "ok" and redis_status == "ok" else "degraded"
    return ReadinessResponse(status=overall, db=db_status, redis=redis_status)
