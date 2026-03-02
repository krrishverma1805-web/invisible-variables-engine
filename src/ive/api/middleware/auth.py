"""
API Key Authentication Middleware.

Enforces API key authentication on all routes except health checks and
WebSocket handshakes. The API key must be present in the X-API-Key header
(configurable via settings.api_key_header).

Excluded paths (no auth required):
    - /api/v1/health
    - /docs, /redoc, /openapi.json
    - /ws/* (WebSocket auth handled at the handler level)
"""

from __future__ import annotations

import structlog
from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from ive.config import get_settings

log = structlog.get_logger(__name__)

# Paths that do NOT require authentication
_EXEMPT_PREFIXES = (
    "/docs",
    "/redoc",
    "/openapi.json",
    "/api/v1/health",
    "/ws/",
)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Starlette middleware that validates the API key on every request.

    Design notes:
        - Uses a set lookup O(1) for key validation
        - Exempt paths skip validation entirely (no Redis/DB hit)
        - Returns 401 with structured JSON error on failure
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Intercept each request and validate the API key if required."""
        settings = get_settings()

        # Skip auth for exempt paths
        path = request.url.path
        if any(path.startswith(prefix) for prefix in _EXEMPT_PREFIXES):
            return await call_next(request)

        # Extract API key from header
        api_key = request.headers.get(settings.api_key_header)
        valid_keys = settings.get_api_keys()

        if not api_key or api_key not in valid_keys:
            log.warning(
                "ive.auth.rejected",
                path=path,
                method=request.method,
                key_present=bool(api_key),
            )
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": {
                        "code": "UNAUTHORIZED",
                        "message": "Invalid or missing API key.",
                    }
                },
            )

        # Attach the raw key to request state for downstream logging
        request.state.api_key = api_key
        return await call_next(request)
