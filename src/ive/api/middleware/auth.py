"""
API Key Authentication Middleware — Invisible Variables Engine.

Validates the ``X-API-Key`` header on every request.
Keys are checked against the ``VALID_API_KEYS`` setting (comma-separated).

Exempt paths (no auth required):
    - ``/``                             — root redirect
    - ``/api/v1/health*``               — liveness / readiness probes
    - ``/docs``, ``/redoc``, ``/openapi.json`` — Swagger UI
    - ``/ws/*``                         — WebSocket handshakes (auth in handler)
"""

from __future__ import annotations

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from ive.config import get_settings
from ive.utils.logging import get_logger

log = get_logger(__name__)

# Paths exempted from API key authentication
_EXEMPT_PREFIXES: tuple[str, ...] = (
    "/",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/api/v1/health",
    "/ws/",
)


def _is_exempt(path: str) -> bool:
    """Return ``True`` if *path* does not require an API key."""
    if path == "/":
        return True
    return any(path.startswith(prefix) for prefix in _EXEMPT_PREFIXES[1:])


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that validates the ``X-API-Key`` header.

    On failure, returns HTTP 401 with a structured JSON error body.

    Design:
        - Uses a ``set`` lookup for O(1) key validation.
        - Exempt paths skip validation entirely (no Redis / DB round-trip).
        - The validated key is stored in ``request.state.api_key`` for
          downstream audit logging.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Intercept each request and validate the API key if required."""
        path = request.url.path

        # Skip auth for exempt paths
        if _is_exempt(path):
            return await call_next(request)

        settings = get_settings()
        api_key = request.headers.get(settings.api_key_header)
        valid_keys = settings.api_keys_set

        if not api_key or api_key not in valid_keys:
            log.warning(
                "ive.auth.rejected",
                path=path,
                method=request.method,
                key_present=bool(api_key),
            )
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": {
                        "code": "UNAUTHORIZED",
                        "message": "Invalid or missing API key.",
                        "request_id": getattr(request.state, "request_id", None),
                    }
                },
            )

        # Attach key to request state for downstream use
        request.state.api_key = api_key
        log.debug("ive.auth.accepted", path=path, method=request.method)
        return await call_next(request)


# ---------------------------------------------------------------------------
# Route-level dependency (alternative to middleware)
# ---------------------------------------------------------------------------

async def require_api_key(request: Request) -> str:
    """FastAPI dependency that extracts and validates the API key.

    Use as ``Depends(require_api_key)`` on individual routes when
    middleware is not in use or for extra-strict endpoints.

    Args:
        request: The incoming :class:`~fastapi.Request`.

    Returns:
        The validated API key string.

    Raises:
        HTTPException: 401 if the key is missing or invalid.
    """
    settings = get_settings()
    api_key = request.headers.get(settings.api_key_header)
    valid_keys = settings.api_keys_set

    if not api_key or api_key not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
    return api_key
