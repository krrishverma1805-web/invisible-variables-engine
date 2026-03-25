"""
Rate Limiting Middleware — Invisible Variables Engine.

Uses ``slowapi`` (a Starlette/FastAPI adapter around ``limits``) to enforce
per-client request rate limits.

Clients are identified by their ``X-API-Key`` header.  If no key is present
the client IP address is used as the fallback identifier.

Configuration (from ``ive.config.Settings``):
    rate_limit_requests — max requests per window (default 100)
    rate_limit_window   — window size in seconds (default 60)

Exceeding the limit returns HTTP 429 with a JSON body and ``Retry-After``
header.

NOTE: ``slowapi`` must be in ``pyproject.toml`` (``slowapi = "^0.1.9"``).
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ive.config import get_settings
from ive.utils.logging import get_logger

log = get_logger(__name__)


def _key_func(request: Request) -> str:
    """Identify the client for rate-limiting purposes.

    Priority:
        1. ``X-API-Key`` header (per-key quotas)
        2. Client IP address (anonymous requests)

    Args:
        request: The incoming Starlette request.

    Returns:
        A string identifying the client.
    """
    settings = get_settings()
    api_key = request.headers.get(settings.api_key_header)
    if api_key:
        return f"key:{api_key}"
    if request.client:
        return f"ip:{request.client.host}"
    return "ip:unknown"


def setup_rate_limiter(app: FastAPI) -> None:
    """Configure ``slowapi`` rate limiting on the FastAPI application.

    Should be called during ``create_app()`` in ``main.py``.

    Args:
        app: The FastAPI application instance.
    """
    try:
        from slowapi import Limiter
        from slowapi.errors import RateLimitExceeded
    except ImportError:
        log.warning(
            "ive.rate_limit.disabled",
            reason="slowapi not installed — rate limiting is off",
        )
        return

    settings = get_settings()
    default_limit = f"{settings.rate_limit_requests}/{settings.rate_limit_window}second"

    limiter = Limiter(
        key_func=_key_func,
        default_limits=[default_limit],
        enabled=True,
        storage_uri=settings.redis_url,
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _custom_rate_limit_handler)

    log.info(
        "ive.rate_limit.configured",
        default_limit=default_limit,
    )


async def _custom_rate_limit_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a structured JSON 429 response when the rate limit is exceeded."""
    settings = get_settings()
    retry_after = settings.rate_limit_window

    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "code": "RATE_LIMITED",
                "message": (
                    f"Rate limit exceeded. Max {settings.rate_limit_requests} "
                    f"requests per {settings.rate_limit_window}s."
                ),
                "request_id": getattr(request.state, "request_id", None),
            }
        },
        headers={"Retry-After": str(retry_after)},
    )
