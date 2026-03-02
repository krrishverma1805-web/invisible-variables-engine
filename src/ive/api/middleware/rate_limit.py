"""
Rate Limiting Middleware.

Uses slowapi / Redis counters to enforce per-client request rate limits.
Clients are identified by their API key (or IP address as a fallback).
Exceeding the limit returns HTTP 429 with a Retry-After header.
"""

from __future__ import annotations

import time

import structlog
from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from ive.config import get_settings

log = structlog.get_logger(__name__)

# In-memory fallback store for development (not suitable for production multi-process)
# TODO: Replace with Redis-backed counter using aioredis for production
_rate_store: dict[str, tuple[int, float]] = {}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple sliding-window rate limiter.

    For each client (identified by API key or IP):
        - Tracks (request_count, window_start_time) in Redis (or in-memory dev store)
        - Resets counter after settings.rate_limit_window seconds
        - Returns 429 if count exceeds settings.rate_limit_requests

    TODO: Replace in-memory _rate_store with Redis for multi-process correctness.
    TODO: Implement proper sliding window algorithm (currently fixed window).
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Check rate limit before passing request downstream."""
        settings = get_settings()

        # Identify client by API key header or fallback to IP
        client_id = request.headers.get(settings.api_key_header) or (
            request.client.host if request.client else "unknown"
        )

        now = time.monotonic()
        count, window_start = _rate_store.get(client_id, (0, now))

        # Reset window if expired
        if now - window_start > settings.rate_limit_window:
            count = 0
            window_start = now

        count += 1
        _rate_store[client_id] = (count, window_start)

        if count > settings.rate_limit_requests:
            retry_after = int(settings.rate_limit_window - (now - window_start))
            log.warning(
                "ive.rate_limit.exceeded",
                client_id=client_id,
                count=count,
                limit=settings.rate_limit_requests,
            )
            response = JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": {
                        "code": "RATE_LIMITED",
                        "message": (
                            f"Rate limit exceeded. Max {settings.rate_limit_requests} "
                            f"requests per {settings.rate_limit_window}s."
                        ),
                    }
                },
            )
            response.headers["Retry-After"] = str(retry_after)
            return response

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(settings.rate_limit_requests)
        response.headers["X-RateLimit-Remaining"] = str(
            max(0, settings.rate_limit_requests - count)
        )
        return response
