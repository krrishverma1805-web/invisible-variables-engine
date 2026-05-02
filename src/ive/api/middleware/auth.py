"""
API Key Authentication Middleware — Invisible Variables Engine.

Validates the ``X-API-Key`` header on every non-exempt request via the
:mod:`ive.auth.resolver` two-tier lookup:

1. DB-managed keys (``api_keys`` table) — full scope and rate-limit data.
2. Legacy env-CSV keys (``VALID_API_KEYS``) — accepted with a default
   ``read+write`` scope; ``api_key_id`` is ``None``.

Outcomes are written to the ``auth_audit_log`` table on the same
transaction-bounded session used for resolution.  When the DB is
unavailable the legacy CSV fallback still works; the audit-log write is
best-effort and never blocks the request.

Exempt paths (no auth required):
    - ``/``                             — root redirect
    - ``/api/v1/health*``               — liveness / readiness probes
    - ``/docs``, ``/redoc``, ``/openapi.json`` — Swagger UI
    - ``/ws/*``                         — WebSocket handshakes (auth in handler)

Plan reference: §47 (resolved), §113, §155.
"""

from __future__ import annotations

import logging

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from ive.auth.resolver import resolve_api_key
from ive.auth.scopes import AuthContext, AuthOutcome
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
    # Phase C2.2 — public share-read endpoint is token-gated inside the
    # handler; X-API-Key is not required.
    "/api/v1/share/",
    # Phase C4 — Prometheus exporter exposed un-authenticated; deployments
    # that need it locked down put a network-level firewall in front.
    "/metrics",
)


def _is_exempt(path: str) -> bool:
    """Return ``True`` if *path* does not require an API key."""
    if path == "/":
        return True
    return any(path.startswith(prefix) for prefix in _EXEMPT_PREFIXES[1:])


class APIKeyMiddleware(BaseHTTPMiddleware):  # type: ignore[misc]
    """Starlette middleware that authenticates each non-exempt request.

    Attaches an :class:`~ive.auth.scopes.AuthContext` to
    ``request.state.auth`` on success.  Writes an ``auth_audit_log`` row
    for both successes and failures (best-effort — log write failures do
    not block the request).
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Intercept each request and validate the API key if required."""
        path = request.url.path

        # Skip auth for exempt paths
        if _is_exempt(path):
            return await call_next(request)

        settings = get_settings()
        raw_key = request.headers.get(settings.api_key_header)

        # The resolver may need an async DB session. We pass None when
        # there is no app-level session factory (e.g. test harness
        # without DB) — the env-CSV fallback still works.
        session = await _maybe_get_session(request)
        try:
            outcome: AuthOutcome = await resolve_api_key(raw_key, session, settings)
        finally:
            if session is not None:
                # Resolver did not commit; we close to release the
                # connection regardless of outcome.
                await session.close()

        if not outcome.authenticated:
            log.warning(
                "ive.auth.rejected",
                path=path,
                method=request.method,
                event_type=outcome.event_type,
                key_present=bool(raw_key),
            )
            await _write_audit_log_safe(request, outcome, status_code=401)
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

        # Attach the resolved context. ``request.state.api_key`` is kept
        # for backwards compatibility with code that read the raw key.
        ctx: AuthContext = outcome.context  # type: ignore[assignment]
        request.state.auth = ctx
        request.state.api_key = raw_key
        log.debug(
            "ive.auth.accepted",
            path=path,
            method=request.method,
            api_key_name=ctx.api_key_name,
            scopes=[s.value for s in ctx.scopes],
        )

        response = await call_next(request)

        await _write_audit_log_safe(request, outcome, status_code=response.status_code)
        return response


async def _maybe_get_session(_request: Request) -> AsyncSession | None:
    """Acquire an :class:`AsyncSession` for the resolver, or ``None``.

    Uses the application-level session factory when available (set in
    ``ive.db.database``).  In test environments without a DB the factory
    is absent and we fall through to the env-CSV-only path.

    The ``_request`` parameter is reserved for future per-request session
    overrides (e.g. read-replica routing) — currently unused.
    """
    try:
        from ive.db.database import get_session_factory
    except Exception:
        return None
    try:
        factory = get_session_factory()
    except Exception:
        return None
    if factory is None:
        return None
    try:
        return factory()
    except Exception:
        return None


async def _write_audit_log_safe(
    request: Request,
    outcome: AuthOutcome,
    *,
    status_code: int,
) -> None:
    """Write an ``auth_audit_log`` row. Best-effort: never raises."""
    try:
        from ive.db.database import get_session_factory
        from ive.db.models import AuthAuditLog
    except Exception:  # pragma: no cover - defensive
        return
    try:
        factory = get_session_factory()
    except Exception:
        factory = None
    if factory is None:
        return

    api_key_id = None
    api_key_name = None
    if outcome.context is not None:
        api_key_id = outcome.context.api_key_id
        api_key_name = outcome.context.api_key_name

    row = AuthAuditLog(
        api_key_id=api_key_id,
        api_key_name=api_key_name,
        event_type=outcome.event_type,
        path=request.url.path[:512],
        method=request.method,
        status_code=status_code,
        ip_address=_client_ip(request),
        user_agent=request.headers.get("user-agent", "")[:512] or None,
        request_id=getattr(request.state, "request_id", None),
    )
    try:
        async with factory() as session:
            session.add(row)
            await session.commit()
    except Exception:  # pragma: no cover - audit failures must never break request
        logging.getLogger(__name__).warning("ive.auth.audit_log_failed")


def _client_ip(request: Request) -> str | None:
    """Extract a best-effort client IP, respecting common proxy headers."""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()[:64]
    if request.client is not None:
        return request.client.host[:64]
    return None


# ---------------------------------------------------------------------------
# Route-level dependency (alternative to middleware)
# ---------------------------------------------------------------------------


async def require_api_key(request: Request) -> str:
    """FastAPI dependency that returns the authenticated key name.

    Prefer ``Depends(require_scope(Scope.X))`` for new code — it both
    enforces a scope and returns the full :class:`AuthContext`. This
    legacy helper is kept so existing routes don't break.
    """
    ctx: AuthContext | None = getattr(request.state, "auth", None)
    if ctx is not None:
        return ctx.api_key_name
    # No middleware attached — fall through to legacy env-CSV check.
    settings = get_settings()
    api_key = request.headers.get(settings.api_key_header)
    if not api_key or api_key not in settings.api_keys_set:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
    return api_key
