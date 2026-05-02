"""Scope model and authorization primitives.

Three scopes (per plan §155):

- ``read``  — list/get datasets, experiments, latent variables, feedback
- ``write`` — create/upload, run experiments, submit feedback, annotations
- ``admin`` — manage api keys, rotate keys, force-regenerate, share-token mgmt

``admin`` implies ``read`` + ``write`` for convenience but is checked
explicitly so admin-only endpoints don't accidentally fall through.

Plan reference: §47, §113, §155.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from fastapi import HTTPException, Request, status


class Scope(str, Enum):
    """Authorization scopes — string-typed so they serialize cleanly."""

    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


@dataclass
class AuthContext:
    """Authenticated identity attached to ``request.state.auth``.

    Carries enough information for downstream code to (a) make scope
    decisions, (b) write audit-log rows, (c) attribute writes to a user
    identifier.
    """

    api_key_id: str | None  # UUID as string when DB-resolved, None for env-only keys
    api_key_name: str
    scopes: frozenset[Scope]
    rate_limit: int = 100

    def has_scope(self, scope: Scope) -> bool:
        """Return True when this context grants ``scope``. ``admin`` implies all."""
        return scope in self.scopes or Scope.ADMIN in self.scopes


@dataclass
class AuthOutcome:
    """Result of running ``resolve_api_key``.

    Either ``context`` is set (authenticated) or ``error`` is set (rejection
    reason, used both for the 401 response and the audit-log entry).
    """

    context: AuthContext | None = None
    error: str | None = None
    event_type: str = "auth_success"
    extras: dict[str, str] = field(default_factory=dict)

    @property
    def authenticated(self) -> bool:
        return self.context is not None


def require_scope(scope: Scope) -> Callable[[Request], AuthContext]:
    """FastAPI dependency factory: enforce ``scope`` on a route.

    Usage::

        @router.post("/api-keys", dependencies=[Depends(require_scope(Scope.ADMIN))])
        async def create_key(...): ...
    """

    async def _check(request: Request) -> AuthContext:
        ctx: AuthContext | None = getattr(request.state, "auth", None)
        if ctx is None:
            # Middleware should have attached auth; if not, treat as 401.
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required.",
            )
        if not ctx.has_scope(scope):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Scope {scope.value!r} required.",
            )
        return ctx

    return _check
