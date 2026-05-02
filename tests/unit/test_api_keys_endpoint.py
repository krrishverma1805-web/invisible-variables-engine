"""Unit tests for the api-keys admin endpoints.

Exercises the FastAPI router with a stub auth-attach middleware so the
real ``require_scope(Scope.ADMIN)`` dependency runs against an authentic
``AuthContext``. No Postgres required.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from ive.api.v1.dependencies import get_db
from ive.api.v1.endpoints.api_keys import router as api_keys_router
from ive.auth.scopes import AuthContext, Scope

pytestmark = pytest.mark.unit


def _ctx(*scopes: Scope) -> AuthContext:
    return AuthContext(
        api_key_id="00000000-0000-0000-0000-000000000001",
        api_key_name="test-admin",
        scopes=frozenset(scopes),
        rate_limit=100,
    )


class _AttachAuthMiddleware(BaseHTTPMiddleware):
    """Test middleware that attaches a fixed AuthContext to every request."""

    def __init__(self, app, ctx: AuthContext):
        super().__init__(app)
        self._ctx = ctx

    async def dispatch(self, request: Request, call_next):
        request.state.auth = self._ctx
        return await call_next(request)


def _make_app(auth_ctx: AuthContext) -> tuple[FastAPI, AsyncMock]:
    """Build a FastAPI app wired to the api-keys router with stub auth.

    Returns ``(app, session_mock)`` so tests can stub repo behaviour.
    """
    app = FastAPI()
    app.add_middleware(_AttachAuthMiddleware, ctx=auth_ctx)
    app.include_router(api_keys_router, prefix="/api/v1/api-keys")

    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock(return_value=None)
    session.get = AsyncMock(return_value=None)
    exec_result = MagicMock()
    exec_result.scalar_one_or_none.return_value = None
    exec_result.scalars.return_value.all.return_value = []
    session.execute = AsyncMock(return_value=exec_result)

    async def _override_get_db():
        yield session

    app.dependency_overrides[get_db] = _override_get_db
    return app, session


class TestAdminScope:
    def test_create_rejected_without_admin(self):
        app, _ = _make_app(_ctx(Scope.READ, Scope.WRITE))
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.post("/api/v1/api-keys/", json={"name": "new-key"})
            assert r.status_code == 403


class TestCreate:
    def test_create_returns_raw_key(self):
        app, session = _make_app(_ctx(Scope.ADMIN))

        # repo.get_by_name -> None (no duplicate)
        # repo.create persists the row; we capture what was added.
        added: list[object] = []
        session.add.side_effect = lambda row: added.append(row)

        # After flush, the new row should have a generated id/created_at;
        # AsyncMock by default leaves attributes that were set by the
        # SQLAlchemy ORM as MagicMocks. We populate them ourselves to
        # simulate the post-flush state.
        async def _flush():
            for row in added:
                row.id = uuid.uuid4()
                from datetime import UTC, datetime
                row.created_at = datetime.now(UTC)
                row.last_used_at = None
                row.last_rotated_at = None
                row.expires_at = None
                row.created_by = "test-admin"

        session.flush = AsyncMock(side_effect=_flush)

        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.post(
                "/api/v1/api-keys/",
                json={"name": "alice", "scopes": ["read", "write"]},
            )
            assert r.status_code == 201, r.text
            body = r.json()
            assert body["name"] == "alice"
            assert "raw_key" in body
            assert body["raw_key"].startswith("ive_")
            assert sorted(body["scopes"]) == ["read", "write"]


class TestList:
    def test_list_requires_admin(self):
        app, _ = _make_app(_ctx(Scope.READ))
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.get("/api/v1/api-keys/")
            assert r.status_code == 403

    def test_list_returns_empty(self):
        app, _ = _make_app(_ctx(Scope.ADMIN))
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.get("/api/v1/api-keys/")
            assert r.status_code == 200
            assert r.json() == {"items": [], "total": 0}


class TestRevoke:
    def test_revoke_404_when_missing(self):
        app, _ = _make_app(_ctx(Scope.ADMIN))
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.delete(f"/api/v1/api-keys/{uuid.uuid4()}")
            assert r.status_code == 404
