"""Unit tests for the column-metadata endpoints with a stub auth context."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from ive.api.v1.dependencies import get_db
from ive.api.v1.endpoints.column_metadata import router as column_metadata_router
from ive.auth.scopes import AuthContext, Scope

pytestmark = pytest.mark.unit


def _ctx(*scopes: Scope) -> AuthContext:
    return AuthContext(
        api_key_id=None,
        api_key_name="test",
        scopes=frozenset(scopes),
        rate_limit=100,
    )


class _AttachAuth(BaseHTTPMiddleware):
    def __init__(self, app, ctx: AuthContext):
        super().__init__(app)
        self._ctx = ctx

    async def dispatch(self, request: Request, call_next):
        request.state.auth = self._ctx
        return await call_next(request)


def _row(name: str, sensitivity: str = "non_public"):
    """Build a dict that satisfies ColumnMetadataResponse via from_attributes."""
    r = MagicMock()
    r.id = uuid.uuid4()
    r.dataset_id = uuid.UUID("11111111-1111-1111-1111-111111111111")
    r.column_name = name
    r.sensitivity = sensitivity
    r.created_at = datetime.now(UTC)
    r.updated_at = datetime.now(UTC)
    return r


def _make_app(ctx: AuthContext, *, list_rows: list, scalar_value=None):
    app = FastAPI()
    app.add_middleware(_AttachAuth, ctx=ctx)
    app.include_router(
        column_metadata_router,
        prefix="/api/v1/datasets/{dataset_id}/columns",
    )

    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock(return_value=None)

    exec_result = MagicMock()
    exec_result.scalars.return_value.all.return_value = list_rows
    exec_result.scalar_one_or_none.return_value = scalar_value
    session.execute = AsyncMock(return_value=exec_result)

    async def _override_get_db():
        yield session

    app.dependency_overrides[get_db] = _override_get_db
    return app, session


class TestList:
    def test_list_requires_read_scope(self):
        app, _ = _make_app(_ctx(), list_rows=[])
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.get(f"/api/v1/datasets/{uuid.uuid4()}/columns/")
            assert r.status_code == 403

    def test_list_returns_metadata(self):
        rows = [_row("age", "public"), _row("ssn", "non_public")]
        app, _ = _make_app(_ctx(Scope.READ), list_rows=rows)
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.get(f"/api/v1/datasets/{uuid.uuid4()}/columns/")
            assert r.status_code == 200
            body = r.json()
            assert body["total"] == 2
            assert body["public_count"] == 1
            names = sorted(item["column_name"] for item in body["items"])
            assert names == ["age", "ssn"]


class TestPut:
    def test_put_requires_write_scope(self):
        app, _ = _make_app(_ctx(Scope.READ), list_rows=[_row("age")])
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.put(
                f"/api/v1/datasets/{uuid.uuid4()}/columns/",
                json={"updates": [{"column_name": "age", "sensitivity": "public"}]},
            )
            assert r.status_code == 403

    def test_put_404_when_no_rows(self):
        app, _ = _make_app(_ctx(Scope.WRITE), list_rows=[])
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.put(
                f"/api/v1/datasets/{uuid.uuid4()}/columns/",
                json={"updates": [{"column_name": "age", "sensitivity": "public"}]},
            )
            assert r.status_code == 404

    def test_put_updates_and_returns_full_list(self):
        rows = [_row("age", "non_public"), _row("city", "public")]
        app, _ = _make_app(_ctx(Scope.WRITE), list_rows=rows)
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.put(
                f"/api/v1/datasets/{uuid.uuid4()}/columns/",
                json={"updates": [{"column_name": "age", "sensitivity": "public"}]},
            )
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["total"] == 2

    def test_put_rejects_invalid_sensitivity(self):
        app, _ = _make_app(_ctx(Scope.WRITE), list_rows=[_row("age")])
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.put(
                f"/api/v1/datasets/{uuid.uuid4()}/columns/",
                json={"updates": [{"column_name": "age", "sensitivity": "weird"}]},
            )
            assert r.status_code == 422

    def test_put_rejects_extra_fields(self):
        app, _ = _make_app(_ctx(Scope.WRITE), list_rows=[_row("age")])
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.put(
                f"/api/v1/datasets/{uuid.uuid4()}/columns/",
                json={"updates": [], "extra_field": True},
            )
            assert r.status_code == 422
