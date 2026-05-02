"""Unit tests for the LV annotation endpoints with stub auth context."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from ive.api.v1.dependencies import get_db
from ive.api.v1.endpoints.lv_annotations import router as lv_annotations_router
from ive.auth.scopes import AuthContext, Scope

pytestmark = pytest.mark.unit


def _ctx(*scopes: Scope, key_id: uuid.UUID | None = None, name: str = "alice") -> AuthContext:
    # AuthContext.api_key_id is `str | None` (UUID-as-string when DB-resolved).
    resolved_key = key_id or uuid.uuid4()
    return AuthContext(
        api_key_id=str(resolved_key),
        api_key_name=name,
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


def _row(
    *,
    lv_id: uuid.UUID,
    body: str = "note",
    api_key_id: uuid.UUID | None = None,
    api_key_name: str = "alice",
):
    r = MagicMock()
    r.id = uuid.uuid4()
    r.latent_variable_id = lv_id
    r.body = body
    r.api_key_id = api_key_id
    r.api_key_name = api_key_name
    r.created_at = datetime.now(UTC)
    r.updated_at = datetime.now(UTC)
    return r


def _make_app(ctx: AuthContext, *, list_rows: list, scalar_value=None):
    app = FastAPI()
    app.add_middleware(_AttachAuth, ctx=ctx)
    app.include_router(
        lv_annotations_router,
        prefix="/api/v1/latent-variables/{lv_id}/annotations",
    )

    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock(return_value=None)
    session.commit = AsyncMock(return_value=None)
    session.delete = AsyncMock(return_value=None)
    session.get = AsyncMock(return_value=scalar_value)

    exec_result = MagicMock()
    exec_result.scalars.return_value.all.return_value = list_rows
    session.execute = AsyncMock(return_value=exec_result)

    async def _override_get_db():
        yield session

    app.dependency_overrides[get_db] = _override_get_db
    return app, session


class TestList:
    def test_list_requires_read_scope(self):
        lv_id = uuid.uuid4()
        app, _ = _make_app(_ctx(), list_rows=[_row(lv_id=lv_id)])
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.get(f"/api/v1/latent-variables/{lv_id}/annotations/")
            assert r.status_code == 403

    def test_list_returns_items(self):
        lv_id = uuid.uuid4()
        rows = [_row(lv_id=lv_id, body="first"), _row(lv_id=lv_id, body="second")]
        app, _ = _make_app(_ctx(Scope.READ), list_rows=rows)
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.get(f"/api/v1/latent-variables/{lv_id}/annotations/")
            assert r.status_code == 200
            body = r.json()
            assert body["total"] == 2
            assert {item["body"] for item in body["items"]} == {"first", "second"}


class TestCreate:
    def test_create_requires_write_scope(self):
        lv_id = uuid.uuid4()
        app, _ = _make_app(_ctx(Scope.READ), list_rows=[])
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.post(
                f"/api/v1/latent-variables/{lv_id}/annotations/",
                json={"body": "hello"},
            )
            assert r.status_code == 403

    def test_create_201(self):
        lv_id = uuid.uuid4()
        app, session = _make_app(_ctx(Scope.WRITE), list_rows=[])
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.post(
                f"/api/v1/latent-variables/{lv_id}/annotations/",
                json={"body": "hello"},
            )
            assert r.status_code == 201, r.text
            assert r.json()["body"] == "hello"
            session.commit.assert_awaited()

    def test_create_validates_body_length(self):
        lv_id = uuid.uuid4()
        app, _ = _make_app(_ctx(Scope.WRITE), list_rows=[])
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.post(
                f"/api/v1/latent-variables/{lv_id}/annotations/",
                json={"body": ""},
            )
            assert r.status_code == 422

    def test_create_rejects_extra_fields(self):
        lv_id = uuid.uuid4()
        app, _ = _make_app(_ctx(Scope.WRITE), list_rows=[])
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.post(
                f"/api/v1/latent-variables/{lv_id}/annotations/",
                json={"body": "x", "weird_field": True},
            )
            assert r.status_code == 422


class TestEditDeleteAuthorization:
    def test_non_author_non_admin_cannot_edit(self):
        lv_id = uuid.uuid4()
        author_key = uuid.uuid4()
        existing = _row(lv_id=lv_id, api_key_id=author_key, api_key_name="alice")
        # Different requester (bob) without admin scope
        ctx = _ctx(Scope.WRITE, key_id=uuid.uuid4(), name="bob")
        app, _ = _make_app(ctx, list_rows=[existing], scalar_value=existing)
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.put(
                f"/api/v1/latent-variables/{lv_id}/annotations/{existing.id}",
                json={"body": "edit"},
            )
            assert r.status_code == 403

    def test_author_can_edit_own(self):
        lv_id = uuid.uuid4()
        author_key = uuid.uuid4()
        existing = _row(lv_id=lv_id, api_key_id=author_key, api_key_name="alice")
        ctx = _ctx(Scope.WRITE, key_id=author_key, name="alice")
        app, _ = _make_app(ctx, list_rows=[existing], scalar_value=existing)
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.put(
                f"/api/v1/latent-variables/{lv_id}/annotations/{existing.id}",
                json={"body": "updated"},
            )
            assert r.status_code == 200
            assert r.json()["body"] == "updated"

    def test_admin_can_edit_other_author(self):
        lv_id = uuid.uuid4()
        author_key = uuid.uuid4()
        existing = _row(lv_id=lv_id, api_key_id=author_key, api_key_name="alice")
        ctx = _ctx(Scope.WRITE, Scope.ADMIN, key_id=uuid.uuid4(), name="ops")
        app, _ = _make_app(ctx, list_rows=[existing], scalar_value=existing)
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.put(
                f"/api/v1/latent-variables/{lv_id}/annotations/{existing.id}",
                json={"body": "moderated"},
            )
            assert r.status_code == 200

    def test_anonymous_annotation_anyone_can_edit(self):
        # Annotations created without an api_key_id (legacy / system) have
        # no author binding — any write-scope user can edit.
        lv_id = uuid.uuid4()
        existing = _row(lv_id=lv_id, api_key_id=None, api_key_name=None)
        ctx = _ctx(Scope.WRITE, key_id=uuid.uuid4(), name="bob")
        app, _ = _make_app(ctx, list_rows=[existing], scalar_value=existing)
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.put(
                f"/api/v1/latent-variables/{lv_id}/annotations/{existing.id}",
                json={"body": "edit"},
            )
            assert r.status_code == 200

    def test_404_when_annotation_not_on_lv(self):
        lv_id = uuid.uuid4()
        other_lv = uuid.uuid4()
        existing = _row(lv_id=other_lv, api_key_id=uuid.uuid4(), api_key_name="alice")
        ctx = _ctx(Scope.WRITE)
        app, _ = _make_app(ctx, list_rows=[], scalar_value=existing)
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.put(
                f"/api/v1/latent-variables/{lv_id}/annotations/{existing.id}",
                json={"body": "edit"},
            )
            assert r.status_code == 404


class TestDelete:
    def test_author_can_delete(self):
        lv_id = uuid.uuid4()
        author_key = uuid.uuid4()
        existing = _row(lv_id=lv_id, api_key_id=author_key, api_key_name="alice")
        ctx = _ctx(Scope.WRITE, key_id=author_key, name="alice")
        app, _ = _make_app(ctx, list_rows=[existing], scalar_value=existing)
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.delete(
                f"/api/v1/latent-variables/{lv_id}/annotations/{existing.id}"
            )
            assert r.status_code == 204

    def test_non_author_cannot_delete(self):
        lv_id = uuid.uuid4()
        existing = _row(lv_id=lv_id, api_key_id=uuid.uuid4())
        ctx = _ctx(Scope.WRITE, key_id=uuid.uuid4(), name="bob")
        app, _ = _make_app(ctx, list_rows=[existing], scalar_value=existing)
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.delete(
                f"/api/v1/latent-variables/{lv_id}/annotations/{existing.id}"
            )
            assert r.status_code == 403
