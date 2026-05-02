"""Unit tests for LVAnnotationRepo with a mocked AsyncSession."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from ive.db.repositories.lv_annotation_repo import LVAnnotationRepo

pytestmark = pytest.mark.unit


def _row(*, body: str = "note", api_key_name: str = "alice", lv_id=None):
    r = MagicMock()
    r.id = uuid.uuid4()
    r.latent_variable_id = lv_id or uuid.uuid4()
    r.api_key_id = uuid.uuid4()
    r.api_key_name = api_key_name
    r.body = body
    return r


def _session(rows=None, get_value=None):
    session = AsyncMock()
    session.flush = AsyncMock(return_value=None)
    session.add = MagicMock()
    session.delete = AsyncMock(return_value=None)
    session.get = AsyncMock(return_value=get_value)
    result = MagicMock()
    result.scalars.return_value.all.return_value = rows or []
    session.execute = AsyncMock(return_value=result)
    return session


class TestListForLv:
    @pytest.mark.asyncio
    async def test_returns_rows(self):
        rows = [_row(body="a"), _row(body="b")]
        repo = LVAnnotationRepo(_session(rows=rows))
        out = await repo.list_for_lv(uuid.uuid4())
        assert len(out) == 2

    @pytest.mark.asyncio
    async def test_empty_returns_empty_list(self):
        repo = LVAnnotationRepo(_session(rows=[]))
        out = await repo.list_for_lv(uuid.uuid4())
        assert list(out) == []


class TestCreate:
    @pytest.mark.asyncio
    async def test_create_persists_with_author_metadata(self):
        session = _session()
        repo = LVAnnotationRepo(session)
        lv_id = uuid.uuid4()
        key_id = uuid.uuid4()
        row = await repo.create(
            latent_variable_id=lv_id,
            body="hello",
            api_key_id=key_id,
            api_key_name="alice",
        )
        assert row.body == "hello"
        assert row.api_key_id == key_id
        assert row.api_key_name == "alice"
        assert row.latent_variable_id == lv_id
        session.add.assert_called_once()
        session.flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_create_allows_anonymous(self):
        session = _session()
        repo = LVAnnotationRepo(session)
        row = await repo.create(
            latent_variable_id=uuid.uuid4(),
            body="x",
        )
        assert row.api_key_id is None
        assert row.api_key_name is None


class TestUpdate:
    @pytest.mark.asyncio
    async def test_update_existing_row(self):
        existing = _row(body="old")
        session = _session(get_value=existing)
        repo = LVAnnotationRepo(session)
        out = await repo.update_body(existing.id, "new")
        assert out is existing
        assert existing.body == "new"

    @pytest.mark.asyncio
    async def test_update_missing_returns_none(self):
        session = _session(get_value=None)
        repo = LVAnnotationRepo(session)
        out = await repo.update_body(uuid.uuid4(), "x")
        assert out is None


class TestDelete:
    @pytest.mark.asyncio
    async def test_delete_existing_returns_true(self):
        existing = _row()
        session = _session(get_value=existing)
        repo = LVAnnotationRepo(session)
        ok = await repo.delete(existing.id)
        assert ok is True
        session.delete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_missing_returns_false(self):
        session = _session(get_value=None)
        repo = LVAnnotationRepo(session)
        ok = await repo.delete(uuid.uuid4())
        assert ok is False
