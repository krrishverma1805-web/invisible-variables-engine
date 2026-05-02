"""Unit tests for ive.db.repositories.api_key_repo.

Uses a mock AsyncSession; integration coverage against a real DB is
left to a follow-up integration suite (Postgres-required).
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from ive.auth.utils import hash_api_key
from ive.db.repositories.api_key_repo import APIKeyRepo

pytestmark = pytest.mark.unit


def _session_with_get(row=None):
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock(return_value=None)
    session.get = AsyncMock(return_value=row)
    exec_result = MagicMock()
    exec_result.scalar_one_or_none.return_value = None
    exec_result.scalars.return_value.all.return_value = []
    session.execute = AsyncMock(return_value=exec_result)
    return session


class TestCreate:
    @pytest.mark.asyncio
    async def test_create_returns_raw_key_starting_with_prefix(self):
        session = _session_with_get()
        repo = APIKeyRepo(session)
        row, raw = await repo.create(name="bob", scopes=["read"])
        assert raw.startswith("ive_")
        assert row.name == "bob"
        assert row.scopes == ["read"]
        # The persisted hash matches the raw key we returned
        assert row.key_hash == hash_api_key(raw)

    @pytest.mark.asyncio
    async def test_create_legacy_permissions_mirror_scopes(self):
        session = _session_with_get()
        repo = APIKeyRepo(session)
        row, _raw = await repo.create(name="alice", scopes=["read", "admin"])
        assert row.permissions == {"read": True, "write": False, "admin": True}


class TestRevoke:
    @pytest.mark.asyncio
    async def test_revoke_marks_inactive(self):
        existing = MagicMock()
        existing.is_active = True
        session = _session_with_get(row=existing)
        repo = APIKeyRepo(session)
        ok = await repo.revoke(uuid.uuid4())
        assert ok is True
        assert existing.is_active is False

    @pytest.mark.asyncio
    async def test_revoke_returns_false_when_missing(self):
        session = _session_with_get(row=None)
        repo = APIKeyRepo(session)
        assert await repo.revoke(uuid.uuid4()) is False


class TestRotate:
    @pytest.mark.asyncio
    async def test_rotate_updates_hash(self):
        existing = MagicMock()
        existing.key_hash = "OLDHASH"
        session = _session_with_get(row=existing)
        repo = APIKeyRepo(session)
        out = await repo.rotate(uuid.uuid4())
        assert out is not None
        row, raw = out
        assert row.key_hash == hash_api_key(raw)
        assert row.key_hash != "OLDHASH"
        assert row.last_rotated_at is not None

    @pytest.mark.asyncio
    async def test_rotate_missing_returns_none(self):
        session = _session_with_get(row=None)
        repo = APIKeyRepo(session)
        assert await repo.rotate(uuid.uuid4()) is None
