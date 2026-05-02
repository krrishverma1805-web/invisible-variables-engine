"""Unit tests for DatasetColumnMetadataRepo with a mocked AsyncSession."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from ive.db.repositories.dataset_column_metadata_repo import (
    DatasetColumnMetadataRepo,
)

pytestmark = pytest.mark.unit


def _row(name: str, sensitivity: str = "non_public"):
    r = MagicMock()
    r.column_name = name
    r.sensitivity = sensitivity
    return r


def _session_with_rows(rows: list, scalar_one_value=None):
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock(return_value=None)

    list_result = MagicMock()
    list_result.scalars.return_value.all.return_value = rows
    list_result.scalar_one_or_none.return_value = scalar_one_value
    list_result.all.return_value = [(r.column_name,) for r in rows if r.sensitivity == "public"]

    session.execute = AsyncMock(return_value=list_result)
    return session


class TestListAndGet:
    @pytest.mark.asyncio
    async def test_list_for_dataset_returns_rows(self):
        rows = [_row("age"), _row("income")]
        session = _session_with_rows(rows)
        repo = DatasetColumnMetadataRepo(session)
        out = await repo.list_for_dataset(uuid.uuid4())
        assert len(out) == 2

    @pytest.mark.asyncio
    async def test_get_returns_match(self):
        existing = _row("ssn")
        session = _session_with_rows([], scalar_one_value=existing)
        repo = DatasetColumnMetadataRepo(session)
        out = await repo.get(uuid.uuid4(), "ssn")
        assert out is existing


class TestBulkCreateDefault:
    @pytest.mark.asyncio
    async def test_creates_one_per_new_column(self):
        session = _session_with_rows([])  # no existing rows
        repo = DatasetColumnMetadataRepo(session)
        created = await repo.bulk_create_default(uuid.uuid4(), ["a", "b", "c"])
        assert len(created) == 3
        assert all(r.sensitivity == "non_public" for r in created)

    @pytest.mark.asyncio
    async def test_skips_existing_columns(self):
        existing = [_row("a"), _row("b")]
        session = _session_with_rows(existing)
        repo = DatasetColumnMetadataRepo(session)
        created = await repo.bulk_create_default(uuid.uuid4(), ["a", "b", "c"])
        # Only "c" should be created
        assert len(created) == 1
        assert created[0].column_name == "c"

    @pytest.mark.asyncio
    async def test_no_flush_when_nothing_new(self):
        existing = [_row("a")]
        session = _session_with_rows(existing)
        repo = DatasetColumnMetadataRepo(session)
        created = await repo.bulk_create_default(uuid.uuid4(), ["a"])
        assert created == []
        session.flush.assert_not_called()


class TestBulkSet:
    @pytest.mark.asyncio
    async def test_updates_existing_rows_only(self):
        rows = [_row("age", "non_public"), _row("income", "non_public")]
        session = _session_with_rows(rows)
        repo = DatasetColumnMetadataRepo(session)
        updated = await repo.bulk_set(
            uuid.uuid4(),
            {"age": "public", "income": "public", "missing_col": "public"},
        )
        # Only existing rows updated
        assert len(updated) == 2
        assert all(r.sensitivity == "public" for r in updated)

    @pytest.mark.asyncio
    async def test_unchanged_sensitivity_not_in_changed(self):
        rows = [_row("age", "public"), _row("income", "non_public")]
        session = _session_with_rows(rows)
        repo = DatasetColumnMetadataRepo(session)
        updated = await repo.bulk_set(
            uuid.uuid4(),
            {"age": "public", "income": "public"},
        )
        assert len(updated) == 1
        assert updated[0].column_name == "income"


class TestPublicColumnNames:
    @pytest.mark.asyncio
    async def test_returns_only_public(self):
        rows = [_row("age", "public"), _row("ssn", "non_public"), _row("city", "public")]
        session = _session_with_rows(rows)
        repo = DatasetColumnMetadataRepo(session)
        out = await repo.public_column_names(uuid.uuid4())
        assert out == {"age", "city"}
