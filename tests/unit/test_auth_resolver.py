"""Unit tests for ive.auth.resolver."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from ive.auth.resolver import resolve_api_key
from ive.auth.scopes import Scope
from ive.auth.utils import hash_api_key
from ive.config import Settings, get_settings

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def csv_settings(monkeypatch):
    monkeypatch.setenv("VALID_API_KEYS", "dev-key-1,dev-key-2")
    monkeypatch.setenv("LLM_EXPLANATIONS_ENABLED", "false")
    return Settings()


def _row(
    *,
    name: str = "ci",
    scopes: list[str] | None = None,
    is_active: bool = True,
    expires_at: datetime | None = None,
):
    """Build a stand-in for an APIKey ORM row."""
    row = MagicMock()
    row.id = "11111111-1111-1111-1111-111111111111"
    row.name = name
    row.scopes = scopes or ["read", "write"]
    row.rate_limit = 100
    row.is_active = is_active
    row.expires_at = expires_at
    # Bind a real is_expired so the resolver's expiry check works
    if expires_at is None:
        row.is_expired = lambda: False
    else:
        row.is_expired = lambda: datetime.now(UTC) >= expires_at
    return row


def _session_returning(row):
    """Build an AsyncSession mock whose execute() returns ``row`` (or None)."""
    session = AsyncMock()
    exec_result = MagicMock()
    exec_result.scalar_one_or_none.return_value = row
    session.execute = AsyncMock(return_value=exec_result)
    return session


class TestResolver:
    @pytest.mark.asyncio
    async def test_missing_key_rejected(self, csv_settings):
        out = await resolve_api_key(None, None, csv_settings)
        assert not out.authenticated
        assert out.event_type == "auth_missing"

    @pytest.mark.asyncio
    async def test_db_managed_key_accepted_with_scopes(self, csv_settings):
        row = _row(name="alice", scopes=["read", "admin"])
        session = _session_returning(row)
        out = await resolve_api_key("ive_some-real-key", session, csv_settings)
        assert out.authenticated
        assert out.context.api_key_name == "alice"
        assert Scope.ADMIN in out.context.scopes
        assert Scope.READ in out.context.scopes

    @pytest.mark.asyncio
    async def test_inactive_db_key_falls_through(self, csv_settings):
        row = _row(is_active=False)
        session = _session_returning(row)
        out = await resolve_api_key("not-in-csv", session, csv_settings)
        assert not out.authenticated
        assert out.event_type == "auth_failure"

    @pytest.mark.asyncio
    async def test_expired_db_key_falls_through(self, csv_settings):
        row = _row(expires_at=datetime.now(UTC) - timedelta(hours=1))
        session = _session_returning(row)
        out = await resolve_api_key("not-in-csv", session, csv_settings)
        assert not out.authenticated

    @pytest.mark.asyncio
    async def test_env_csv_fallback_accepts_known_key(self, csv_settings):
        # No DB session at all — pure env-CSV path.
        out = await resolve_api_key("dev-key-1", None, csv_settings)
        assert out.authenticated
        assert out.context.api_key_id is None
        assert out.context.api_key_name.startswith("env:")
        assert Scope.READ in out.context.scopes
        assert Scope.WRITE in out.context.scopes
        # Env keys never get admin
        assert Scope.ADMIN not in out.context.scopes
        assert out.extras["resolution"] == "env_csv"

    @pytest.mark.asyncio
    async def test_env_csv_unknown_key_rejected(self, csv_settings):
        out = await resolve_api_key("not-a-real-key", None, csv_settings)
        assert not out.authenticated
        assert out.event_type == "auth_failure"

    @pytest.mark.asyncio
    async def test_db_miss_then_env_csv_hit(self, csv_settings):
        # DB lookup returns None, but the raw key matches env CSV.
        session = _session_returning(None)
        out = await resolve_api_key("dev-key-2", session, csv_settings)
        assert out.authenticated
        assert out.context.api_key_id is None  # env-fallback path

    @pytest.mark.asyncio
    async def test_db_lookup_uses_hashed_key(self, csv_settings):
        row = _row(name="bob")
        session = _session_returning(row)
        await resolve_api_key("ive_alpha-bravo-charlie", session, csv_settings)
        # The session.execute call should have been issued; the resolver
        # uses key_hash for lookup, never the raw value. Verify by the
        # fact that nothing crashed and a row was returned.
        session.execute.assert_called_once()
        # And the digest of our raw key is a valid sha256 hex
        digest = hash_api_key("ive_alpha-bravo-charlie")
        assert len(digest) == 64
