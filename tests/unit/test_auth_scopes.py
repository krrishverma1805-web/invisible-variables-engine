"""Unit tests for ive.auth.scopes."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from ive.auth.scopes import AuthContext, Scope, require_scope

pytestmark = pytest.mark.unit


def _ctx(*scopes: Scope) -> AuthContext:
    return AuthContext(
        api_key_id=None,
        api_key_name="test",
        scopes=frozenset(scopes),
        rate_limit=100,
    )


class TestScopeEnum:
    def test_string_value(self):
        assert Scope.READ == "read"
        assert Scope.WRITE == "write"
        assert Scope.ADMIN == "admin"

    def test_round_trip(self):
        assert Scope("read") is Scope.READ


class TestAuthContext:
    def test_explicit_scope(self):
        ctx = _ctx(Scope.READ)
        assert ctx.has_scope(Scope.READ)
        assert not ctx.has_scope(Scope.WRITE)

    def test_admin_implies_all(self):
        ctx = _ctx(Scope.ADMIN)
        assert ctx.has_scope(Scope.READ)
        assert ctx.has_scope(Scope.WRITE)
        assert ctx.has_scope(Scope.ADMIN)


class TestRequireScope:
    @pytest.mark.asyncio
    async def test_passes_when_scope_present(self):
        request = MagicMock()
        request.state.auth = _ctx(Scope.READ)
        check = require_scope(Scope.READ)
        result = await check(request)
        assert result.has_scope(Scope.READ)

    @pytest.mark.asyncio
    async def test_passes_when_admin(self):
        request = MagicMock()
        request.state.auth = _ctx(Scope.ADMIN)
        check = require_scope(Scope.WRITE)
        result = await check(request)
        assert result is request.state.auth

    @pytest.mark.asyncio
    async def test_403_when_scope_missing(self):
        request = MagicMock()
        request.state.auth = _ctx(Scope.READ)
        check = require_scope(Scope.ADMIN)
        with pytest.raises(HTTPException) as info:
            await check(request)
        assert info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_401_when_no_auth_context(self):
        request = MagicMock()
        request.state.auth = None  # no middleware ran
        check = require_scope(Scope.READ)
        with pytest.raises(HTTPException) as info:
            await check(request)
        assert info.value.status_code == 401
