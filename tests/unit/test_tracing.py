"""Tests for the OTel tracing scaffolding (plan §117 + §150)."""

from __future__ import annotations

import pytest

from ive.observability import tracing


@pytest.fixture(autouse=True)
def _reset_tracing():
    tracing.reset_for_tests()
    yield
    tracing.reset_for_tests()


class TestNoOpWhenDisabled:
    def test_install_returns_false_when_setting_off(self, monkeypatch):
        from ive.config import get_settings

        monkeypatch.delenv("ENABLE_TRACING", raising=False)
        get_settings.cache_clear()
        assert tracing.install_tracing() is False

    def test_trace_span_yields_none_when_disabled(self, monkeypatch):
        from ive.config import get_settings

        monkeypatch.delenv("ENABLE_TRACING", raising=False)
        get_settings.cache_clear()
        tracing.install_tracing()
        with tracing.trace_span("ive.test", {"x": "y"}) as span:
            assert span is None

    def test_trace_span_propagates_exceptions(self, monkeypatch):
        from ive.config import get_settings

        monkeypatch.delenv("ENABLE_TRACING", raising=False)
        get_settings.cache_clear()
        tracing.install_tracing()
        with pytest.raises(RuntimeError):
            with tracing.trace_span("ive.test"):
                raise RuntimeError("boom")


class TestInstallIdempotent:
    def test_install_twice_is_safe(self, monkeypatch):
        from ive.config import get_settings

        monkeypatch.delenv("ENABLE_TRACING", raising=False)
        get_settings.cache_clear()
        first = tracing.install_tracing()
        second = tracing.install_tracing()
        # Both report False (disabled) and don't blow up.
        assert first == second
