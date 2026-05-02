"""Unit tests for the LLMSettings sub-config and its production validator.

These don't need DB/Redis — they exercise the Pydantic Settings layer.
"""

from __future__ import annotations

import pytest

from ive.config import Settings, get_settings

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class TestLLMSettingsDefaults:
    def test_disabled_by_default(self, monkeypatch):
        # Strip any inherited env so defaults take effect.
        for k in ("LLM_EXPLANATIONS_ENABLED", "LLM_SELF_HOSTED_MODE", "GROQ_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        settings = Settings()
        assert settings.llm_explanations_enabled is False
        assert settings.llm_self_hosted_mode is False

    def test_default_model_is_pinned(self, monkeypatch):
        monkeypatch.delenv("GROQ_MODEL", raising=False)
        settings = Settings()
        assert settings.groq_model == "llama-3.3-70b-versatile"

    def test_default_temperature_is_zero(self, monkeypatch):
        monkeypatch.delenv("GROQ_TEMPERATURE", raising=False)
        settings = Settings()
        assert settings.groq_temperature == 0.0

    def test_default_redis_db_is_two(self, monkeypatch):
        monkeypatch.delenv("LLM_REDIS_DB", raising=False)
        settings = Settings()
        assert settings.llm_redis_db == 2

    def test_default_data_egress_is_deny(self, monkeypatch):
        monkeypatch.delenv("LLM_DATA_EGRESS_DEFAULT", raising=False)
        settings = Settings()
        assert settings.llm_data_egress_default == "deny"


class TestLLMCacheRedisURL:
    def test_uses_correct_db_index(self, monkeypatch):
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
        monkeypatch.setenv("LLM_REDIS_DB", "2")
        settings = Settings()
        assert settings.llm_cache_redis_url == "redis://localhost:6379/2"

    def test_isolated_from_celery_dbs(self, monkeypatch):
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
        settings = Settings()
        assert settings.celery_broker_url.endswith("/0")
        assert settings.celery_result_backend.endswith("/1")
        assert settings.llm_cache_redis_url.endswith("/2")


class TestProductionValidator:
    def test_production_with_llm_enabled_requires_api_key(self, monkeypatch):
        monkeypatch.setenv("ENV", "production")
        monkeypatch.setenv(
            "SECRET_KEY",
            "a-sufficiently-long-production-secret-key-here",
        )
        monkeypatch.setenv("VALID_API_KEYS", "prodkey1")
        monkeypatch.setenv("LLM_EXPLANATIONS_ENABLED", "true")
        monkeypatch.setenv("GROQ_API_KEY", "")
        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            Settings()

    def test_production_with_llm_disabled_accepts_empty_key(self, monkeypatch):
        monkeypatch.setenv("ENV", "production")
        monkeypatch.setenv(
            "SECRET_KEY",
            "a-sufficiently-long-production-secret-key-here",
        )
        monkeypatch.setenv("VALID_API_KEYS", "prodkey1")
        monkeypatch.setenv("LLM_EXPLANATIONS_ENABLED", "false")
        monkeypatch.setenv("GROQ_API_KEY", "")
        # Should not raise.
        Settings()

    def test_development_with_llm_enabled_accepts_empty_key(self, monkeypatch):
        monkeypatch.setenv("ENV", "development")
        monkeypatch.setenv("LLM_EXPLANATIONS_ENABLED", "true")
        monkeypatch.setenv("GROQ_API_KEY", "")
        # Should not raise (dev permits empty).
        Settings()
