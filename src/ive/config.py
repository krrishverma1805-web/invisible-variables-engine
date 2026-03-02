"""
IVE Configuration Module.

Centralises all application configuration using Pydantic Settings.
Values are read from environment variables (or a .env file) and validated
at startup. This is the single source of truth for all config.

Usage:
    from ive.config import get_settings
    settings = get_settings()
    print(settings.database_url)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AnyUrl, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """PostgreSQL connection settings."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    database_url: str = Field(
        default="postgresql+asyncpg://ive:ivepassword@localhost:5432/ive_db",
        description="Async SQLAlchemy DSN for PostgreSQL.",
    )
    database_pool_size: int = Field(default=10, ge=1, le=100)
    database_max_overflow: int = Field(default=20, ge=0, le=100)
    database_pool_timeout: int = Field(default=30, ge=5)

    @field_validator("database_url")
    @classmethod
    def ensure_async_driver(cls, v: str) -> str:
        """Ensure asyncpg driver is used for async operations."""
        if v.startswith("postgresql://"):
            return v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v


class RedisSettings(BaseSettings):
    """Redis connection settings."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis DSN for Celery broker and cache.",
    )


class AuthSettings(BaseSettings):
    """Authentication settings."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    api_key_header: str = Field(default="X-API-Key")
    valid_api_keys: str = Field(
        default="dev-key-1",
        description="Comma-separated list of valid API keys.",
    )

    def get_api_keys(self) -> set[str]:
        """Return parsed set of valid API keys."""
        return {k.strip() for k in self.valid_api_keys.split(",") if k.strip()}


class RateLimitSettings(BaseSettings):
    """Rate limiting settings."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    rate_limit_requests: int = Field(default=100, ge=1)
    rate_limit_window: int = Field(default=60, ge=1, description="Window in seconds.")


class MLSettings(BaseSettings):
    """Machine learning pipeline settings."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    default_cv_folds: int = Field(default=5, ge=2, le=20)
    default_test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    random_seed: int = Field(default=42)
    max_features: int = Field(default=100, ge=1)
    min_cluster_size: int = Field(default=10, ge=2)
    shap_sample_size: int = Field(default=500, ge=50)


class StorageSettings(BaseSettings):
    """Artifact storage settings."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    artifact_store_type: Literal["local", "s3"] = Field(default="local")
    artifact_base_dir: str = Field(default="/app/artifacts")

    # S3 settings (only used when artifact_store_type == "s3")
    aws_access_key_id: str | None = Field(default=None)
    aws_secret_access_key: SecretStr | None = Field(default=None)
    aws_region: str = Field(default="us-east-1")
    s3_bucket_name: str | None = Field(default=None)


class Settings(
    DatabaseSettings,
    RedisSettings,
    AuthSettings,
    RateLimitSettings,
    MLSettings,
    StorageSettings,
):
    """
    Master settings class for the IVE application.

    All environment variables are read at startup. Missing required variables
    will raise a ValidationError immediately, preventing silent misconfigurations.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    env: Literal["development", "staging", "production"] = Field(default="development")
    secret_key: SecretStr = Field(default=SecretStr("change-me-in-production"))
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO"
    )
    debug: bool = Field(default=False)
    api_port: int = Field(default=8000, ge=1024, le=65535)

    # Sentry (optional)
    sentry_dsn: str | None = Field(default=None)
    enable_metrics: bool = Field(default=False)

    @property
    def is_production(self) -> bool:
        """Return True if running in production environment."""
        return self.env == "production"

    @property
    def sync_database_url(self) -> str:
        """Return a synchronous DSN for tools like Alembic."""
        return self.database_url.replace("postgresql+asyncpg://", "postgresql://", 1)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached Settings instance.

    Using lru_cache ensures we only read env vars once per process,
    which is important for performance. In tests, call get_settings.cache_clear()
    before each test to reset the cache.
    """
    return Settings()
